"""The ALSTM-TTM network: attention-LSTM encoder with a t-mixture output layer."""

import torch
import torch.nn as nn

from .layers import Attention, CustomLayer_TTM_Corr, chunks_for

__all__ = ["ALSTM_TTM_Model", "init_weights"]


class ALSTM_TTM_Model(nn.Module):
    """Attention-LSTM that outputs a two-component Student-t mixture return distribution.

    A two-layer LSTM with attention pooling feeds dense layers whose output is mapped by
    ``CustomLayer_TTM_Corr`` to the mixture parameters. The two degrees of freedom are
    learned scalars ``nu = 4 + softplus(u)``, so they stay above four.

    The fixed internal sizes are: first LSTM ``hidden_size`` units (two layers), second
    LSTM 24 units, attention over those 24 units, dense layers 24 -> 48 (ReLU) ->
    ``chunks_for(p)``.

    Args:
        input_size: Number of input features per time step (the predictors fed to the
            LSTM at each look-back day). Set by your data.
        hidden_size: Units of the first (two-layer) LSTM. The paper's notebook uses 48.
        num_layers: Layers of the first LSTM. The first LSTM is built with two layers,
            so use 2 (any other value fails with a shape error).
        p: Number of assets, the dimension of the return vector being modelled, for
            example 26 for the U.S. universe and 9 for the Japan universe.
        mu1_prior: Anchor for the mean of the first mixture component; see
            ``CustomLayer_TTM_Corr``. Compute it from your own training returns.
        nu_1_initial: Starting value of the unconstrained trainable value ``u`` behind the
            first component's degrees of freedom ``nu = 4 + softplus(u)``, where
            ``softplus(z) = ln(1 + e^z)``; it is not the degrees of freedom themselves
            (5 gives nu = 9.007, 10 gives nu = 14.00). The paper's training draws it from
            uniform(3, 20), which gives nu between 7.05 and 24.0. Default 5.
        nu_2_initial: The same for the second component. Default 5.
    """
    def __init__(self, input_size, hidden_size, num_layers, p, mu1_prior, nu_1_initial=5, nu_2_initial=5):
        super(ALSTM_TTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # nu_1 and nu_2 are reparameterised through a softplus so each stays above four.
        self.raw_nu1 = nn.Parameter(torch.tensor([nu_1_initial], dtype=torch.float64)) # learnable parameters
        self.raw_nu2 = nn.Parameter(torch.tensor([nu_2_initial], dtype=torch.float64)) # learnable parameters

        # nu_1 and nu_2 are not used in forward (raw_nu1 and raw_nu2 are); they are kept
        # so that the trained checkpoints load with strict=True.
        self.nu_1 = nn.Parameter(torch.tensor([nu_1_initial], dtype = torch.float64))
        self.nu_2 = nn.Parameter(torch.tensor([nu_2_initial], dtype = torch.float64))
        self.lstm1 = torch.nn.LSTM(input_size = input_size, hidden_size=self.hidden_size, num_layers=2, batch_first = True) # num_layers tells how many LSTM layers are there.
        self.lstm2 = torch.nn.LSTM(input_size=self.hidden_size, hidden_size=24, batch_first = True)
        self.attention = Attention(hidden_size=24)
        self.dense1 = torch.nn.Linear(24, 48)
        self.dense2 = torch.nn.Linear(48, chunks_for(p))
        self.custom_layer = CustomLayer_TTM_Corr(p, mu1_prior)

    def forward(self, x):
        """Run the network on a batch of look-back windows.

        Args:
            x: Predictors, shape ``(batch, sequence_length, input_size)``; use float64
                (call ``model.double()``) to match the paper's computations.

        Returns:
            ``(distribution_outputs, attn_weights, nu_1, nu_2)``: the distribution
            parameters ``(batch, chunks_for(p))`` (pass them to ``output_extraction_TTM``
            or to the loss), the attention weights over the look-back days
            ``(batch, sequence_length)``, and the two degrees of freedom, each a tensor
            of shape ``(1,)`` shared by every sample. Use these returned values: the
            attributes ``model.nu_1`` and ``model.nu_2`` are not used by the forward pass.
        """
        # Degrees of freedom are constrained to nu >= 4 via 4 + softplus(.), giving the
        # range [4, inf); the +4 floor keeps each Student-t component's kurtosis finite
        # (kurtosis exists only for nu > 4).
        nu_1 = 4 + torch.nn.functional.softplus(self.raw_nu1)
        nu_2 = 4 + torch.nn.functional.softplus(self.raw_nu2)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)

        lstm_out, _ = self.lstm1(x, (h0,c0))
        lstm_out, _ = self.lstm2(lstm_out)
        attn_out, attn_weights = self.attention(lstm_out)
        dense1_out = torch.nn.ReLU()(self.dense1(attn_out))
        dense2_out = self.dense2(dense1_out)
        distribution_outputs = self.custom_layer.TTM_dist_layer_corr2(dense2_out)

        return distribution_outputs, attn_weights, nu_1, nu_2 #is used to check the importance of sequence.


def init_weights(m):
    """Initialise Linear and LSTM weights for a freshly constructed model.

    Applies Xavier-uniform to Linear and input-hidden LSTM weights, orthogonal to the
    hidden-hidden LSTM weights, sets Linear biases and LSTM biases to 0.01, and sets the
    forget-gate slice of each LSTM bias vector to one. Use as ``model.apply(init_weights)``.

    Args:
        m: Module visited by ``nn.Module.apply``.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.01)
                # ensure LSTM bias is initialised properly
                n = param.size(0)
                param[n//4:n//2].data.fill_(1.0)  # forget gate bias
