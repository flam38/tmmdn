"""Attention layer and the distribution (output) layer of the ALSTM-TTM model.

``CustomLayer_TTM_Corr`` turns the dense output of the network into the parameters
of a two-component multivariate Student-t mixture and supplies the loss used to fit
it. The computations are those used for the paper. The only changes are that
quantities the original code read from notebook-level variables (the number of
assets ``p``, the prior mean ``mu1_prior`` and the device) are now explicit arguments
or are taken from the input tensor, and that two unused diagnostic helpers and an
unused local variable were left out.

Layout of the distribution parameters (width ``chunks_for(p)``), in this order:
    component 1: ``p`` means, then ``p`` diagonal entries and ``p * (p - 1) / 2``
    lower-triangular entries of a factor ``L`` of the precision matrix ``L L'``;
    component 2: the same block;
    one mixture-weight entry.
"""

import numpy as np
import torch
import torch.nn as nn

from .distributions import log_t_dist

__all__ = ["Attention", "CustomLayer_TTM_Corr", "chunks_for", "output_extraction_TTM"]


def chunks_for(p):
    """Width of the raw network output for ``p`` assets.

    Args:
        p: Number of assets (dimension of the return vector).

    Returns:
        ``2 * (p + p + p * (p - 1) / 2) + 1``, for example 755 for ``p = 26`` and
        109 for ``p = 9``.
    """
    diag = p
    low_triang = p * (p - 1) // 2  # Integer division is used here
    return int(2 * (p + diag + low_triang) + 1)


class Attention(nn.Module):
    """Attention pooling over an LSTM output sequence.

    Each hidden state is scored against one learned vector, the scores are passed through
    a softmax over the sequence, and the hidden states are averaged with those weights.

    Args:
        hidden_size: Size of the hidden states the attention scores.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        # Initialize the attention weights
        nn.init.xavier_uniform_(self.att_weights)

    def forward(self, outputs):
        # outputs shape: (batch_size, sequence_length, hidden_size)
        scores = torch.bmm(outputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(outputs.size(0), 1, 1))
        # scores shape: (batch_size, sequence_length, 1)
        att_weights = torch.softmax(scores.squeeze(2), dim=1) #to transform the scores into a set of probabilities (attention weights) across the sequence length. This step ensures that higher scores have higher weights, and it sums to 1 across the sequence for each sample in the batch.
        # New outputs shape: (batch_size, hidden_size)
        weighted = torch.mul(outputs, att_weights.unsqueeze(-1).expand_as(outputs))
        representations = weighted.sum(1)
        return representations, att_weights


class CustomLayer_TTM_Corr(nn.Module):
    """Output layer that maps a network head to a two-component Student-t mixture.

    Converts the dense output into the means, precision-matrix factors and mixture
    weight of a two-component Student-t mixture, and supplies the loss used for
    training. The degrees of freedom are not set here; they are learned scalars in
    ``ALSTM_TTM_Model``. The layer has no trainable parameters.

    Args:
        p: Number of assets, the dimension of the return vector being modelled.
        mu1_prior: Anchor for the mean of the first mixture component, a single
            number or one number per asset, in the same units as the targets. The
            component mean is ``mu1_prior * (1 + 0.5 * tanh(raw))``, so it always lies
            between 0.5 and 1.5 times ``mu1_prior``. It is computed from the training
            period only. In the paper's notebook it is the mean of the first target
            asset's training returns after the most extreme 5% of observations
            (outside the 2.5th and 97.5th percentiles) are set to zero, with the
            targets being 21-day returns multiplied by 1000.
    """
    def __init__(self, p, mu1_prior):
        super(CustomLayer_TTM_Corr, self).__init__()
        ############################# Multivariate Dimension #############################
        self.p = int(p)
        self.diag = self.p
        self.low_triang = self.p * (self.p - 1) // 2
        self.chunks = chunks_for(self.p)
        # A plain attribute, not a buffer: keeps the state_dict identical to the
        # trained checkpoints, which store no prior.
        self.mu1_prior = np.asarray(mu1_prior, dtype=np.float64)

    def TTM_dist_layer_corr2(self, x):
        """Build the t-mixture distribution parameters from the network output.

        Splits ``x`` into the per-component means, the diagonal and lower-triangular
        entries of the precision-matrix factors, and the mixture weight. The first
        component mean is anchored on the training-period prior mean.

        Args:
            x: Dense output of the network, shape ``(batch, chunks_for(p))``.

        Returns:
            The assembled distribution parameters for the mixture components, shape
            ``(batch, chunks_for(p))``, in the layout described in the module docstring.
        """
        p = self.p
        low_triang = self.low_triang
        chunks = self.chunks
        device = x.device

        num_dims = len(x.size())
        output = torch.chunk(x, chunks=chunks, dim=-1)

        ############################## 1st Component ##############################
        mu1 = torch.cat(output[0:p], dim = num_dims - 1).to(device) # All the ""p"" mean.`
        mu1 = torch.from_numpy(np.array(self.mu1_prior)).to(torch.float64).to(device) + 0.5 * torch.from_numpy(np.array(self.mu1_prior)).to(torch.float64).to(device) * torch.nn.Tanh()(mu1)
        L100 = torch.cat(output[p:p+1], dim = num_dims -1).to(device) # This is the first element of the diagonal variance matrix.
        L1_diag = torch.cat(output[p+1:2*p], dim = num_dims -1).to(device) # This is the diagonal of the Cholesky matrix.
        LT1 = torch.cat(output[2*p: int(2*p+low_triang)], dim = num_dims-1).to(device) # The lower triangle of the Cholesky decomposition of the correlation Matrix.
        #=========================== constraints of 1st component ===========================#
        L100 = torch.nn.Softplus()(L100).to(device)
        L1_diag = torch.nn.Softplus()(L1_diag)
        #=========================== constraints of 1st component ===========================#

        ############################## 2nd Component ##############################
        mu2 = torch.cat(output[int(2*p+low_triang):int(3*p+low_triang)], dim = num_dims - 1).to(device)
        L200 = torch.cat(output[int(3*p+low_triang):int(3*p+low_triang+1)], dim = num_dims -1).to(device) # This is the first element of the diagonal variance matrix.
        L2_diag = torch.cat(output[int(3*p+low_triang+1):int(3*p+low_triang+p)], dim = num_dims -1).to(device) # This is the diagonal of the Cholesky matrix.
        LT22 = torch.cat(output[int(4*p+low_triang):int(4*p+2*low_triang)], dim = num_dims-1).to(device)
        #=========================== constraints of 2nd component ===========================#
        L200 = torch.nn.Softplus()(L200).to(device)
        L2_diag = torch.nn.Softplus()(L2_diag)
        #=========================== constraints of 2nd component ===========================#

        ############################## Weight Component ##############################
        w1 = torch.cat(output[int(4*p+2*low_triang):int(4*p+2*low_triang+1)], dim = num_dims-1).to(device)
        #=========================== constraints of Weight component ===========================#
        w1 = torch.nn.Sigmoid()(w1).to(device)
        #=========================== constraints of Weight component ===========================#

        out_tensor = torch.cat([mu1, L100, L1_diag, LT1, mu2, L200, L2_diag, LT22, w1], dim = num_dims - 1)
        return out_tensor

    @staticmethod
    def mu_scale(output, p):
        """Convert the distribution parameters into means, precision matrices and weight.

        Args:
            output: Distribution parameters produced by ``TTM_dist_layer_corr2``,
                shape ``(batch, chunks_for(p))``.
            p: Number of assets.

        Returns:
            ``(mu1, inv_scale1, mu2, inv_scale2, w1)``: the two component means
            ``(batch, p)``, the two precision (inverse scale) matrices
            ``(batch, p, p)``, each built as ``L @ L.T`` from a lower-triangular
            factor, and the weight of the first component ``(batch, 1)``.
        """
        device = output.device
        low_triang = p * (p - 1) // 2
        chunks = chunks_for(p)
        output = torch.chunk(output, chunks=chunks, dim=-1)
        ######################################### First component ############################################
        mu1 = torch.cat(output[0:p], dim = - 1).to(device)
        L100 = torch.cat(output[p:p+1], dim = - 1).to(device)
        D11 = torch.cat(output[p+1:int(2*p)], dim = -1).to(device)
        LT11 = torch.cat(output[int(2*p):int(2*p+low_triang)], dim = -1).to(device)

        # Building the Cholesky Decomposition of the Variance Covariance Matrix
        # Assuming the first tensor in the tuple has the batch dimension
        batch_size = mu1.shape[0]
        zeros_matrix1 = torch.zeros(batch_size, p, p, dtype=torch.float64).to(device)
        # Create a lower triangular matrix with T21 as values
        lower_triangle1 = zeros_matrix1.clone()
        lower_triangle1[:, (torch.tril(torch.ones(p, p)) == 1).T==0] = LT11.to(torch.float64)

        # Compute the diagonal elements
        diag_elements = torch.cat([
            L100,
            D11[:, 0:p-1],
            ], dim=1)

        # Create a batch of diagonal matrices
        batch_of_matrices = torch.diag_embed(diag_elements)
        lower_triangle1 += batch_of_matrices
        matrices_tensor1 = torch.bmm(lower_triangle1, torch.transpose(lower_triangle1,1,2))
        inv_scale1 = matrices_tensor1

        ######################################### Second component ############################################
        mu2 = torch.cat(output[int(2*p+low_triang):int(3*p+low_triang)], dim =  - 1)
        L200 = torch.cat(output[int(3*p+low_triang):int(3*p+low_triang)+1], dim =  - 1)
        D22 = torch.cat(output[int(3*p+low_triang+1):int(4*p+low_triang)], dim =  -1)
        LT22 = torch.cat(output[int(4*p+low_triang):int(4*p+2*low_triang)], dim = -1)
        w1 = torch.cat(output[int(4*p+2*low_triang):int(4*p+2*low_triang)+1], dim = -1)
        # Assuming the first tensor in the tuple has the batch dimension
        zeros_matrix2 = torch.zeros(batch_size, p, p, dtype=torch.float64).to(device)
        # Create a lower triangular matrix with T22 as values
        lower_triangle2 = zeros_matrix2.clone()
        lower_triangle2[:, (torch.tril(torch.ones(p, p)) == 1).T==0] = LT22.to(torch.float64)

        # Compute the diagonal elements
        diag_elements = torch.cat([
            L200,
            D22[:, 0:p-1],
            ], dim=1)
        # Create a batch of diagonal matrices
        batch_of_matrices2 = torch.diag_embed(diag_elements)
        lower_triangle2 += batch_of_matrices2
        matrices_tensor2 = torch.bmm(lower_triangle2, torch.transpose(lower_triangle2,1,2))
        inv_scale2 = matrices_tensor2

        return mu1, inv_scale1, mu2, inv_scale2, w1

    @staticmethod
    def negative_log_likelihood_loss_tt(y_pred, y_true, nu_1, nu_2):
        """Mean negative log-likelihood of the realized returns under the t-mixture.

        Averages the per-sample negative log mixture density over the batch. It differs
        from the loss used to train the models in the paper: the density helper used there
        returned a ``(batch, batch)`` array that the loss averaged in full (equal to this
        quantity only for a batch of one) and left one term out of the gradient with respect
        to the degrees of freedom (see the README).

        Args:
            y_pred: Distribution parameters produced by ``TTM_dist_layer_corr2``.
            y_true: Realized asset returns, shape ``(batch, p)``; ``p`` is read from here.
            nu_1: Degrees of freedom of the first mixture component, shape ``(1,)``.
            nu_2: Degrees of freedom of the second mixture component, shape ``(1,)``.

        Returns:
            The mean negative log-likelihood over the batch.
        """
        p = y_true.shape[1]
        mu1, inv_scale1, mu2, inv_scale2, w1 = CustomLayer_TTM_Corr.mu_scale(y_pred, p)
        log_t_1 = log_t_dist(y_true , mu1 , inv_scale1, nu_1)
        log_t_2 = log_t_dist(y_true , mu2 , inv_scale2, nu_2)
        p1 = torch.exp(log_t_1)
        p2 = torch.exp(log_t_2)
        mixture_density = w1 * p1 + (1 - w1) * p2
        log_likelihood = torch.log(mixture_density)
        neg_log_likelihood = -torch.mean(log_likelihood)
        return neg_log_likelihood


def output_extraction_TTM(output, p):
    """Extract the component means and scale matrices from a t-mixture output.

    Args:
        output: Distribution parameters produced by ``TTM_dist_layer_corr2``,
            shape ``(batch, chunks_for(p))``.
        p: Number of assets.

    Returns:
        ``(mu1, scale1, mu2, scale2, w1)``: the component mean vectors ``(batch, p)``,
        the component scale matrices ``(batch, p, p)`` and the weight of the first
        component ``(batch, 1)``. The scale matrices are the Student-t scale (not the
        covariance); for ``nu > 2`` a component's covariance is ``nu / (nu - 2)``
        times its scale matrix.
    """
    # mu_scale returns the inverse (precision) matrices; invert to recover scale matrices.
    mu1, inv_scale1, mu2, inv_scale2, w1 = CustomLayer_TTM_Corr.mu_scale(output, p)
    scale1 = torch.inverse(inv_scale1)
    scale2 = torch.inverse(inv_scale2)
    return mu1, scale1, mu2, scale2, w1
