"""Forward pass of the ALSTM-TTM model on synthetic data. No market data is needed.

Builds the model, runs one forward pass, extracts the two Student-t components and
evaluates the loss, checking the output shapes and constraints.

    python examples/forward_pass.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmmdn import (ALSTM_TTM_Model, CustomLayer_TTM_Corr, chunks_for,
                   init_weights, output_extraction_TTM)

torch.manual_seed(0)

BATCH = 8        # samples (look-back windows)
SEQ_LEN = 100    # look-back days per window
N_FEATURES = 12  # predictors per day (input_size)
N_ASSETS = 6     # p: dimension of the return vector
MU1_PRIOR = 0.5  # illustrative anchor for the first component mean, in target units

# Synthetic inputs: predictors and realised returns (arbitrary units).
x = torch.randn(BATCH, SEQ_LEN, N_FEATURES, dtype=torch.float64)
y = 10 * torch.randn(BATCH, N_ASSETS, dtype=torch.float64)

model = ALSTM_TTM_Model(input_size=N_FEATURES, hidden_size=48, num_layers=2,
                        p=N_ASSETS, mu1_prior=MU1_PRIOR,
                        nu_1_initial=10.0, nu_2_initial=10.0).double()  # nu = 4 + softplus(10) = 14
model.apply(init_weights)
model.eval()

with torch.no_grad():
    dist, attn_weights, nu_1, nu_2 = model(x)
    mu1, scale1, mu2, scale2, w1 = output_extraction_TTM(dist, N_ASSETS)
    nll = CustomLayer_TTM_Corr.negative_log_likelihood_loss_tt(dist, y, nu_1, nu_2)

assert dist.shape == (BATCH, chunks_for(N_ASSETS))
assert attn_weights.shape == (BATCH, SEQ_LEN)
assert mu1.shape == mu2.shape == (BATCH, N_ASSETS)
assert scale1.shape == scale2.shape == (BATCH, N_ASSETS, N_ASSETS)
assert w1.shape == (BATCH, 1) and bool(((w1 > 0) & (w1 < 1)).all())
assert bool((nu_1 > 4).all() and (nu_2 > 4).all())
assert bool((torch.linalg.eigvalsh(scale1) > 0).all()), "scale matrices must be positive definite"
assert bool((torch.linalg.eigvalsh(scale2) > 0).all()), "scale matrices must be positive definite"
assert torch.isfinite(nll)

print(f"raw output {tuple(dist.shape)} = chunks_for({N_ASSETS}) = {chunks_for(N_ASSETS)}")
print(f"component means {tuple(mu1.shape)}, scale matrices {tuple(scale1.shape)}, weight {tuple(w1.shape)}")
print(f"degrees of freedom: nu_1 = {nu_1.item():.3f}, nu_2 = {nu_2.item():.3f} (both > 4)")
print(f"negative log-likelihood on the synthetic returns: {nll.item():.4f}")
print("OK")
