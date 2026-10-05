# TMMDN: Temporal Multivariate Mixture Density Network (ALSTM-TTM)

Network architecture of the **ALSTM-TTM** model, an attention-LSTM encoder with a
two-component multivariate Student-*t* mixture output layer, from

> Lam, F., Chan, J. S. K., and Choy, S. T. B. (2026).
> *Temporal Multivariate Mixture Density Networks for Enhanced Portfolio Optimization.*
> Intelligent Systems with Applications, 31, 200700.
> https://doi.org/10.1016/j.iswa.2026.200700

This repository contains the **network architecture and its distribution layer**, the
loss used to fit it, and a forward-pass example on synthetic data. It is published so
that the architecture can be read, cited and re-implemented. It does not contain market
data, trained models, data preprocessing, the training loop, the portfolio strategies or
the benchmark models of the paper. The paper's Code and Data Availability statement says
that the deep-learning forecasting pipeline (data preprocessing, network construction and
model training) will be made available on GitHub upon request. The data preprocessing and
training code are not part of this repository and can be requested from the corresponding
author listed in the paper.

## The model

For each look-back window of predictors (shape `batch x days x features`) the network
outputs the parameters of a two-component Student-*t* mixture over the joint vector of
`p` asset returns:

1. a two-layer LSTM (`hidden_size` units), then a second LSTM (24 units);
2. attention pooling over the look-back days: each of the 24-unit outputs is scored
   against one learned vector, the scores go through a softmax over the days, and the
   outputs are averaged with those weights;
3. a dense layer 24 to 48 with ReLU, then a dense layer 48 to `chunks_for(p)`;
4. the **distribution layer** (`CustomLayer_TTM_Corr`), which splits that output into
   the two components. Each component has a mean vector and a precision matrix (the
   inverse of the Student-*t* scale matrix) built as `L L'` from a lower-triangular
   factor `L` with a softplus diagonal. A sigmoid gives the weight of the first
   component. The two degrees of freedom are learned scalars `nu = 4 + softplus(u)`,
   where `u` is an unconstrained trainable value, so `nu > 4` always.

The predictive density is `w1 * t(mu1, scale1, nu_1) + (1 - w1) * t(mu2, scale2, nu_2)`.
How the paper turns the mixture into a mean vector and covariance matrix for portfolio
construction is described in the paper and is not included here.

`CustomLayer_TTM_Corr.negative_log_likelihood_loss_tt` is the mean per-sample negative
log-likelihood of realised returns under this mixture. The code used to train the models
in the paper evaluated an earlier version of the density helper. That version returned a
batch-by-batch array and left one term out of the gradient with respect to the degrees of
freedom, so loss values and degrees-of-freedom updates computed with it differ from this
one.

## Parameters

**Arguments of `ALSTM_TTM_Model(input_size, hidden_size, num_layers, p, mu1_prior, nu_1_initial, nu_2_initial)`**

| Argument | Meaning | Value used in the paper (U.S. data) |
|---|---|---|
| `input_size` | Number of predictors per time step, set by your data. | 153 |
| `hidden_size` | Units of the first (two-layer) LSTM. The second LSTM (24 units), the attention and the dense layers have fixed sizes. | 48 |
| `num_layers` | Layers of the first LSTM. It is built with two layers, so use 2. | 2 |
| `p` | Number of assets, the dimension of the return vector being modelled. The network emits `chunks_for(p) = 2 * (2p + p(p-1)/2) + 1` numbers per sample: 755 for `p = 26`, 109 for `p = 9`. | 26 (U.S.), 9 (Japan) |
| `mu1_prior` | Anchor for the mean of the first mixture component; see below. | one data-derived number |
| `nu_1_initial`, `nu_2_initial` | Optional, default 5. Starting values of the unconstrained value `u` behind each degrees-of-freedom parameter, `nu = 4 + softplus(u)` with `softplus(z) = ln(1 + e^z)`. They are not degrees of freedom themselves: 5 gives `nu = 9.007` and 10 gives `nu = 14.00`. | drawn from uniform(3, 20), i.e. `nu` between 7.05 and 24.0 |

**`mu1_prior`.** The mean of the first component is `mu1_prior * (1 + 0.5 * tanh(a))`,
where `a` is the network's output for that asset, so it always lies between 0.5 and 1.5
times `mu1_prior` (the bounds swap order for a negative prior, and a prior of 0 pins the
mean at 0). One number, or one number per asset, anchors all `p` means. It must come from
the training period only. In the paper it is the mean of the first target asset's training
returns, taken over all training observations after setting the most extreme 5%
(outside the 2.5th and 97.5th percentiles) to zero. It is computed from licensed data, so
it is not supplied here: compute it from your own training returns, in the units of your
targets.

**Units.** In the paper's pipeline the targets are 21-day forward simple returns
multiplied by 1000 (a 1% return is 10 units). Means and `mu1_prior` are in those units
and scale matrices are in squared units. The look-back window (100 days in the paper) is a
property of your input data, not of the model.

**Terms.** *Predictors* are the input features at each day. *Degrees of freedom* control
tail heaviness (smaller values give heavier tails). The *scale matrix* of a Student-*t*
plays the role of the covariance matrix; its inverse is the *precision matrix*.

**What the model returns**

`model(x)`, with `x` of shape `(batch, days, input_size)` in float64 (call `model.double()`),
returns `(distribution_outputs, attn_weights, nu_1, nu_2)`:

* `distribution_outputs`, shape `(batch, chunks_for(p))`: the distribution parameters.
  Pass them to `output_extraction_TTM(distribution_outputs, p)` or to the loss.
* `attn_weights`, shape `(batch, days)`: the attention weights over the look-back days.
* `nu_1`, `nu_2`, each of shape `(1,)`: the degrees of freedom, shared by every sample.
  Use these returned values. The attributes `model.nu_1` and `model.nu_2` are not used by
  the forward pass; they are kept only so that trained checkpoints load without errors.

`output_extraction_TTM(distribution_outputs, p)` returns `(mu1, scale1, mu2, scale2, w1)`:

* `mu1`, `mu2`, shape `(batch, p)`: the component mean vectors;
* `scale1`, `scale2`, shape `(batch, p, p)`: the component scale matrices. These are the
  Student-*t* scale, not the covariance; for `nu > 2` a component's covariance is
  `nu / (nu - 2)` times its scale matrix;
* `w1`, shape `(batch, 1)`: the weight of the first component, between 0 and 1.

## Files

| Path | Purpose |
|---|---|
| `tmmdn/model.py` | `ALSTM_TTM_Model` and the weight initialiser `init_weights`. |
| `tmmdn/layers.py` | `Attention`, the distribution layer `CustomLayer_TTM_Corr` with its loss, `output_extraction_TTM` and `chunks_for`. |
| `tmmdn/distributions.py` | The Student-*t* log-density `log_t_dist` used by the loss. |
| `examples/forward_pass.py` | One forward pass on synthetic data with shape and constraint checks. |
| `requirements.txt` | Python dependencies. |
| `CITATION.cff` | Citation metadata. |

## Quick start

```bash
pip install -r requirements.txt
python examples/forward_pass.py
```

The example needs no data. To use the layers in your own code:

```python
from tmmdn import ALSTM_TTM_Model, init_weights, output_extraction_TTM

model = ALSTM_TTM_Model(input_size=153, hidden_size=48, num_layers=2,
                        p=26, mu1_prior=0.5).double()   # 0.5 is only an illustration
model.apply(init_weights)
distribution_outputs, attn_weights, nu_1, nu_2 = model(x)   # x: (batch, days, 153), float64
mu1, scale1, mu2, scale2, w1 = output_extraction_TTM(distribution_outputs, 26)
```

## Data

No data are included. The data used in the paper were retrieved from a Bloomberg
Terminal under a university licence and cannot be redistributed. The architecture
itself needs no data to run.

## Citation

If you use this code, please cite the paper above. Citation metadata is in
`CITATION.cff`.

## Licence

See `LICENSE`.
