"""Student-t log-density called by the ALSTM-TTM loss.

Only the density that the loss of the t-mixture calls is included. It returns one log
density per sample and passes the full gradient with respect to the degrees of freedom.
The version used to train the models in the paper returned a ``(batch, batch)`` array and
detached the ``-(d / 2) * log(nu * pi)`` term from the gradient (see the README).
"""

import torch

__all__ = ["log_t_dist"]


def log_t_dist(x, mu, inv_cov, nu):
    """Log multivariate Student-t density using a precision (inverse scale) matrix.

    Args:
        x: Observations, shape ``(batch, p)``, one row per sample.
        mu: Location vectors, shape ``(batch, p)``.
        inv_cov: Precision (inverse scale) matrices, shape ``(batch, p, p)``.
        nu: Degrees of freedom controlling tail heaviness, a tensor of shape ``(1,)``.

    Returns:
        The log density of each observation, shape ``(batch, 1)``. A warning is printed
        if any value is NaN.
    """
    d = x.shape[1]  # Dimension of the data
    x_mu = x - mu
    mahalanobis = torch.bmm(x_mu.unsqueeze(-1).transpose(1, 2), torch.bmm(inv_cov, x_mu.unsqueeze(-1))).squeeze(-1)
    # When using inv_cov, you need the determinant of the original covariance matrix.
    # Assuming inv_cov is correctly given as the inverse, you can compute the determinant of the covariance matrix:
    det_cov = 1 / torch.det(inv_cov)
    # Logarithm of the determinant of the covariance matrix
    log_det_cov = torch.log(det_cov).unsqueeze(-1)  # column, so it pairs with each sample's own quadratic term
    # Calculating the log-gamma terms
    log_gamma_part = torch.lgamma((nu + d) / 2) - torch.lgamma(nu / 2)
    # Final log probability density calculation
    log_prob_density = (log_gamma_part
                        - 0.5 * log_det_cov
                        - (d / 2) * torch.log(nu * torch.pi)
                        - ((nu + d) / 2) * torch.log(1 + (1 / nu) * mahalanobis))
    nan_mask = torch.isnan(log_prob_density)

    if torch.any(nan_mask):
        print('NaN detected!')
    return log_prob_density
