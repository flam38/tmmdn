"""TMMDN: the ALSTM-TTM network architecture and its distribution layer.

Attention-LSTM encoder with a two-component multivariate Student-t mixture output
layer, from Lam, Chan and Choy (2026), Intelligent Systems with Applications 31,
200700, https://doi.org/10.1016/j.iswa.2026.200700.
"""

from ._version import __version__
from .distributions import log_t_dist
from .layers import Attention, CustomLayer_TTM_Corr, chunks_for, output_extraction_TTM
from .model import ALSTM_TTM_Model, init_weights

__all__ = [
    "__version__",
    "ALSTM_TTM_Model",
    "Attention",
    "CustomLayer_TTM_Corr",
    "chunks_for",
    "init_weights",
    "log_t_dist",
    "output_extraction_TTM",
]
