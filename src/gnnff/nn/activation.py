import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = ["softplus", "shifted_softplus"]


@torch.jit.script
def softplus(x: Tensor) -> Tensor:
    """
    Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right)

    Parameters
    ----------
    x : torch.Tensor
        input tensor.

    Returns
    -------
    torch.Tensor
        soft-plus of input.
    """
    return F.softplus(x)


@torch.jit.script
def shifted_softplus(x: Tensor) -> Tensor:
    """
    Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Parameters
    ----------
    x : torch.Tensor
        input tensor.

    Returns
    -------
    torch.Tensor
        shifted soft-plus of input.
    """
    return F.softplus(x) - np.log(2.0)
