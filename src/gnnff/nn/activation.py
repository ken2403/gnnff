import torch.nn as nn
from torch import Tensor

import gnnff.nn.functional as F


__all__ = ["ShiftedSoftplus"]


class ShiftedSoftplus(nn.Module):
    """
    Applies the element-wise function:

    .. math::
       y = \ln\left(1 + e^{-x}\right)

    Notes
    -----
    - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - Output: :math:`(*)`, same shape as the input.

    Examples
    --------
    >>> ss = gnnff.nn.ShiftedSoftplus()
    >>> input = torch.randn(2)
    >>> output = ss(input)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.shifted_softplus(input)
