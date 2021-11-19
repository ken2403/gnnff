import torch
from torch import Tensor
import torch.nn as nn
from gnnff.nn import Dense


__all__ = ["ForcePrediction"]


class ForcePrediction(nn.Module):
    """

    Attributes
    ----------
    n_edge_feature : int
        dimension of the embedded edge features.
    n_layer : int, default=2

    """

    def __init__(self, n_edge_feature: int, n_layer: int = 2) -> None:
        super().__init__()
