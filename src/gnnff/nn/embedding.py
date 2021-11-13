import torch
from torch import Tensor
import torch.nn as nn


__all__ = ["NodeEmbedding", "EdgeEmbedding"]


class NodeEmbedding(nn.Embedding):
    """
    Initial node embedding layer.
    From atomic-numbers, calculates the node embedding tensor.

    Attributes
    ----------
    embedding_dim : int
        the size of each embedding vector.
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__(num_embeddings=100, embedding_dim=embedding_dim, padding_idx=0)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Compute layer output.

        Parameters
        ----------
        inputs : torch.Tensor
            batch of input values.

        Returns
        -------
        y : torch.Tensor
            layer output.
        """
        y = super().forward(inputs)
        return y


def gaussian_filter(distances, offsets, widths, centered=True):
    """
    Filtered interatomic distance values using Gaussian functions.

    Parameters
    ----------
    distances : torch.Tensor
        interatomic distances of (N_batch x N_atoms x N_neighbors) shape.
    offsets : torch.Tensor
        offsets values of Gaussian functions.
    widths : torch.Tensor
        width values of Gaussian functions.
    centered : bool, default=True
        If True, Gaussians are centered at the origin and the offsets are used
        to as their widths (used e.g. for angular functions).

    Returns
    -------
    filtered_distances : torch.Tensor
        filtered distances of (N_batch x N_atoms x N_neighbor x N_gaussian) shape.
    """
    if centered:
        # if Gaussian functions are centered, use offsets to compute widths
        eta = 0.5 / torch.pow(offsets, 2)
        # if Gaussian functions are centered, no offset is subtracted
        myu = distances[:, :, :, None]

    else:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        eta = 0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        myu = distances[:, :, :, None] - offsets[None, None, None, :]

    # compute smear distance values
    filtered_distances = torch.exp(-eta * torch.pow(myu, 2))
    return filtered_distances


class EdgeEmbedding(nn.Module):
    """
    Initial edge embedding layer.
    From inter-atomic distaces, calculates the edge embedding tensor.

    Attributes
    ----------
    start : float, default=0.0
        width of first Gaussian function, :math:`\mu_0`.
    stop : float, default=8.0
        width of last Gaussian function, :math:`\mu_{N_g}`
    n_gaussians : int, default=100
        total number of Gaussian functions, :math:`N_g`.
    centered : bool, default=True
        If False, Gaussian's centered values are varied at the offset values and the width value is constant.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 8.0,
        n_gaussian: int = 100,
        centered: bool = True,
    ) -> None:
        super().__init__()
        offsets = torch.linspace(start=start, end=stop, steps=n_gaussian)
        widths = torch.FloatTensor((offsets[1] - offsets[0]) * torch.ones_like(offsets))
        self.register_buffer = ("offsets", offsets)
        self.register_buffer = ("widths", widths)
        self.centered = centered

    def forward(self, distances: Tensor) -> Tensor:
        """
        Compute filtered distance values with Gaussian filter.

        Parameters
        ----------
        distances : torch.Tensor
            interatomic distance values of (N_batch x N_atoms x N_neighbors) shape.

        Returns
        -------
        filtered_distances : torch.Tensor
            filtered distances of (N_batch x N_atoms x N_neighbor x N_gaussian) shape.
        """
        return gaussian_filter(
            distances, self.offsets, self.widths, centered=self.centered
        )
