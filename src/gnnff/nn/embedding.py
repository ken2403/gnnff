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

        B   :  Batch size
        At  :  Total number of atoms in the batch

        Parameters
        ----------
        inputs : torch.Tensor
            batch of input values. (B x At) of shape.

        Returns
        -------
        y : torch.Tensor
            layer output. (B x At x embeddig_dim) of shape.
        """
        y = super().forward(inputs)
        return y


def gaussian_filter(distances, offsets, widths, centered=True):
    """
    Filtered interatomic distance values using Gaussian functions.

    B   :  Batch size
    At  :  Total number of atoms in the batch
    Nbr :  Total number of neighbors of each atom
    G   :  Filtered features

    Parameters
    ----------
    distances : torch.Tensor
        interatomic distances of (B x At x Nbr) shape.
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
        filtered distances of (B x At x Nbr x G) shape.

    References
    ----------
    .. [1] https://github.com/atomistic-machine-learning/schnetpack/blob/67226795af55719a7e4565ed773881841a94d130/src/schnetpack/nn/acsf.py
    """
    if centered:
        # if Gaussian functions are centered, use offsets to compute widths
        eta = 0.5 / torch.pow(offsets, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, :, None]

    else:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        eta = 0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offsets[None, None, None, :]

    # compute smear distance values
    filtered_distances = torch.exp(-eta * torch.pow(diff, 2))
    return filtered_distances


class EdgeEmbedding(nn.Module):
    """
    Initial edge embedding layer.
    From inter-atomic distaces, calculates the edge embedding tensor.

    Attributes
    ----------
    start : float, default=0.0
        center of first Gaussian function, :math:`\mu_0`.
    stop : float, default=8.0
        center of last Gaussian function, :math:`\mu_{N_g}`
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
        self.register_buffer("offset", offsets)
        self.register_buffer("width", widths)
        self.centered = centered

    def forward(self, distances: Tensor) -> Tensor:
        """
        Compute filtered distance values with Gaussian filter.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom
        G   :  Filtered features. (n_gaussian)

        Parameters
        ----------
        distances : torch.Tensor
            interatomic distance values of (B x At x Nbr) shape.

        Returns
        -------
        filtered_distances : torch.Tensor
            filtered distances of (B x At x Nbr x G) shape.
        """
        return gaussian_filter(
            distances, offsets=self.offset, widths=self.width, centered=self.centered
        )
