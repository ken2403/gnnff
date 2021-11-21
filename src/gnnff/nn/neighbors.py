import torch
from torch import Tensor
import torch.nn as nn
import numpy as np


__all__ = ["GetNodeK", "GetEdgeK", "AtomicDistances"]


class GetNodeK(nn.Module):
    """
    Extract the node embedding of the third atom of triples.

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    """

    def __init__(self, n_node_feature: int) -> None:
        super().__init__()
        self.n_node_feature = n_node_feature

    def forward(self, node_embedding: Tensor, nbr_idx: Tensor) -> Tensor:
        """
        Get the node embedding of the third atom of triples of atom(i, j, k).
        The centered atom corresponds to each index of At.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (B x At x n_node_feature) shape.
        nbr_idx : torch.Tensor
            Indices of neighbors of each atom. (B x At x Nbr) of shape.

        Returns
        -------
        node_k : torch.Tensor
            node embedding of third atom. (B x At x Nbr x Nbr-1 x n_node_feature)
        """
        B, At, Nbr = nbr_idx.size()
        # make index list of atom k
        k_idx_list = []
        for i in range(Nbr):
            list_ = np.arange(Nbr)
            list_ = np.delete(list_, i)
            k_idx_list.append(list_)
        k_idx_list = np.array(k_idx_list)
        k_idx_list = torch.tensor(k_idx_list)
        k_idx_list = k_idx_list.unsqueeze(0).expand(At, Nbr, Nbr - 1)
        k_idx_list = k_idx_list.unsqueeze(0).expand(B, At, Nbr, Nbr - 1)
        nbr_k = nbr_idx.unsqueeze(2).expand(B, At, Nbr, Nbr)
        nbr_k = torch.gather(nbr_k, 3, k_idx_list)
        nbr_k = nbr_k.reshape(B, At * Nbr * (Nbr - 1), 1)
        nbr_k = nbr_k.expand(-1, -1, self.n_node_feature)
        # get atom k's embedding. (B, At, Nbr, Nbr-1, n_node_feature) of shape.
        node_k = torch.gather(node_embedding, 1, nbr_k)
        node_k = node_k.view(B, At, Nbr, Nbr - 1, self.n_node_feature)
        return node_k


class GetEdgeK(nn.Module):
    """
    Extract the edge embedding of the third atom of triples.

    Attributes
    ----------
    n_edge_feature : int
        dimension of the embedded edge features.
    """

    def __init__(self, n_edge_feature: int) -> None:
        super().__init__()
        self.n_edge_feature = n_edge_feature

    def forward(self, edge_embedding: Tensor, nbr_idx: Tensor) -> Tensor:
        """
        Get the edge emmbedding of the third atom of triples of atom(i, j, k).
        The centered atom corresponds to each index of At.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        edge_embedding : torch.Tensor
            batch of node embedding tensor of (B x At x x Nbr x n_edge_feature) shape.
        nbr_idx : torch.Tensor
            Indices of neighbors of each atom. (B x At x Nbr) of shape.

        Returns
        -------
        edge_kj : torch.Tensor
            node embedding of third atom. (B x At x Nbr x Nbr-1 x n_edge_feature)
        """
        B, At, Nbr = nbr_idx.size()

        # expande edge_embdding, (B x At x Nbr x n_edge_featutre)→(B x At x Nbr x Nbr x n_edge_feature)
        nbr_idx_edge = nbr_idx.unsqueeze(3).expand(B, At, Nbr, self.n_edge_feature)
        nbr_idx_edge = nbr_idx_edge.unsqueeze(3).expand(
            B, At, Nbr, Nbr, self.n_edge_feature
        )
        nbr_idx_edeg = nbr_idx_edge.reshape(B, At * Nbr, Nbr, self.n_edge_feature)
        edge_embedding_expand = torch.gather(edge_embedding, 1, nbr_idx_edeg).view(
            B, At, Nbr, Nbr, self.n_edge_feature
        )
        # expande nbr_idx, (B x At x Nbr)→(B x At x Nbr x Nbr)
        nbr_idx_2 = nbr_idx.unsqueeze(3).expand(B, At, Nbr, Nbr)
        nbr_idx_2 = nbr_idx_2.reshape(B, At * Nbr, Nbr)
        nbr_idx_expand = torch.gather(nbr_idx, 1, nbr_idx_2).view(B, At, Nbr, Nbr)
        # make index list of atom k
        k_idx_list = []
        for i in range(Nbr):
            list_ = np.arange(Nbr)
            list_ = np.delete(list_, i)
            k_idx_list.append(list_)
        k_idx_list = np.array(k_idx_list)
        k_idx_list = torch.tensor(k_idx_list)
        k_idx_list = k_idx_list.unsqueeze(0).expand(At, Nbr, Nbr - 1)
        k_idx_list = k_idx_list.unsqueeze(0).expand(B, At, Nbr, Nbr - 1)
        nbr_k = nbr_idx.unsqueeze(2).expand(B, At, Nbr, Nbr)
        nbr_k = torch.gather(nbr_k, 3, k_idx_list)
        nbr_k = nbr_k.reshape(B, At * Nbr * (Nbr - 1), 1)
        nbr_k = nbr_k.expand(-1, -1, self.n_edge_feature)


def atomic_distances(
    positions: Tensor,
    nbr_idx: Tensor,
    cell: Tensor = None,
    cell_offsets: Tensor = None,
    return_vecs: bool = True,
    normalize_vecs: bool = True,
    neighbor_mask: Tensor = None,
):
    """
    Compute distance of every atom to its neighbors.
    This function uses advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors.

    Parameters
    ----------
    positions : torch.Tensor
        atomic Cartesian coordinates with (B x At x 3) shape.
    neighbors (torch.Tensor):
        indices of neighboring atoms to consider with (B x At x Nbr) shape.
    cell : torch.tensor or None, default=None
        periodic cell of (N_b x 3 x 3) shape.
    cell_offsets : torch.Tensor or None, default=None
        offset of atom in cell coordinates with (B x At x Nbr x 3) shape.
    return_vecs : bool, default=True
        if False, not returns direction vectors.
    normalize_vecs : bool, default=True
        if False, direction vectorsnot are not normalized.
    neighbor_mask : torch.Tensor or None, default=None
        boolean mask for neighbor positions.

    Returns
    -------
    distances : torch.Tensor
        distance of every atom to its neighbors with (B x At x Nbr) shape.
    dist_vec : torch.Tensor
        direction cosines of every atom to its neighbors with (B x At x Nbr x 3) shape.
    """

    # Construct auxiliary index vector
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch, device=positions.device, dtype=torch.long)[
        :, None, None
    ]
    # Get atomic positions of all neighboring indices
    pos_xyz = positions[idx_m, nbr_idx[:, :, :], :]

    # Subtract positions of central atoms to get distance vectors
    dist_vec = pos_xyz - positions[:, :, None, :]

    # add cell offset
    if cell is not None:
        B, At, Nbr, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, At * Nbr, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, At, Nbr, D)
        dist_vec += offsets

    # Compute vector lengths
    distances = torch.norm(dist_vec, 2, 3)

    if neighbor_mask is not None:
        # Avoid problems with zero distances in forces (instability of square
        # root derivative at 0) This way is neccessary, as gradients do not
        # work with inplace operations, such as e.g.
        # -> distances[mask==0] = 0.0
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
        distances = tmp_distances

    if return_vecs:
        tmp_distances = torch.ones_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]

        if normalize_vecs:
            dist_vec = dist_vec / tmp_distances[:, :, :, None]
        return distances, dist_vec
    return distances


class AtomicDistances(nn.Module):
    """
    Layer for computing distance of every atom to its neighbors.

    Attributes
    ----------
    return_directions : bool, default=True
        if False, the `forward` method does not return normalized direction vectors.
    """

    def __init__(self, return_directions: bool = True) -> None:
        super().__init__()
        self.return_directions = return_directions

    def forward(
        self,
        positions: Tensor,
        neighbors: Tensor,
        cell=None,
        cell_offsets=None,
        neighbor_mask=None,
    ) -> Tensor:
        """
        Compute distance of every atom to its neighbors.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        positions : torch.Tensor
            atomic Cartesian coordinates with (B x At x 3) shape.
        neighbors (torch.Tensor):
            indices of neighboring atoms to consider with (B x At x Nbr) shape.
        cell : torch.tensor or None, default=None
            periodic cell of (B x 3 x 3) shape.
        cell_offsets : torch.Tensor or None, default=None
            offset of atom in cell coordinates with (B x At x Nbr x 3) shape.
        neighbor_mask : torch.Tensor or None, default=None
            boolean mask for neighbor positions.

        Returns
        -------
        torch.Tensor
            layer output of (B x At x Nbr) shape.
        dist_vec : torch.Tensor
            direction cosines of every atom to its neighbors with (B x At x Nbr x 3) shape.
        """
        return atomic_distances(
            positions,
            neighbors,
            cell,
            cell_offsets,
            return_vecs=self.return_directions,
            normalize_vecs=True,
            neighbor_mask=neighbor_mask,
        )
