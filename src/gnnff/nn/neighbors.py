import torch
from torch import Tensor
import torch.nn as nn


__all__ = ["GetNodeK", "GetEdgeJK", "AtomicDistances"]


def get_node_k(node_embedding: Tensor, nbr_idx: Tensor, nbr_mask: Tensor) -> Tensor:
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
    nbr_mask : torch.Tensor
        boolean mask for neighbor positions.(B x At x Nbr) of shape.

    Returns
    -------
    node_k : torch.Tensor
        node embedding of third atom. (B x At x Nbr x Nbr-1 x n_node_feature) of shape.
    """
    B, At, n_node_feature = node_embedding.size()
    _, _, Nbr = nbr_idx.size()
    device = node_embedding.device

    # make index list of atom k
    k_idx_list = torch.row_stack([torch.arange(Nbr) for _ in range(Nbr)]).to(device)
    k_idx_list = (
        k_idx_list.flatten()[1:].view(Nbr - 1, Nbr + 1)[:, :-1].reshape(Nbr, Nbr - 1)
    )
    k_idx_list = k_idx_list.unsqueeze(0).expand(At, Nbr, Nbr - 1)
    k_idx_list = k_idx_list.unsqueeze(0).expand(B, At, Nbr, Nbr - 1)
    # k_nbr_idx = nbr_idx.unsqueeze(2).expand(B, At, Nbr, Nbr)
    # k_nbr_idx = k_nbr_idx.reshape(B, At * Nbr * Nbr, 1)
    # k_nbr_idx = k_nbr_idx.expand(-1, -1, n_node_feature)

    nbr_k = nbr_idx.unsqueeze(2).expand(B, At, Nbr, Nbr)
    nbr_k = torch.gather(nbr_k, 3, k_idx_list)
    nbr_k = nbr_k.reshape(B, At * Nbr * (Nbr - 1), 1)
    nbr_k = nbr_k.expand(-1, -1, n_node_feature)
    nbr_mask = nbr_mask.unsqueeze(2).expand(B, At, Nbr, Nbr)
    nbr_mask = torch.gather(nbr_mask, 3, k_idx_list)
    nbr_mask = nbr_mask.unsqueeze(-1).expand(B, At, Nbr, Nbr - 1, n_node_feature)

    # get atom k's embedding. (B x At x Nbr x Nbr-1 x n_node_feature) of shape.
    node_k = torch.gather(node_embedding, 1, nbr_k)
    node_k = node_k.view(B, At, Nbr, Nbr - 1, n_node_feature)
    # avoid inplace operations
    tmp_node_k = torch.zeros_like(node_k)
    tmp_node_k[nbr_mask != 0] = node_k[nbr_mask != 0]
    node_k = tmp_node_k

    ## get atom k's embedding. (B x At x Nbr x Nbr x n_node_feature) of shape.
    # node_k = torch.gather(node_embedding, 1, k_nbr_idx)
    # node_k = node_k.view(B, At, Nbr, Nbr, n_node_feature)
    # # padding with 0 for atom j's embedding
    # nbr_bool = torch.ones((Nbr, Nbr), dtype=torch.int16) - torch.eye(
    #     Nbr, dtype=torch.int16
    # )
    # nbr_bool = nbr_bool.unsqueeze(-1).expand(-1, -1, n_node_feature)
    # nbr_bool.to(device)
    # node_k[:, :, nbr_bool == 0] = 0

    return node_k


class GetNodeK(nn.Module):
    """
    Extract the node embedding of the third atom of triples.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, node_embedding: Tensor, nbr_idx: Tensor, nbr_mask: Tensor
    ) -> Tensor:
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
        nbr_mask : torch.Tensor
            boolean mask for neighbor positions.(B x At x Nbr) of shape.

        Returns
        -------
        node_k : torch.Tensor
            node embedding of third atom. (B x At x Nbr x Nbr x n_node_feature) of shape.
        """
        return get_node_k(node_embedding, nbr_idx, nbr_mask)


def get_edge_jk(edge_embedding: Tensor, nbr_idx: Tensor, cell_offset: Tensor) -> Tensor:
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
    cell_offset : torch.Tensor
        offset of atom in cell coordinates with (B x At x Nbr x 3) shape.

    Returns
    -------
    edge_jk : torch.Tensor
        edge embedding from third atom(k) to second atom(j) of each triples.
        (B x At x Nbr x Nbr x n_edge_feature) of shape.
    """
    B, At, Nbr, n_edge_feature = edge_embedding.size()
    device = edge_embedding.device

    # get edge_embdding from atom i to neighbor k. (B x At x Nbr x Nbr x n_edge_feature) of shape.
    k_nbr_edge = edge_embedding.unsqueeze(2).expand(B, At, Nbr, Nbr, n_edge_feature)
    # nbr_idx_edge = nbr_idx.unsqueeze(3).expand(B, At, Nbr, n_edge_feature)
    # nbr_idx_edge = nbr_idx_edge.unsqueeze(3).expand(B, At, Nbr, Nbr, n_edge_feature)
    # nbr_idx_edge = nbr_idx_edge.reshape(B, At * Nbr, Nbr, n_edge_feature)
    # j_nbr_edge = torch.gather(edge_embedding, 1, nbr_idx_edge).view(
    #     B, At, Nbr, Nbr, n_edge_feature
    # )

    # get j's neighbor indices. (B x At x Nbr x Nbr) of shape.
    nbr_idx_expand = nbr_idx.unsqueeze(3).expand(B, At, Nbr, Nbr)
    nbr_idx_expand = nbr_idx_expand.reshape(B, At * Nbr, Nbr)
    j_nbr_idx = torch.gather(nbr_idx, 1, nbr_idx_expand).view(B, At, Nbr, Nbr)

    # expande nbr_idx, (B x At x Nbr) -> (B x At x Nbr x Nbr)
    k_nbr_idx = nbr_idx.unsqueeze(2).expand(B, At, Nbr, Nbr)
    # set the diagonal component to -1, to eliminate indices of same atom j
    eye = torch.eye(Nbr, dtype=torch.int16).to(device)
    k_nbr_idx = k_nbr_idx * (torch.ones_like(k_nbr_idx) - eye) - eye

    idx_bool = torch.zeros_like(k_nbr_idx)
    for i in range(Nbr):
        # compare the each neighbor idx, and extract if idx matches.
        same_nbr = (j_nbr_idx == k_nbr_idx[:, :, :, i : i + 1]).nonzero()
        # check the position of the each idx.
        pos_check = (
            cell_offset[same_nbr[:, 0], same_nbr[:, 1], i]
            == cell_offset[
                same_nbr[:, 0],
                nbr_idx[same_nbr[:, 0], same_nbr[:, 1], same_nbr[:, 2]],
                same_nbr[:, 3],
            ]
        ).all(dim=1)
        # extract only the idx which is the same location.
        same_nbr = same_nbr[pos_check, :]
        idx_bool[same_nbr[:, 0], same_nbr[:, 1], same_nbr[:, 2], i] = 1.0

    # get edge_embedding from j to k. (B x At x Nbr x Nbr) of shape.
    edge_jk = torch.zeros_like(k_nbr_edge)
    edge_jk[idx_bool == 1, :] = k_nbr_edge[idx_bool == 1, :]

    return edge_jk


class GetEdgeJK(nn.Module):
    """
    Extract the edge embedding of the third atom of triples.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, edge_embedding: Tensor, nbr_idx: Tensor, cell_offset: Tensor
    ) -> Tensor:
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
        cell_offset : torch.Tensor
            offset of atom in cell coordinates with (B x At x Nbr x 3) shape.

        Returns
        -------
        edge_kj : torch.Tensor
            edge embedding from third atom(k) to second atom(j) of each triples.
            (B x At x Nbr x Nbr x n_edge_feature) of shape.
        """
        return get_edge_jk(edge_embedding, nbr_idx, cell_offset)


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

    References
    ----------
    .. [1] https://github.com/ken2403/schnetpack/blob/master/src/schnetpack/nn/neighbors.py
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

    References
    ----------
    .. [1] https://github.com/ken2403/schnetpack/blob/master/src/schnetpack/nn/neighbors.py
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
