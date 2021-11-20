import torch
from torch import Tensor
import torch.nn as nn
from gnnff.nn import AtomicDistances
from gnnff.nn import NodeEmbedding, EdgeEmbedding
from gnnff.nn import MessagePassing


__all__ = ["GraphToFeatures"]


class GraphToFeatures(nn.Module):
    """
    Layer of combining Initial embedding and repeated message passing layers.

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    n_edge_feature : int
        dimension of the embedded edge features.
    n_message_passing : int, default=3
        number of message passing layers.
    gaussian_filter_end : float, default=8.0
        center of last Gaussian function.
    share_weights : bool, default=False
        if True, share the weights across all message passing layers.
    return_intermediate : bool, default=False
        if True, `forward` method also returns intermediate atomic representations
        after each message passing is applied.
    """

    def __init__(
        self,
        n_node_feature: int,
        n_edge_feature: int,
        n_message_passing: int = 3,
        gaussian_filter_end: float = 8.0,
        share_weights: bool = False,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        # layer for computing interatomic distances.
        self.distances = AtomicDistances()
        # layer for initial embedding of node and edge.
        self.initial_node_embedding = NodeEmbedding(n_node_feature)
        self.initial_edge_embedding = EdgeEmbedding(
            start=0.0,
            stop=gaussian_filter_end,
            n_gaussian=n_edge_feature,
            centered=True,
        )
        # layers for computing some message passing layers.
        if share_weights:
            self.message_passings = nn.ModuleList(
                [
                    MessagePassing(
                        n_node_feature=n_node_feature, n_edge_feature=n_edge_feature
                    )
                ]
                * n_message_passing
            )
        else:
            self.message_passings = nn.ModuleList(
                [
                    MessagePassing(
                        n_node_feature=n_node_feature, n_edge_feature=n_edge_feature
                    )
                    for _ in range(n_message_passing)
                ]
            )
        # set the attribute
        self.return_intermediate = return_intermediate

    def forward(self, inputs: dict) -> Tensor:
        """

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        inputs : dict of torch.Tenso
            dictionary of property tensors in unit cell.

        Returns
        -------
        last_edge_embedding : torch.Tensor
            inter atomic edge embedding tensors with (B x At x Nbr x n_edge_feature) shape.
        unit_vecs : torch.Tensor
            direction cosines of every atom to its neighbors with (B x At x Nbr x 3) shape.
        2 lists of torch.Tensor
            intermediate node and edge embeddings, if return_intermediate=True was used.
        """
        # get tensors from input dictionary
        atomic_numbers = inputs["_atomic_numbers"]
        positions = inputs["_positions"]
        cell = inputs["_cell"]
        cell_offset = inputs["_cell_offset"]
        nbr_idx = inputs["_neighbors"]
        neighbor_mask = inputs["_neighbor_mask"]
        atom_mask = inputs["_atom_mask"]

        # get inter atomic distances
        r_ij, unit_vecs = self.distances(
            positions, nbr_idx, cell, cell_offset, neighbor_mask=neighbor_mask
        )

        # get initial embedding
        node = self.initial_node_embedding(atomic_numbers)
        edge = self.initial_edge_embedding(r_ij)

        # store inter mediate values
        if self.return_intermediate:
            node_list = [node]
            edge_list = [edge]

        # message passing
        for message_passing in self.message_passings:
            node, edge = message_passing(node, edge, nbr_idx)
            if self.return_intermediate:
                node_list.append(node)
                edge_list.append(edge)

        if self.return_intermediate:
            return edge, unit_vecs, node_list, edge_list
        return edge, unit_vecs
