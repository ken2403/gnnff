from torch import Tensor
import torch.nn as nn

from gnnff.data.keys import Keys
from gnnff.nn.embedding import NodeEmbedding, EdgeEmbedding
from gnnff.nn.message import MessagePassing


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
    gaussian_filter_end : float, default=6.0
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
        gaussian_filter_end: float = 6.0,
        share_weights: bool = False,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        # layer for initial embedding of node and edge.
        self.initial_node_embedding = NodeEmbedding(n_node_feature)
        self.initial_edge_embedding = EdgeEmbedding(
            start=0.0,
            stop=gaussian_filter_end,
            n_gaussian=n_edge_feature,
            centered=False,
        )
        # layers for computing some message passing layers.
        if share_weights:
            self.message_passings = nn.ModuleList(
                [
                    MessagePassing(
                        n_node_feature=n_node_feature,
                        n_edge_feature=n_edge_feature,
                    )
                ]
                * n_message_passing
            )
        else:
            self.message_passings = nn.ModuleList(
                [
                    MessagePassing(
                        n_node_feature=n_node_feature,
                        n_edge_feature=n_edge_feature,
                    )
                    for _ in range(n_message_passing)
                ]
            )
        # set the attribute
        self.return_intermediate = return_intermediate

    def forward(self, inputs: dict) -> Tensor:
        """
        Compute initial embedding and repeated message passings.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        inputs : dict of torch.Tensor
            dictionary of property tensors in unit cell.

        Returns
        -------
        last_node_embedding : torch.Tensor
            atomic node embedding tensors with (B x At x n_node_feature) shape.
        last_edge_embedding : torch.Tensor
            inter atomic edge embedding tensors with (B x At x Nbr x n_edge_feature) shape.
        2 lists of torch.Tensor
            intermediate node and edge embeddings, if return_intermediate=True was used.
        """
        # get tensors from input dictionary
        atomic_numbers = inputs[Keys.Z]
        nbr_idx = inputs[Keys.neighbors]
        nbr_mask = inputs[Keys.neighbor_mask]
        # atom_mask = inputs[Keys.atom_mask]

        # get cell_offsets and inter atomic distances.
        r_ij = inputs[Keys.distances]
        cell_offset = inputs[Keys.cell_offset]

        # get initial embedding
        node_embedding = self.initial_node_embedding(atomic_numbers)
        edge_embedding = self.initial_edge_embedding(r_ij)
        # # apply neighbor mask, if there are no neighbor, padding with 0
        # edge_embedding[nbr_mask == 0] = 0.0

        # store inter mediate values
        if self.return_intermediate:
            node_list = [node_embedding]
            edge_list = [edge_embedding]

        # message passing
        for message_passing in self.message_passings:
            node_embedding, edge_embedding = message_passing(
                node_embedding, edge_embedding, nbr_idx, nbr_mask, cell_offset
            )
            if self.return_intermediate:
                node_list.append(node_embedding.detach().cpu().numpy())
                edge_list.append(edge_embedding.detach().cpu().numpy())

        if self.return_intermediate:
            return node_embedding, edge_embedding, node_list, edge_list
        return node_embedding, edge_embedding
