from torch import Tensor
from torch._C import _set_mkldnn_enabled
import torch.nn as nn
from torch.nn import Sigmoid, Tanh
from gnnff.nn import Dense


__all__ = ["NodeUpdate", "EdgeUpdate", "MessagePassing"]


class NodeUpdate(nn.Module):
    """
    Updated the node embedding tensor from the previous node embedding and the previous edge embedding.

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    n_edge_feature : int
        dimension of the embedded edge features.
    """

    def __init__(
        self,
        n_node_feature: int,
        n_edge_feature: int,
    ) -> None:
        super().__init__()
        self.gate = Dense(
            n_node_feature + n_edge_feature, n_node_feature, activation=Sigmoid
        )
        self.extract = Dense(
            n_node_feature + n_edge_feature, n_node_feature, activation=Tanh
        )

    def forward(self, node_embedding: Tensor, edge_embeding: Tensor) -> Tensor:
        """
        Calculate the updated node embedding.

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (N_batch x N_atoms x N_node_features) shape.
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (N_batch x N_atoms x N_neighbors x N_edge_feutures) shape.

        Returns
        -------
        updated_node : torch.Tensor
            updated node embedding tensor of (N_batch x N_atoms x N_node_feutures) shape.
        """


class EdgeUpdate(nn.Module):
    """
    Updated the edge embedding tensor from the new node embedding and the previous edge embedding.

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    n_edge_feature : int
        dimension of the embedded edge features.
    """

    def __init__(
        self,
        n_node_feature: int,
        n_edge_feature: int,
    ) -> None:
        super().__init__()
        self.two_body_gate = Dense(n_node_feature, n_edge_feature, activation=Sigmoid)
        self.two_body_extract = Dense(n_node_feature, n_edge_feature, activation=Tanh)
        self.three_body_gate = Dense(
            3 * n_node_feature + 2 * n_edge_feature,
            n_edge_feature,
            activation=Sigmoid,
        )
        self.three_body_extract = Dense(
            3 * n_node_feature + 2 * n_edge_feature,
            n_edge_feature,
            activation=Tanh,
        )

    def forward(
        self, node_embedding: Tensor, edge_embeding: Tensor, neighbors_k: Tensor
    ) -> Tensor:
        """
        Calculate the updated edge embedding.

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (N_batch x N_atoms x N_node_feutures) shape.
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (N_batch x N_atoms x N_neighbors x N_edge_feutures) shape.
        neighbors_k : torch.Tensor


        Returns
        -------
        updated_edge : torch.Tensor
            updated edge embedding tensor of (N_batch x N_atoms x N_neighbors x N_edge_feutures) shape.
        """


class MessagePassing(nn.Module):
    """
    Automated feature extraction layer in GNNFF

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    n_edge_feature : int
        dimension of the embedded edge features.

    Referrences
    -----------
    .. [1] Park, C.W., Kornbluth, M., Vandermause, J. et al.
       Accurate and scalable graph neural network force field
       and molecular dynamics with direct force architecture.
       npj Comput Mater 7, 73 (2021).
    """

    def __init__(
        self,
        n_node_feature: int,
        n_edge_feature: int,
    ) -> None:
        super().__init__()
        self.update_node = NodeUpdate(n_node_feature, n_edge_feature)
        self.update_edge = EdgeUpdate(n_node_feature, n_node_feature)

    def forward(self, node_embedding: Tensor, edge_embeding: Tensor) -> Tensor:
        """
        Calculate the updated node embedding.

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (N_batch x N_atoms x N_node_features) shape.
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (N_batch x N_atoms x N_neighbors x N_edge_feutures) shape.

        Returns
        -------
        updated_node : torch.Tensor
            updated node embedding tensor of (N_batch x N_atoms x N_node_feutures) shape.
        updated_edge : torch.Tensor
            updated edge embedding tensor of (N_batch x N_atoms x N_neighbors x N_edge_feutures) shape.
        """
