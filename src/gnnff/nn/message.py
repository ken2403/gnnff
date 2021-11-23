import torch
from torch import Tensor
import torch.nn as nn
from gnnff.nn.base import Dense
from gnnff.nn.neighbors import GetNodeK, GetEdgeK


__all__ = ["NodeUpdate", "EdgeUpdate", "MessagePassing"]


class NodeUpdate(nn.Module):
    """
    Updated the node embedding tensor from the previous node and edge embedding.

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
        self.fc = Dense(
            n_node_feature + n_edge_feature, 2 * n_node_feature, activation=None
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(2 * n_node_feature)
        self.bn2 = nn.BatchNorm1d(n_node_feature)

    def forward(
        self, node_embedding: Tensor, edge_embeding: Tensor, nbr_mask: Tensor
    ) -> Tensor:
        """
        Calculate the updated node embedding.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (B x At x n_node_feature) shape.
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        nbr_mask : torch.Tensor
            boolean mask for neighbor positions.(B x At x Nbr) of shape.

        Returns
        -------
        updated_node : torch.Tensor
            updated node embedding tensor of (B x At x n_node_feature) shape.

        References
        ----------
        .. [1] https://github.com/ken2403/cgcnn/blob/master/cgcnn/model.py
        """
        B, At, Nbr, _ = edge_embeding.size()
        _, _, n_node_feature = node_embedding.size()

        # make c1-ij tensor
        c1 = torch.cat(
            [
                node_embedding.unsqueeze(2).expand(B, At, Nbr, n_node_feature),
                edge_embeding,
            ],
            dim=3,
        )
        # apply neighbor mask and if there are no neighbor, padding with 0
        c1[nbr_mask == 0] = 0.0
        # fully connected layter
        c1 = self.fc(c1)
        c1 = self.bn1(c1.view(-1, 2 * n_node_feature)).view(
            B, At, Nbr, 2 * n_node_feature
        )
        # calculate the gate and extract features
        nbr_gate, nbr_extract = c1.chunk(2, dim=3)
        nbr_gate = self.sigmoid(nbr_gate)
        nbr_extract = self.tanh(nbr_extract)
        # elemet-wise multiplication with gate
        nbr_sumed = torch.sum(nbr_gate * nbr_extract, dim=2)
        nbr_sumed = self.bn2(nbr_sumed.view(-1, n_node_feature)).view(
            B, At, n_node_feature
        )
        # last activation layer
        updated_node = self.tanh(node_embedding + nbr_sumed)
        return updated_node


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
        self.get_node_k = GetNodeK(n_node_feature)
        self.get_edge_k = GetEdgeK(n_edge_feature)
        self.fc_two_body = Dense(n_node_feature, 2 * n_edge_feature, activation=None)
        self.fc_three_body = Dense(
            3 * n_node_feature + 2 * n_edge_feature,
            2 * n_edge_feature,
            activation=None,
        )
        self.bn_two_body = nn.BatchNorm1d(2 * n_edge_feature)
        self.bn_three_body = nn.BatchNorm1d(2 * n_edge_feature)
        self.bn_sum = nn.BatchNorm1d(n_edge_feature)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh

    def forward(
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        nbr_idx: Tensor,
        nbr_mask: Tensor,
    ) -> Tensor:
        """
        Calculate the updated edge embedding.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (B x At x n_node_feature) shape.
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        nbr_idx : torch.Tensor
            Indices of neighbors of each atom. (B x At x Nbr) of shape.
        nbr_mask : torch.Tensor
            boolean mask for neighbor positions.(B x At x Nbr) of shape.

        Returns
        -------
        updated_edge : torch.Tensor
            updated edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        """
        B, At, Nbr, n_edge_feature = edge_embedding.size()
        _, _, n_node_feature = node_embedding.size()

        # make c2_ij tensor. (B x At x Nbr x n_node_feature) of shape.
        node_i = node_embedding.unsqueeze(2).expand(B, At, Nbr, n_node_feature)
        nbh = nbr_idx.reshape(-1, At * Nbr, 1)
        nbh = nbh.expand(-1, -1, n_node_feature)
        node_j = torch.gather(node_embedding, dim=1, index=nbh)
        node_j = node_j.view(B, At, Nbr, -1)
        c2 = node_i * node_j
        # apply neighbor mask and if there are no neighbor, padding with 0
        c2[nbr_mask == 0] = 0.0
        # fully connected layter with c2
        c2 = self.fc_two_body(c2)
        c2 = self.bn_two_body(c2.view(-1, 2 * n_edge_feature)).view(
            B, At, Nbr, 2 * n_edge_feature
        )
        # calculate the gate and extract features with two-body interaction
        two_body_gate, two_body_extract = c2.chunk(2, dim=3)
        two_body_gate = self.sigmoid(two_body_gate)
        two_body_extract = self.tanh(two_body_extract)
        # elemet-wise multiplication with gate on two-body interaction
        two_body_embedding = two_body_gate * two_body_extract

        # make c3_ijk tensor. (B x At x Nbr x Nbr_k x 3*n_node_feature + 2*n_edge_feature) of shape.
        Nbr_k = Nbr - 1
        node_i = node_i.unsqueeze(3).expand(B, At, Nbr, Nbr_k, n_node_feature)
        node_j = node_j.unsqueeze(3).expand(B, At, Nbr, Nbr_k, n_node_feature)
        node_k = self.get_node_k(node_embedding, nbr_idx)
        edge_ij = edge_embedding.unsqueeze(3).expand(B, At, Nbr, Nbr_k, n_edge_feature)
        edge_kj = self.get_edge_k(edge_embedding, nbr_idx)
        c3 = torch.cat([node_i, node_j, node_k, edge_ij, edge_kj], dim=4)
        # fully connected layter with c3
        c3 = self.fc_three_body(c3)
        c3 = self.bn_three_body(c3.view(-1, 2 * n_edge_feature)).view(
            B, At, Nbr, Nbr_k, 2 * n_edge_feature
        )
        # calculate the gate and extract features with three-body interaction
        three_body_gate, three_body_extract = c3.chunk(2, dim=4)
        three_body_gate = self.sigmoid(three_body_gate)
        three_body_extract = self.tanh(three_body_extract)
        # elemet-wise multiplication with gate on three-body interaction
        three_body_embedding = torch.sum(three_body_gate * three_body_extract, dim=3)
        three_body_embedding = self.bn_sum(
            three_body_embedding.view(-1, n_edge_feature)
        ).view(B, At, Nbr, n_edge_feature)

        # last activation layer
        updated_edge = self.tanh(
            edge_embedding + two_body_embedding + three_body_embedding
        )
        return updated_edge


class MessagePassing(nn.Module):
    """
    Automated feature extraction layer in GNNFF.

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
        self.update_node = NodeUpdate(n_node_feature, n_edge_feature)
        self.update_edge = EdgeUpdate(n_node_feature, n_node_feature)

    def forward(
        self,
        node_embedding: Tensor,
        edge_embeding: Tensor,
        nbr_idx: Tensor,
        nbr_mask: Tensor,
    ) -> Tensor:
        """
        Calculate the updated node and edge embedding by message passing layer.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        node_embedding : torch.Tensor
            batch of node embedding tensor of (B x At x n_node_feature) shape.
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        nbr_idx : torch.Tensor
            Indices of neighbors of each atom. (B x At x Nbr) of shape.
        nbr_mask : torch.Tensor

        Returns
        -------
        updated_node : torch.Tensor
            updated node embedding tensor of (B x At x n_node_feature) shape.
        updated_edge : torch.Tensor
            updated edge embedding tensor of (B x At x Nbr x n_edge_feuture) shape.
        """
        updated_node = self.update_node(node_embedding, edge_embeding, nbr_mask)
        updated_edge = self.update_edge(updated_node, edge_embeding, nbr_idx, nbr_mask)
        return updated_node, updated_edge
