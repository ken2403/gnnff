import torch
from torch import Tensor
import torch.nn as nn
from gnnff.nn import output
from gnnff.nn.gnn import GraphToFeatures
from gnnff.nn.output import ForceMagnitudeMapping


class GNNFF(nn.Module):
    """
    GNNFF architecture for learning inter atomic interactions of atomistic systems and predict the inter atomic forces.

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    n_edge_feature : int
        dimension of the embedded edge features.
    n_message_passing : int
        number of message passing layers.
    gaussian_filter_end : float
        center of last Gaussian function.
    share_weights : bool
        if True, share the weights across all message passing layers.
    return_intermediate : bool
        if True, `forward` method also returns intermediate atomic representations
        after each message passing is applied.
    n_output_layers : int
        number of output layers.

    Referrences
    -----------
    .. [1] Park, C.W., Kornbluth, M., Vandermause, J. et al.
       Accurate and scalable graph neural network force field
       and molecular dynamics with direct force architecture.
       npj Comput Mater 7, 73 (2021).
    .. [2] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.
    """

    def __init__(
        self,
        n_node_feature: int,
        n_edge_feature: int,
        n_message_passing: int,
        gaussian_filter_end: float,
        share_weights: bool,
        return_intermediate: bool,
        n_output_layers: int,
    ) -> None:
        super().__init__()
        self.gnn = GraphToFeatures(
            n_node_feature,
            n_edge_feature,
            n_message_passing,
            gaussian_filter_end,
            share_weights,
            return_intermediate,
        )
        self.return_intermediate = return_intermediate
        self.output_module = ForceMagnitudeMapping(n_edge_feature, n_output_layers)

    def forward(self, inputs: dict) -> Tensor:
        """
        Forward gnn output through output_module.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        inputs : dict of torch.Tensor
            dictionary of property tensors in unit cell.

        Returns
        -------
        predicted_forces : torch.Tensor
            Predicted values of inter atomic forces with (B x At x 3) shape.
        2 lists of torch.Tensor
            intermediate node and edge embeddings, if return_intermediate=True was used.
        """
        if self.return_intermediate:
            edge_embedding, unit_vecs, node_list, edge_list = self.gnn(inputs)
        else:
            edge_embedding, unit_vecs = self.gnn(inputs)
        force_magnitude = self.output_module(edge_embedding)
        # calculate inter atomic forces vector
        force_magnitude = force_magnitude.expand(-1, -1, -1, 3)
        preditcted_forces = force_magnitude * unit_vecs
        # summation of all neighbors effection
        preditcted_forces = preditcted_forces.sum(dim=2)
        if self.return_intermediate:
            return preditcted_forces, node_list, edge_list
        return preditcted_forces
