import torch
from torch import Tensor
import torch.nn as nn
from gnnff.data.keys import Keys
from gnnff.nn import output
from gnnff.nn.gnn import GraphToFeatures
from gnnff.nn.output import ForceMagnitudeMapping


class GNNFF(nn.Module):
    """
    GNNFF architecture for learning inter atomic interactions of atomistic systems and predict the inter atomic forces.

    Attributes
    ----------
    n_node_feature : int, default=128
        dimension of the embedded node features.
    n_edge_feature : int, default=128
        dimension of the embedded edge features.
    n_message_passing : int, default=3
        number of message passing layers.
    cutoff : float, default=8.0
        cutoff radius.
    gaussian_filter_end : float or None, default=None
        center of last Gaussian function.
    share_weights : bool, default=False
        if True, share the weights across all message passing layers.
    return_intermediate : bool, default=False
        if True, `forward` method also returns intermediate atomic representations
        after each message passing is applied.
    n_output_layers : int, default=2
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
    .. [3] M. Gastegger, L. Schwiedrzik, M. Bittermann, F. Berzsenyi, and P. Marquetand ,
       "wACSF—Weighted atom-centered symmetry functions as descriptors in machine learning potentials",
       J. Chem. Phys. 148, 241709 (2018)
    """

    def __init__(
        self,
        n_node_feature: int = 128,
        n_edge_feature: int = 128,
        n_message_passing: int = 3,
        cutoff: float = 6.0,
        gaussian_filter_end: float = None,
        share_weights: bool = False,
        return_intermediate: bool = False,
        n_output_layers: int = 2,
    ) -> None:
        super().__init__()
        # implementation of gaussian_filter_end (ref: [3])
        if gaussian_filter_end is None:
            gaussian_filter_end = cutoff - 0.5
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
        # from graph, calculating the force magnitude of each edge.
        if self.return_intermediate:
            edge_embedding, node_list, edge_list = self.gnn(inputs)
        else:
            edge_embedding = self.gnn(inputs)
        force_magnitude = self.output_module(edge_embedding)

        # calculate inter atomic forces vector
        force_magnitude = force_magnitude.expand(-1, -1, -1, 3)
        unit_vecs = inputs[Keys.unit_vecs]
        preditcted_forces = force_magnitude * unit_vecs
        # summation of all neighbors effection
        preditcted_forces = preditcted_forces.sum(dim=2)
        if self.return_intermediate:
            return preditcted_forces, node_list, edge_list
        return preditcted_forces
