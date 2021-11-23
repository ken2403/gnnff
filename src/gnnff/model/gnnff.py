import torch
from torch import Tensor
import torch.nn as nn
from gnnff.data.keys import Keys
from gnnff.nn.gnn import GraphToFeatures
from gnnff.nn.output import OutputModuleError, ForceMagnitudeMapping, EnergyMapping


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
    output_activation : collable or None, default=torch.nn.functional.softplus
        activation function for output layers. All hidden layers would the same activation function
        except the last layer that does not apply any activation function.
    property : str, default="forces"
        name of the output property. Choose "forces" or "energy".
    n_output_layers : int, default=2
        number of output layers.
    device : torch.device, default=torch.device("cpu")
        computing device.

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
        output_activation=nn.functional.softplus,
        property: str = "forces",
        n_output_layers: int = 2,
        device: torch.device = torch.device("cpu"),
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
            device=device,
        )
        self.return_intermediate = return_intermediate
        if property == "forces":
            self.output_module = ForceMagnitudeMapping(
                n_edge_feature,
                n_output_layers,
                activation=output_activation,
                property=property,
            )
        elif property == "energy":
            self.output_module = EnergyMapping(
                n_node_feature,
                n_edge_feature,
                n_output_layers,
                activation=output_activation,
                property=property,
            )
        elif property != "forces" and property != "energy":
            raise OutputModuleError(
                "Invalid property ({})! Please set the property parameter from 'energy' or 'forces'.".format(
                    property
                )
            )
        self.property = property

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
        result : dict of torch.Tensor
            dict of the predicted value of some property.
        2 lists of torch.Tensor
            intermediate node and edge embeddings, if return_intermediate=True was used.
        """
        # from graph, calculating the inter atomic interaction
        if self.return_intermediate:
            node_embedding, edge_embedding, node_list, edge_list = self.gnn(inputs)
        else:
            node_embedding, edge_embedding = self.gnn(inputs)
        # from node and edge, calculateing the propety magnitude
        if self.property == "forces":
            result = self.output_module(edge_embedding)
        if self.property == "energy":
            result = self.output_module(node_embedding, edge_embedding)

        # calculate inter atomic forces vector
        if self.property == "forces":
            force_magnitude = result[self.property]
            force_magnitude = force_magnitude.expand(-1, -1, -1, 3)
            unit_vecs = inputs[Keys.unit_vecs]
            preditcted_forces = force_magnitude * unit_vecs
            # summation of all neighbors effection
            preditcted_forces = preditcted_forces.sum(dim=2)
            result[self.property] = preditcted_forces
        # calculate the total energy
        if self.property == "energy":
            # TODO: 原子で和を取る
            energy_mapping = result[self.property]

        if self.return_intermediate:
            return result, node_list, edge_list
        return result
