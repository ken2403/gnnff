from torch import Tensor
import torch.nn as nn

from gnnff.data.keys import Keys
from gnnff.nn.gnn import GraphToFeatures
from gnnff.nn.functional import shifted_softplus
from gnnff.nn.output import OutputModuleError, ForceMapping, EnergyMapping


class GNNFF(nn.Module):
    """
    GNNFF architecture for learning inter atomic interactions of atomistic systems and predict
    inter atomic forces or total energy.

    Attributes
    ----------
    n_node_feature : int, default=128
        dimension of the embedded node features.
    n_edge_feature : int, default=20
        dimension of the embedded edge features.
    n_message_passing : int, default=3
        number of message passing layers.
    cutoff : float, default=6.0
        cutoff radius.
    gaussian_filter_end : float or None, default=None
        center of last Gaussian function.
        if None, use cutoff-0.5 as gaussian_filter_end.
    trainble_gaussian : bool, default=False
        If True, widths and offset of gaussian_filter are adjusted during training.
    share_weights : bool, default=False
        if True, share the weights across all message passing layers.
    return_intermid : bool, default=False
        if True, `forward` method also returns intermediate atomic representations
        after each message passing is applied.
    output_activation : collable or None, default=gnnff.nn.activation.shifted_softplus
        activation function for output layers. All hidden layers would the same activation function
        except the last layer that does not apply any activation function.
    property : dict of property and property_name, default={"forces": "forces", "energy": None}
        name of the output property. Set "forces" and "energy" values.
    n_output_layers : int, default=2
        number of output layers.

    Referrences
    -----------
    .. [1] Park, C.W., Kornbluth, M., Vandermause, J. et al.
       "Accurate and scalable graph neural network force field
       and molecular dynamics with direct force architecture."
       npj Comput Mater 7, 73 (2021).
    .. [2] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       "SchNet - a deep learning architecture for molceules and materials."
       The Journal of Chemical Physics 148 (24), 241722. 2018.
    .. [3] Tian Xie and Jeffrey C. Grossman,
        "Crystal Graph Convolutional Neural Networks for an Accurate and
        Interpretable Prediction of Material Properties"
        Phys. Rev. Lett. 120, 145301 (2018)
    .. [4] M. Gastegger, L. Schwiedrzik, M. Bittermann, F. Berzsenyi, and P. Marquetand ,
       "wACSF—Weighted atom-centered symmetry functions as descriptors in machine learning potentials",
       J. Chem. Phys. 148, 241709 (2018)
    """

    def __init__(
        self,
        n_node_feature: int = 128,
        n_edge_feature: int = 20,
        n_message_passing: int = 3,
        cutoff: float = 6.0,
        gaussian_filter_end: float = None,
        trainable_gaussian: bool = False,
        share_weights: bool = False,
        return_intermid: bool = False,
        output_activation=shifted_softplus,
        properties: dict = {"forces": "forces", "energy": None},
        n_output_layers: int = 2,
    ) -> None:
        super().__init__()
        # implementation of gaussian_filter_end (ref: [4])
        if gaussian_filter_end is None:
            gaussian_filter_end = cutoff - 0.5
        self.gnn = GraphToFeatures(
            n_node_feature=n_node_feature,
            n_edge_feature=n_edge_feature,
            n_message_passing=n_message_passing,
            gaussian_filter_end=gaussian_filter_end,
            trainable_gaussian=trainable_gaussian,
            share_weights=share_weights,
            return_intermid=return_intermid,
        )
        self.return_intermid = return_intermid
        if properties["forces"] is not None:
            self.output_force = ForceMapping(
                n_edge_feature,
                n_output_layers,
                activation=output_activation,
            )
        if properties["energy"] is not None:
            self.output_energy = EnergyMapping(
                n_node_feature,
                n_output_layers,
                activation=output_activation,
            )
        if properties["forces"] is None and properties["energy"] is None:
            raise OutputModuleError(
                "Invalid property key ({})! Please set the property key from 'forces' or 'energy'.".format(
                    properties.keys()
                )
            )
        self.properties = properties

    def forward(self, inputs: dict) -> Tensor:
        """
        layer output of GNNFF.

        Parameters
        ----------
        inputs : dict of torch.Tensor
            dictionary of property tensors in unit cell.

        Returns
        -------
        result : dict of torch.Tensor
            dict of the predicted value of some property.
            intermediate node and edge embeddings are also contanined, if 'return_intermediate=True' was used.
        """
        # from graph, calculating the inter atomic interaction
        if self.return_intermid:
            (
                inputs["last_node_embedding"],
                inputs["last_edge_embedding"],
                node_list,
                edge_list,
            ) = self.gnn(inputs)
        else:
            (
                inputs["last_node_embedding"],
                inputs["last_edge_embedding"],
            ) = self.gnn(inputs)
        # from embedding node and edge, calculating the propety.
        result = {}
        if self.properties["forces"] is not None:
            result[self.properties["forces"]] = self.output_force(
                inputs["last_edge_embedding"], inputs[Keys.unit_vecs]
            )
        elif self.properties["energy"] is not None:
            result[self.properties["energy"]] = self.output_energy(
                inputs["last_node_embedding"]
            )

        if self.return_intermid:
            result["node_list"], result["edge_list"] = node_list, edge_list
            return result
        return result
