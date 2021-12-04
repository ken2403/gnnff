from torch import Tensor
from torch.functional import tensordot
import torch.nn as nn

from gnnff.data.keys import Keys
from gnnff.nn.functional import shifted_softplus
from gnnff.nn.base import Dense


__all__ = ["OutputModuleError", "ForceMapping", "EnergyMapping"]


class OutputModuleError(Exception):
    pass


class ForceMapping(nn.Module):
    """
    From edge embedding tensor, calculating the force magnitude of all inter atomic forces.
    And then, calculate the inter atomic forces by multiplying unit vectors.

    Attributes
    ----------
    n_edge_feature : int
        dimension of the embedded edge features.
    n_layers : int, default=2
        number of output layers.
    activation : collable or None, default=gnnff.nn.activation.shifted_softplus
        activation function. All hidden layers would the same activation function
        except the output layer that does not apply any activation function.
    property_name : str, default="forces"
        name of the output property.
    """

    def __init__(
        self,
        n_edge_feature: int,
        n_layers: int = 2,
        activation=shifted_softplus,
        property_name: str = "forces",
    ) -> None:
        super().__init__()
        n_neurons_list = []
        c_neurons = n_edge_feature
        for _ in range(n_layers):
            n_neurons_list.append(c_neurons)
            c_neurons = max(1, c_neurons // 2)
        # The output layer has 1 neurons.
        n_neurons_list.append(1)
        layers = [
            Dense(n_neurons_list[i], n_neurons_list[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]
        layers.append(Dense(n_neurons_list[-2], n_neurons_list[-1], activation=None))
        self.out_net = nn.Sequential(*layers)
        self.property_name = property_name

    def forward(self, inputs: dict) -> Tensor:
        """
        Calculates the inter atomic forces.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        inputs : dict of torch.Tensor
            dictionary of property tensors in unit cell.
            This should retain calculated embedding.

        Returns
        -------
        predicted_forces : torch.Tensor
            predicting inter atomic forces for each atoms. (B x At x 3) shape.
        """
        # calculate force_magnitude from last edge_embedding
        force_magnitude = self.out_net(inputs["last_edge_embedding"])
        force_magnitude = force_magnitude.expand(-1, -1, -1, 3)
        # predict inter atpmic forces by multiplying the unit vectors
        unit_vecs = inputs[Keys.unit_vecs]
        preditcted_forces = force_magnitude * unit_vecs
        # summation of all neighbors effection
        preditcted_forces = preditcted_forces.sum(dim=2)
        return preditcted_forces


class EnergyMapping(nn.Module):
    """
    From node and edge embedding tensor, calculating the total energy.

    Attributes
    ----------
    n_node_feature : int
        dimension of the embedded node features.
    n_edge_feature : int
        dimension of the embedded edge features.
    n_layers : int, default=2
        number of output layers.
    activation : collable or None, default=gnnff.nn.activation.shifted_softplus
        activation function. All hidden layers would the same activation function
        except the output layer that does not apply any activation function.
    property_name : str, default="energy"
        name of the output property.
    """

    def __init__(
        self,
        n_node_feature: int,
        n_edge_feature: int,
        n_layers: int = 2,
        activation=shifted_softplus,
        property_name: str = "energy",
    ) -> None:
        super().__init__()
        n_neurons_list = []
        c_neurons = n_edge_feature
        for _ in range(n_layers):
            n_neurons_list.append(c_neurons)
            c_neurons = max(1, c_neurons // 2)
        # The output layer has 1 neurons.
        n_neurons_list.append(1)
        layers = [
            Dense(n_neurons_list[i], n_neurons_list[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]
        layers.append(Dense(n_neurons_list[-2], n_neurons_list[-1], activation=None))
        self.out_net = nn.Sequential(*layers)
        self.property_name = property_name

    def forward(self, inputs: dict) -> Tensor:
        """
        Calculates the total energy of the cell.

        B   :  Batch size

        Parameters
        ----------
        inputs : dict of torch.Tensor
            dictionary of property tensors in unit cell.
            This should retain calculated node and edge embedding.

        Returns
        -------
        predicted_energy : torch.Tensor
            predicting total energy for each atoms. (B x 1) shape.
        """
