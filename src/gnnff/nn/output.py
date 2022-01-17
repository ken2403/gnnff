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
    """

    def __init__(
        self,
        n_edge_feature: int,
        n_layers: int = 2,
        activation=shifted_softplus,
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

    def forward(self, last_edge_embedding: Tensor, unit_vecs: Tensor) -> Tensor:
        """
        Calculates the inter atomic forces.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        last_edge_embedding : torch.Tensor
            calculated edge embedding tensor of (B x At x Nbr x n_edge_features) shape.
        unit_vecs : torch.Tensor
            unit vecs of each edge.

        Returns
        -------
        predicted_forces : torch.Tensor
            predicting inter atomic forces for each atoms. (B x At x 3) shape.
        """
        # calculate force_magnitude from last edge_embedding
        force_magnitude = self.out_net(last_edge_embedding)
        force_magnitude = force_magnitude.expand(-1, -1, -1, 3)
        # predict inter atpmic forces by multiplying the unit vectors
        preditcted_forces = force_magnitude * unit_vecs
        # summation of all neighbors effection
        preditcted_forces = preditcted_forces.sum(dim=2)
        return preditcted_forces


class EnergyMapping(nn.Module):
    """
    From node embedding tensor, calculating the total energy.

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
    """

    def __init__(
        self,
        n_node_feature: int,
        n_layers: int = 2,
        activation=shifted_softplus,
    ) -> None:
        super().__init__()
        n_neurons_list = []
        c_neurons = n_node_feature
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

    def forward(self, last_node_embedding: Tensor) -> Tensor:
        """
        Calculates the total energy of the cell.

        B   :  Batch size
        At  :  Total number of atoms in the batch

        Parameters
        ----------
        last_node_embedding : torch.Tensor
            calculated node embedding of (B x At x Nbr x n_node_features) shape.

        Returns
        -------
        predicted_energy : torch.Tensor
            predicting total energy with (B x 1) shape.
        """
        # calculate atomic energy from last node_embedding
        atomic_energy = self.out_net(last_node_embedding)
        # sumation of all atomic energy in batch
        preditcted_energy = atomic_energy.sum(dim=1)
        return preditcted_energy
