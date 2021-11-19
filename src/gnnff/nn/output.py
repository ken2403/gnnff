import torch
from torch import Tensor
import torch.nn as nn
from gnnff.nn import Dense


__all__ = ["ForceMagnitudeMapping"]


class ForceMagnitudeMapping(nn.Module):
    """
    From edge embedding tensor, calculating the force magnitude of all inter atomic forces.

    Attributes
    ----------
    n_edge_feature : int
        dimension of the embedded edge features.
    n_layers : int, default=2
        number of output layers.
    activation : collable or None, default=torch.nn.functional.softplus
        activation function. All hidden layers would the same activation function
        except the output layer that does not apply any activation function.
    """

    def __init__(
        self, n_edge_feature: int, n_layers: int = 2, activation=nn.functional.softplus
    ) -> None:
        super().__init__()
        n_neurons_list = []
        c_neurons = n_edge_feature
        for i in range(n_layers):
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

    def forward(self, edge_embedding: Tensor) -> Tensor:
        """
        Calculates the force magnitude.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom

        Parameters
        ----------
        edge_embedding : torch.Tensor
            batch of edge embedding tensor of (B x At x Nbr x n_edge_feature) shape.

        Returns
        -------
        torch.Tensor
            force magnitude for each pair of neighboring atoms. (B x At x Nbr x 1) shape.
        """
        return self.out_net(edge_embedding)
