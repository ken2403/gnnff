The latest implementation of this model has been merged into [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). Please use the PyG version in the future.
- PR: [#5866](https://github.com/pyg-team/pytorch_geometric/pull/5866)
- Doc: [link](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=gnnff#torch_geometric.nn.models.GNNFF)

# Atomistic Force Fields based on [GNNFF](https://www.nature.com/articles/s41524-021-00543-3)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

![model](./docs/image/model.jpeg)

GNNFF [1] is a graph neural network framework to directly predict atomic forces from automatically extracted features of the local atomic environment that are translationally-invariant, but rotationally-covariant to the coordinate of the atoms.
This package is an atomistic force fields that constructed based on GNNFF.

## References

- [1] C. W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J. P. Mailoa, *Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture.* npj Computational Materials. **7**, 1–9 (2021). [link](https://www.nature.com/articles/s41524-021-00543-3)
