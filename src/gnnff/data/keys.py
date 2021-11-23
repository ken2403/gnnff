__all__ = ["Keys"]


class Keys:
    """
    Keys to access structure properties in `CrystalData` object.

    References
    ----------
    .. [1] https://github.com/atomistic-machine-learning/schnetpack/blob/67226795af/src/schnetpack/__init__.py
    """

    # geometry
    Z = "_atomic_numbers"
    charge = "_charge"
    atom_mask = "_atom_mask"
    R = "_positions"
    cell = "_cell"
    pbc = "_pbc"
    neighbors = "_neighbors"
    neighbor_mask = "_neighbor_mask"
    cell_offset = "_cell_offset"
    distances = "_distances"
    unit_vecs = "_unit_vecs"
    n_atoms = "_n_atoms"

    # chemical properties
    energy = "energy"
    forces = "forces"
