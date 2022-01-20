import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from gnnff.data.keys import Keys
from gnnff.data.split import train_test_split


__all__ = ["get_loader"]


def get_loader(dataset, args, logging=None):
    """
    Parameters
    ----------
    dataset : gnnff.data.Celldata
        dataset of cell.
    args : Namespace
        Namespace dict.
    logging : logging
        logger

    Returns
    -------
    train_data, val_loader, test_loader : torch.utils.data.DataLoader
    """
    train_data, val_data, test_data = train_test_split(
        dataset, *args.split, split_path=args.modelpath, logging=logging
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_data),
        num_workers=4,
        pin_memory=args.cuda,
        collate_fn=_collate_aseatoms,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=args.cuda,
        collate_fn=_collate_aseatoms,
    )
    if len(test_data) != 0:
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=args.cuda,
            collate_fn=_collate_aseatoms,
        )
    elif len(test_data) == 0:
        test_loader = None
    return train_loader, val_loader, test_loader


def _collate_aseatoms(examples):
    """
    Build batch from systems and properties & apply padding

    Parameters
    ----------
    examples : list

    Returns
    -------
    dict : [str->torch.Tensor]
        mini-batch of atomistic systems

    References
    ----------
    .. [1] https://github.com/ken2403/schnetpack/blob/master/src/schnetpack/data/loader.py
    """
    properties = examples[0]

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int32) for prop, val in properties.items()
    }

    # get maximum sizes
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int32)
            )

    # initialize batch
    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
            examples[0][p].type()
        )
        for p, size in max_size.items()
    }
    has_atom_mask = Keys.atom_mask in batch.keys()
    has_neighbor_mask = Keys.neighbor_mask in batch.keys()

    if not has_neighbor_mask:
        batch[Keys.neighbor_mask] = torch.zeros_like(batch[Keys.neighbors]).float()
    if not has_atom_mask:
        batch[Keys.atom_mask] = torch.zeros_like(batch[Keys.Z]).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        if not has_neighbor_mask:
            nbh = properties[Keys.neighbors]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch[Keys.neighbor_mask][s] = mask
            batch[Keys.neighbors][s] = nbh * mask.long()

        if not has_atom_mask:
            z = properties[Keys.Z]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Keys.atom_mask][s] = z > 0

    return batch
