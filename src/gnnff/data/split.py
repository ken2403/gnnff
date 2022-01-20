import os
import logging
import numpy as np
from torch.utils.data import Subset


__all__ = ["train_test_split"]


def train_test_split(
    dataset, num_train, num_val, seed=0, split_path=None, logging=None
):
    """
    Splits the dataset into train/validation/test splits, writes split to an npz file and returns subsets.

    Parameters
    ----------
    dataset : gnnff.data.CellData
        full batch of cell dataset.
    num_train : int or None
        number of training examples
    num_val : int or None
        number of validation examples
    split_path : str or None
        Path to split file. Split file will be created where the generated split is stored.
    logging : logging
        logger

    Returns
    -------
    train : torch.utils.data.Subset
        subset with training data.
    val : torch.utils.data.Subset
        subset with validation data.
    test : torch.utils.data.Subset
        subset with test data.
    """
    if split_path is not None:
        split_path = os.path.abspath(split_path)
        split_file = os.path.join(split_path, "split.npz")
        if os.path.exists(split_file):
            if logging:
                logging.info("loading exisiting split file ...")
            file = np.load(split_file)
            train_idx = file["train_idx"].tolist()
            val_idx = file["val_idx"].tolist()
            test_idx = file["test_idx"].tolist()
        else:
            if logging:
                logging.info("make new split file ...")
            # set random state
            rs = np.random.RandomState(seed)
            num_all_data = len(dataset)
            if num_train + num_val > num_all_data:
                raise ValueError(
                    "Make sure that the sum of 'num_train' and 'num_val' is smaller than the size of the data set"
                )
            indices = np.arange(num_all_data)
            # shuffle indices
            rs.shuffle(indices)
            # get each indices
            train_idx = indices[:num_train].tolist()
            val_idx = indices[num_train : num_train + num_val].tolist()
            test_idx = indices[num_train + num_val :].tolist()
            # save split indices
            np.savez(
                split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
            )

    if split_path is None:
        if num_val is None or num_train is None:
            raise ValueError("Make sure that set 'split_path' or split_num")
        else:
            # set random state
            rs = np.random.RandomState(seed)
            num_all_data = len(dataset)
            if num_train + num_val > num_all_data:
                raise ValueError(
                    "Make sure that the sum of 'num_train' and 'num_val' is smaller than the size of the data set"
                )
            indices = np.arange(num_all_data)
            # shuffle indices
            rs.shuffle(indices)
            # get each indices
            train_idx = indices[:num_train].tolist()
            val_idx = indices[num_train : num_train + num_val].tolist()
            test_idx = indices[num_train + num_val :].tolist()

    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx)
    test = Subset(dataset, test_idx)

    return train, val, test
