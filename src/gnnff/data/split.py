import os
import numpy as np
from torch.utils.data import Subset


__all__ = ["train_test_split"]


def train_test_split(dataset, num_train, num_val, seed=None, split_path=None):
    """
    Splits the dataset into train/validation/test splits, writes split to an npz file and returns subsets.

    Parameters
    ----------
    dataset : gnnff.data.CellData
        full batch of cell dataset.
    num_train : int
        number of training examples
    num_val : int
        number of validation examples
    split_path : str
        Path to split file. Split file will be created where the generated split is stored.

    Returns
    -------
    train : torch.utils.data.Subset
        subset with training data.
    val : torch.utils.data.Subset
        subset with validation data.
    test : torch.utils.data.Subset
        subset with test data.
    """
    # set random state
    if seed is not None:
        rs = np.random.RandomState(seed)
    else:
        rs = np.random.RandomState(0)

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

    if split_path is not None:
        split_file = os.path.join(split_path, "split.npz")
        np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx)
    test = Subset(dataset, test_idx)

    return train, val, test
