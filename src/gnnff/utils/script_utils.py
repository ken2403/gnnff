import time
import random

import numpy as np
import torch


__all__ = ["ScriptError", "set_random_seed", "count_params"]


class ScriptError(Exception):
    pass


def set_random_seed(seed: int, logging=None):
    """
    This function sets the random seed (if given) or creates one for torch and numpy random state initialization

    Parameters
    ----------
    seed : int or None, default=None
        if seed not present, it is generated based on time
    loggin : logging
        logger
    """
    if seed is None:
        seed = int(time.time() * 1000.0)
        # Reshuffle current time to get more different seeds within shorter time intervals
        # Taken from https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        # & Gets overlapping bits, << and >> are binary right and left shifts
        seed = (
            ((seed & 0xFF000000) >> 24)
            + ((seed & 0x00FF0000) >> 8)
            + ((seed & 0x0000FF00) << 8)
            + ((seed & 0x000000FF) << 24)
        )
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if logging is not None:
        logging.info("Random state initialized with seed {:<10d}".format(seed))


def count_params(model):
    """
    This function takes a model as an input and returns the number of
    trainable parameters.

    Parameters
    ----------
    model : torch.nn.Module
        model for which you want to count the trainable parameters.

    Returns
    -------
    params : int
        number of trainable parameters for the model.
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
