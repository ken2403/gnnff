import numpy as np


__all__ = ["ScriptError", "count_params"]


class ScriptError(Exception):
    pass


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
