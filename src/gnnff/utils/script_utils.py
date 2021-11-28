import json
from argparse import Namespace
from logging import exception

from torch._C import ScriptFunction


__all__ = ["ScriptError", "read_from_json"]


class ScriptError(Exception):
    pass


def read_from_json(jsonpath: str) -> Namespace:
    """
    This function reads args from a .json file and returns the content as a namespace dict

    Parameters
    ----------
    jsonpath : str
        path to the .json file

    Returns
    -------
    namespace_dict : Namespace
        namespace object build from the dict stored into the given .json file.

    References
    ----------
    .. [1] https://github.com/atomistic-machine-learning/schnetpack/blob/67226795af55719a7e4565ed773881841a94d130/src/schnetpack/utils/spk_utils.py
    """
    with open(jsonpath) as handle:
        dict = json.loads(handle.read())
        namespace_dict = Namespace(**dict)
    return namespace_dict
