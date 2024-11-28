import importlib
import os
from pathlib import Path
from typing import Callable, Type, Union

from transformers.utils import HF_MODULES_CACHE


def patch_get_class_in_module(class_name: str,
                              module_path: Union[str,
                                                 os.PathLike],
                              func: Callable = None
                              ) -> Type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    name = os.path.normpath(module_path).replace(
        ".py", "").replace(
        os.path.sep, ".")
    module_path = str(Path(HF_MODULES_CACHE) / module_path)
    module = importlib.machinery.SourceFileLoader(
        name, module_path).load_module()
    func(module, name)
    return getattr(module, class_name)
