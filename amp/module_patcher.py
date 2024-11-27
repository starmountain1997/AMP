import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Type, Union

from transformers.utils import HF_MODULES_CACHE

# https://peps.python.org/pep-0369/

_post_import_hooks = defaultdict(list)
_target_modules = {
    "transformers.models.llama.modeling_llama",
    "transformers.models.qwen2.modeling_qwen2",
    # "transformers.generation.logits_process",
    "transformers.dynamic_module_utils",
    "deepspeed.ops.op_builder",
}  # Whitelist of modules to monitor for post-import hooks


class PostImportFinder:
    def __init__(self):
        self._skip = set()

    def find_spec(self, fullname, path=None, target=None):
        # Only handle modules in the whitelist
        if fullname not in _target_modules or fullname in self._skip:
            return None
        self._skip.add(fullname)
        return importlib.machinery.ModuleSpec(fullname, PostImportLoader(self))


class PostImportLoader:
    def __init__(self, finder):
        self._finder = finder

    def create_module(self, spec):
        # Use the default module creation
        return None

    def exec_module(self, module):
        fullname = module.__name__
        # Execute the actual module code
        if fullname in sys.modules:
            module = sys.modules[fullname]
        else:
            importlib.import_module(fullname)
            module = sys.modules[fullname]

        # Call post-import hooks
        for func in _post_import_hooks[fullname]:
            func(module)

        # Remove the module from the skip list
        self._finder._skip.remove(fullname)


def patch_get_class_in_module(
        class_name: str, module_path: Union[str, os.PathLike], func: Callable) -> Type:
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


def when_imported(fullname):
    """
    Register a hook to be executed after the specified module is imported.
    """
    def decorator(func):
        if fullname in sys.modules:
            func(sys.modules[fullname])
        else:
            _post_import_hooks[fullname].append(func)
            _target_modules.add(fullname)  # Add to the whitelist
        return func
    return decorator


# Insert the finder into sys.meta_path
if not any(isinstance(finder, PostImportFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, PostImportFinder())
