import importlib
import os
from pathlib import Path
from typing import Callable, Type, Union

from loguru import logger
from transformers.utils import HF_MODULES_CACHE
from transformers.dynamic_module_utils import create_dynamic_module, init_hf_modules
import os.path as osp
from amp.module_patcher import AMP_DIR

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

def patch_create_dynamic_module(name: Union[str, os.PathLike]):
    """
    Creates a dynamic module in the cache directory for modules.

    Args:
        name (`str` or `os.PathLike`):
            The name of the dynamic module to create.
    """
    init_hf_modules()
    dynamic_module_path = Path(osp.join(AMP_DIR, "models", name)).resolve()
    logger.info(f"dynamic_module_path: {dynamic_module_path}")

    # If the parent module does not exist yet, recursively create it.
    if not dynamic_module_path.parent.exists():
        patch_create_dynamic_module(dynamic_module_path.parent)
    os.makedirs(dynamic_module_path, exist_ok=True)
    init_path = dynamic_module_path / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        # It is extremely important to invalidate the cache when we change stuff in those modules, or users end up
        # with errors about module that do not exist. Same for all other `invalidate_caches` in this file.
        importlib.invalidate_caches()
