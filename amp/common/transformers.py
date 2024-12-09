import hashlib
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, List, Optional, Type, Union

import transformers
from loguru import logger
from transformers.dynamic_module_utils import get_imports, get_relative_imports
from transformers.utils import HF_MODULES_CACHE


# changed in e1c2b69, tag 4.45.0
def patch_get_class_in_module(
    class_name: str,
    module_path: Union[str, os.PathLike],
    func: Callable = None,
    force_reload: bool = False,
) -> Type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    if transformers.__version__ < "4.45.0":
        # 无法确定 openmind.hf.npu_fused_ops.modeling_utils.om_get_class_in_module 是否给低版本做了适配
        # 因此不建议和openmind一起使用
        name = (
            os.path.normpath(module_path).replace(".py", "").replace(os.path.sep, ".")
        )
        module_path = str(Path(HF_MODULES_CACHE) / module_path)
        module = importlib.machinery.SourceFileLoader(name, module_path).load_module()
    else:
        from transformers.dynamic_module_utils import (
            _HF_REMOTE_CODE_LOCK, get_relative_import_files)

        name = os.path.normpath(module_path)
        if name.endswith(".py"):
            name = name[:-3]
        name = name.replace(os.path.sep, ".")
        module_file: Path = Path(HF_MODULES_CACHE) / module_path
        with _HF_REMOTE_CODE_LOCK:
            if force_reload:
                sys.modules.pop(name, None)
                importlib.invalidate_caches()
            cached_module: Optional[ModuleType] = sys.modules.get(name)
            module_spec = importlib.util.spec_from_file_location(
                name, location=module_file
            )

            # Hash the module file and all its relative imports to check if we
            # need to reload it
            module_files: List[Path] = [module_file] + sorted(
                map(Path, get_relative_import_files(module_file))
            )
            module_hash: str = hashlib.sha256(
                b"".join(bytes(f) + f.read_bytes() for f in module_files)
            ).hexdigest()

            module: ModuleType
            if cached_module is None:
                module = importlib.util.module_from_spec(module_spec)
                # insert it into sys.modules before any loading begins
                sys.modules[name] = module
            else:
                module = cached_module
            # reload in both cases, unless the module is already imported and
            # the hash hits
            if getattr(module, "__transformers_module_hash__", "") != module_hash:
                module_spec.loader.exec_module(module)
                module.__transformers_module_hash__ = module_hash
    func(module, name)
    return getattr(module, class_name)


@DeprecationWarning
def patch_check_imports(filename: Union[str, os.PathLike]) -> List[str]:
    """
    Check if the current Python environment contains all the libraries that are imported in a file. Will raise if a
    library is missing.

    Args:
        filename (`str` or `os.PathLike`): The module file to check.

    Returns:
        `List[str]`: The list of relative imports in the file.
    忽略import error
    """
    imports = get_imports(filename)
    missing_packages = []
    for imp in imports:
        try:
            importlib.import_module(imp)
        except ImportError:
            missing_packages.append(imp)

    if len(missing_packages) > 0:
        logger.warning(
            "This modeling file requires the following packages that were not found in your environment: "
            f"{', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`"
        )

    return get_relative_imports(filename)
