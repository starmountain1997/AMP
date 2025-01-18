import importlib.util
import os
import sys
import types

from loguru import logger
from openmind_hub import snapshot_download


def download_from_modelers(model_name: str):
    om_token = os.getenv("OM_TOKEN")
    model_path = snapshot_download(repo_id=model_name, token=om_token)
    logger.info(f"model_path: {model_path}")
    return model_path


def create_dummy_module(name, attributes=None):
    """
    Creates a dummy module with the given name and optional attributes.
    """
    module = types.ModuleType(name)
    if attributes:
        for attr_name, attr_value in attributes.items():
            setattr(module, attr_name, attr_value)
    spec = importlib.util.spec_from_loader(name, loader=None)
    module.__spec__ = spec
    sys.modules[name] = module
    return module


if __name__ == "__main__":
    download_from_modelers("model_temp/7b_omini")
