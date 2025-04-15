import importlib.util
import os
import sys
import types

from loguru import logger


def download_from_modelers(model_name: str):
    om_token = os.getenv("OM_TOKEN")
    from openmind_hub import snapshot_download

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


def download_from_modelscope(model_id: str, target_dir: str):
    from modelscope.hub.snapshot_download import HubApi, snapshot_download

    modelscope_token = os.getenv("MODELSCOPE_TOKEN")
    if modelscope_token:
        api = HubApi()
        api.login(modelscope_token)

    _, model_name = model_id.split("/")
    model_path = os.path.join(target_dir, model_name)
    logger.info(f"downloading {model_id} to {model_path}")
    model_path = snapshot_download(model_id, local_dir=model_path)
    return model_path
