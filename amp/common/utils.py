import os

from loguru import logger
from openmind_hub import snapshot_download


def download_from_modelers(model_name: str):
    om_token = os.getenv("OM_TOKEN")
    model_path = snapshot_download(repo_id=model_name, token=om_token)
    logger.info(f"model_path: {model_path}")


if __name__ == "__main__":
    # download_from_modelers("model_temp/bc_14b_1T")
    download_from_modelers("model_temp/14b_omini")
