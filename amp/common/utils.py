import os

import openi
from huggingface_hub import snapshot_download as hf_snapshot_download
from loguru import logger


def modelers2openi(model_id, openi_repo_id: str, if_hf: bool = True):
    _, model_name = model_id.split("/")
    if if_hf:
        from huggingface_hub import snapshot_download
    else:
        from openmind_hub import snapshot_download

    path = snapshot_download(model_id)
    openi.upload_model(openi_repo_id, model_name, path)


def reupload2modelers(model_name: str, owner: str):
    new_model_name = f"{owner}/{model_name.split('/')[1].lower()}"
    local_folder = hf_snapshot_download(repo_id=model_name)

    os.remove(os.path.join(local_folder, "README.md"))
    os.environ["HUB_WHITE_LIST_PATHS"] = os.path.dirname(os.path.dirname(local_folder))
    om_token = os.getenv("OM_TOKEN")
    logger.info(f"Repository downloaded to: {local_folder}")
    from openmind_hub import upload_folder

    upload_folder(
        token=om_token,
        folder_path=local_folder,
        repo_id=new_model_name,
    )
