import argparse
import os
import os.path as osp
import platform
import re
import subprocess
from datetime import datetime
from functools import wraps

import torch_npu
from huggingface_hub import snapshot_download as hf_snapshot_download
from loguru import logger
from openi import upload_model
from openmind_hub import snapshot_download as om_snapshot_download
from openmind_hub import upload_folder


def modelers2openi(model_id, openi_repo_id: str, path: str = None, if_hf: bool = False):
    logger.info(f"Uploading model {model_id} to Openi repository {openi_repo_id}")
    _, model_name = model_id.split("/")
    if path is None:
        if if_hf:
            from huggingface_hub import snapshot_download
        else:
            from openmind_hub import snapshot_download
        path = snapshot_download(model_id)
    upload_model(openi_repo_id, model_name, path)


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


def _find_latest_prof_folder(base_path):
    folder_pattern = re.compile(rf"^{platform.node()}_([0-9]+)_([0-9]{{17}})_ascend_pt$")
    latest_folder = None
    latest_time = None
    for folder_name in os.listdir(base_path):
        match = folder_pattern.match(folder_name)
        if match:
            utc_time = match.group(2)
            try:
                utc_dt = datetime.strptime(utc_time, "%Y%m%d%H%M%S%f")

                if latest_time is None or utc_dt > latest_time:
                    latest_time = utc_dt
                    latest_folder = folder_name
            except ValueError:
                print(f"Skipping invalid folder name: {folder_name}")
                continue

    if latest_folder:
        return latest_folder
    raise ValueError(f"No valid folder found under the base path: {base_path}.")



def run_advisor(prof_save_path: str = "./"):
    """
    运行advisor
    """
    prof_folder = _find_latest_prof_folder(prof_save_path)
    command = [
        "msprof-analyze",
        "advisor",
        "all",
        "-d",
        "ASCEND_PROFILER_OUTPUT",
    ]
    logger.info(" ".join(command))
    subprocess.run(command, cwd=os.path.join(prof_save_path, prof_folder))


def run_compare(task_1: str, task_2: str, save_path: str):
    """
    https://gitee.com/ascend/mstt/tree/master/profiler/compare_tools
    比较
    """
    task_1_prof_folder = _find_latest_prof_folder(task_1)
    task_2_prof_folder = _find_latest_prof_folder(task_2)
    command = [
        "msprof-analyze",
        "compare",
        "-d",
        osp.join(task_1_prof_folder, "ASCEND_PROFILER_OUTPUT"),
        "-d",
        osp.join(task_2_prof_folder, "ASCEND_PROFILER_OUTPUT"),
        "--output_path=./",
    ]
    logger.info(" ".join(command))
    subprocess.run(command, cwd=save_path)


if __name__ == "__main__":
    # modelers2openi("zyl9737/deepseek-coder-6.7b-instruct","starmountain1997/AMP")
    # modelers2openi("libo2024/Yi-9B-200K","starmountain1997/AMP")
    # modelers2openi("ccpower/Bunny-Llama-3-8B-V","starmountain1997/AMP")
    modelers2openi("ltdog/Qwen1.5-32B", "starmountain1997/AMP")
    # main()
