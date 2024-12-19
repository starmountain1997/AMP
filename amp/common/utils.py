import os
import os.path as osp
import platform
import re
import subprocess
import time
from datetime import datetime
from functools import wraps

import openi
import torch_npu
from huggingface_hub import snapshot_download as hf_snapshot_download
from loguru import logger


def modelers2openi(model_id, openi_repo_id: str, path: str = None, if_hf: bool = True):
    # _, model_name = model_id.split("/")
    model_name = "AMP_model_q4p5"
    if path is None:
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


def inference_test(test_times: int, warm_up_times: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(warm_up_times):
                logger.debug(f"running warmup iter {i+1}/{warm_up_times}")
                func(*args, **kwargs)

            start = time.time()
            for i in range(test_times):
                logger.debug(f"running test iter {i+1}/{test_times}")
                func(*args, **kwargs)
            end = time.time()

            avg_time = (end - start) / test_times
            logger.info(
                f"Average runtime over {test_times} iterations: {avg_time:.6f} seconds"
            )
            return avg_time  # Optionally return the average runtime

        return wrapper

    return decorator


def find_latest_prof_folder(base_path):
    folder_pattern = re.compile(
        rf"^{platform.node()}_([0-9]+)_([0-9]{{17}})_ascend_pt$"
    )

    latest_folder = None
    latest_time = None

    for folder_name in os.listdir(base_path):
        match = folder_pattern.match(folder_name)
        if match:
            utc_time = match.group(2)
            try:
                # Parse the UTC time
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


def inference_prof(
    prof_save_path: str = "./",
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            prof = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    prof_save_path
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            )
            prof.start()
            func(*args, **kwargs)
            prof.stop()

        return wrapper

    return decorator


def run_advisor(prof_save_path: str = "./"):
    prof_folder = find_latest_prof_folder(prof_save_path)
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
    """
    task_1_prof_folder = find_latest_prof_folder(task_1)
    task_2_prof_folder = find_latest_prof_folder(task_2)
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
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login("4d04a47d-b3d8-4f1d-95a2-791aec924262")

    path = snapshot_download(
        "ZhipuAI/cogagent2-9b", cache_dir="/mnt/data00/guozr/modelscope"
    )
    modelers2openi(
        "THUDM/cogagent2-9b", openi_repo_id="starmountain1997/AMP", path=path
    )
