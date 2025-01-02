import os
import os.path as osp
import platform
import re
import subprocess
from datetime import datetime

from loguru import logger


def _find_latest_prof_folder(base_path: str) -> str:
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
