import os
from typing import List
import json
from enum import Enum

class Dataset(Enum):
    FULL_CEVAL="full_CEval"

def chat_test(model_path:str, mindie_path:str):
    commands=[]
    commands.append(f"bash run.sh pa_bf16 full_CEval 5 16 chatglm {model_path} 2")
    return " && \\\n".join(commands)

def accuracy_test(model_paths:List[str], dataset_name:Dataset, mindie_path:str):
    commands=[]
    model_test_path=os.path.join(mindie_path, "examples/atb_models/tests/modeltest/")
    for model_path in model_paths:
        model_type=json.load(open(os.path.join(model_test_path, "config.json")))["model_type"]
        commands.append(f"bash run.sh pa_bf16 {dataset_name} 5 16 {model_type} {model_path} 2")
    return " && \\\n".join(commands)




