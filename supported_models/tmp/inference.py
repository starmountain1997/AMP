import argparse

from transformers import AutoTokenizer, AutoModel
from openmind_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model",
        default=None,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model_name_or_path:
        model_path = args.model_name_or_path
    else:
        model_path = snapshot_download("zhipuai/chatglm3-6b-32k", revision="main", resume_download=True,
                                       ignore_patterns=["*.h5", "*.ot", "*.msgpack"])

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to("npu")
    model = model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)


if __name__ == "__main__":
    main()
