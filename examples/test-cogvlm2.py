import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from amp.common.prof import measure_performance
from amp.models.cogvlm2_llama3_chat_19b import patch_cogvlm2_llama3_chat_19b

print(patch_cogvlm2_llama3_chat_19b.__name__)

MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
DEVICE = "npu:0"
TORCH_TYPE = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
    )
    .to(DEVICE)
    .eval()
)

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

image_path = "/home/guozr/CODE/AMP/tests/cat.jpg"
if image_path == "":
    print(
        "You did not enter image path, the following will be a plain text conversation."
    )
    image = None
    text_only_first_query = True
else:
    image = Image.open(image_path).convert("RGB")

history = []

query = "can you describe the image?"

if image is None:
    if text_only_first_query:
        query = text_only_template.format(query)
        text_only_first_query = False
    else:
        old_prompt = ""
        for _, (old_query, response) in enumerate(history):
            old_prompt += old_query + " " + response + "\n"
        query = old_prompt + "USER: {} ASSISTANT:".format(query)
if image is None:
    input_by_model = model.build_conversation_input_ids(
        tokenizer, query=query, history=history, template_version="chat"
    )
else:
    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=query,
        history=history,
        images=[image],
        template_version="chat",
    )
inputs = {
    "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
    "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
    "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
    "images": (
        [[input_by_model["images"][0].to(DEVICE).to(TORCH_TYPE)]]
        if image is not None
        else None
    ),
}
gen_kwargs = {
    "max_new_tokens": 2048,
    "pad_token_id": 128002,
}
inputs.update(gen_kwargs)
with torch.no_grad():
    outputs = model.generate(**inputs)
    # with patch_generate_with_profiler(model):
    #     outputs = model.generate(**inputs)
    measure_performance(model, tokenizer, 1024, inputs)

    outputs = outputs[:, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(outputs[0])
    response = response.split("<|end_of_text|>")[0]
    print("\nCogVLM2:", response)
history.append((query, response))
