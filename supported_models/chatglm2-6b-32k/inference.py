from transformers import AutoTokenizer, AutoModel
from amp.models.chatglm import patch_chatglm

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True).half().to("npu")
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)