import html  # Import the html module for escaping
import json
import uuid
from typing import Dict, List

import gradio as gr
import requests
from loguru import logger

# https://huggingface.co/spaces/ysharma/gradio_chatbot_thinking/tree/main

URL = "http://10.77.245.193:8000/v1/chat/completions"
API_KEY = None
MODEL = "Qwen/Qwen3-1.7B"

TITLE = "Qwen/Qwen3-1.7B"
DESCRIPTION = "Qwen/Qwen3-1.7B 是 Qwen 系列中的推理模型..."


def send_message(
    url: str,
    api_key: str,
    model: str,
    messages: List[Dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
    stream: bool = True,
    seed: int = None,
):
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        "max_tokens": max_tokens,
        "seed": seed,
    }

    response = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Content-type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else None,
        },
        data=json.dumps(data),
        stream=True,
    )
    return response


def chat(content, history, temperature=0.7, top_p=0.9, max_tokens=8192):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    if history:
        for i, (user_msg, assistant_msg) in enumerate(history):
            messages.insert(i * 2 + 1, {"role": "user", "content": user_msg})
            if assistant_msg:
                messages.insert(
                    i * 2 + 2, {"role": "assistant", "content": str(assistant_msg)}
                )
    response = send_message(
        URL,
        API_KEY,
        MODEL,
        messages,
        temperature,
        top_p,
        max_tokens,
    )

    if response.status_code != 200:
        error_message = f"Error: {response.status_code} - {response.text}"
        logger.error(error_message)
        yield f"**API Error:**\n```\n{error_message}\n```", ""
        return

    full_response = ""
    reasoning_response = ""
    try:
        for line in response.iter_lines():
            if line:
                try:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                        if line.strip() == "[DONE]":
                            break
                        chunk = json.loads(line)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content_update = False
                            reasoning_update = False

                            if "content" in delta and delta["content"] is not None:
                                full_response += delta["content"]
                                content_update = True

                            if (
                                "reasoning_content" in delta
                                and delta["reasoning_content"] is not None
                            ):
                                reasoning_response += delta["reasoning_content"]
                                reasoning_update = True

                            if content_update or reasoning_update:
                                yield full_response, reasoning_response  # Yield both current states

                except json.JSONDecodeError as e:
                    logger.error(f"JSON Decode Error: {e} on line: {line}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error during response streaming: {e}")
        yield full_response, f"**Streaming Error:**\n```\n{e}\n```"
    finally:
        yield full_response, reasoning_response


def _format_reasoning_html(reasoning_text: str, initially_open: bool = False) -> str:
    """Formats the reasoning text into an HTML collapsible section."""
    if not reasoning_text or not reasoning_text.strip():
        return ""
    escaped_reasoning = html.escape(reasoning_text)
    formatted_reasoning = f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{escaped_reasoning}</pre>"
    details_attr = " open" if initially_open else ""
    return f"""
<details{details_attr} style="margin-top: 10px; border: 1px solid #eee; border-radius: 5px; padding: 5px;">
  <summary style="cursor: pointer; font-weight: bold; color: #555;">显示/隐藏 推理过程</summary>
  {formatted_reasoning}
</details>
"""


if __name__ == "__main__":
    with gr.Blocks() as demo:
        session_id = gr.State(lambda: str(uuid.uuid4()))
        chat_history = gr.State([])

        is_generating = gr.State(False)

        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        chatbot = gr.Chatbot(
            label="聊天记录",
            height=600,
            show_copy_button=True,
            sanitize_html=False,
            bubble_full_width=False,
        )
        with gr.Row():
            msg = gr.Textbox(placeholder="请输入您的问题...", container=False, scale=9)
            send_btn = gr.Button("发送", variant="primary", scale=1)

        examples = gr.Examples(
            examples=["strawberry 里面有几个r？", "你是谁？"],
            inputs=msg,
        )

        gr.Markdown("## 生成参数")
        with gr.Row():
            temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p"
            )
            max_tokens = gr.Slider(
                minimum=256, maximum=32768, value=2048, step=64, label="Max Tokens"
            )

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history, temperature_value, top_p_value, max_tokens_value, session_id):
            if not history or not history[-1][0]:
                yield history, session_id, history  # No change if no user message
                return

            user_message = history[-1][0]
            if len(history[-1]) < 2:
                history[-1].append("")
            else:
                history[-1][1] = "*思考中...*"

            full_final_response = ""
            full_reasoning = ""

            for response_chunk, reasoning_chunk in chat(
                user_message,
                [
                    h[:2] for h in history[:-1]
                ],  # Send previous turns' raw messages if stored differently
                temperature=temperature_value,
                top_p=top_p_value,
                max_tokens=max_tokens_value,
            ):
                full_final_response = response_chunk
                full_reasoning = reasoning_chunk

                reasoning_html = _format_reasoning_html(
                    full_reasoning, initially_open=True
                )
                combined_message = (
                    f"{reasoning_html}\n\n{full_final_response}"
                    if reasoning_html
                    else full_final_response
                )
                history[-1][1] = combined_message

                yield history, session_id, history

            # Final update after stream ends - ensure reasoning is displayed (collapsed)
            if full_reasoning:
                reasoning_html = _format_reasoning_html(full_reasoning)
                combined_message = (
                    f"{reasoning_html}\n{full_final_response}\n"
                    if reasoning_html
                    else full_final_response
                )
                history[-1][1] = combined_message
                # Yield the final state one more time
                yield history, session_id, history

        user_outputs = [msg, chat_history]
        bot_outputs = [chat_history, session_id, chatbot]

        msg.submit(
            user,
            [msg, chat_history],
            user_outputs,
            queue=False,
        ).then(
            bot,
            [chat_history, temperature, top_p, max_tokens, session_id],
            bot_outputs,
        )

        send_btn.click(
            user,
            [msg, chat_history],
            user_outputs,
            queue=False,
        ).then(
            bot,
            [chat_history, temperature, top_p, max_tokens, session_id],
            bot_outputs,
        )

    demo.launch(
        debug=True,
    )
