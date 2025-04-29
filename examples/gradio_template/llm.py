import html  # Import the html module for escaping
import json
import uuid

import gradio as gr
import requests
from loguru import logger


# --- chat function remains the same as the previous version that yields both ---
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

    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "max_tokens": max_tokens,
        "seed": None,
    }

    response = requests.post(
        "https://api.deepseek.com/chat/completions",  # Use your actual API endpoint
        headers={
            "Accept": "application/json",
            "Content-type": "application/json",
            "Authorization": "Bearer sk-xxxxxx",  # Replace with your actual key
        },
        data=json.dumps(data),
        stream=True,
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
        # Yield error in reasoning part too
        yield full_response, f"**Streaming Error:**\n```\n{e}\n```"
    finally:
        # Ensure final state is yielded if loop finishes normally
        yield full_response, reasoning_response


# --- End of chat function ---


if __name__ == "__main__":
    with gr.Blocks() as demo:
        session_id = gr.State(lambda: str(uuid.uuid4()))
        # History now stores tuples: (user_message, assistant_display_message)
        chat_history = gr.State([])

        # Add state to track whether we're in the middle of generating a
        # response
        is_generating = gr.State(False)

        gr.Markdown("# QwQ-32B 对话 (带可折叠推理过程)")
        gr.Markdown("QwQ 是 Qwen 系列中的推理模型... (模型描述)")

        # --- UI Setup ---
        # No separate reasoning display needed
        chatbot = gr.Chatbot(
            label="聊天记录",
            height=600,
            show_copy_button=True,
            sanitize_html=False,
            bubble_full_width=False,
        )  # sanitize_html=False is important!
        with gr.Row():
            msg = gr.Textbox(placeholder="请输入您的问题...", container=False, scale=9)
            send_btn = gr.Button("发送", variant="primary", scale=1)

        examples = gr.Examples(
            examples=[
                "strawberry 里面有几个r？",
                "简单解释一下相对论。",
                "写一个计算阶乘的 Python 函数。",
            ],
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
                minimum=100, maximum=8192, value=2048, step=100, label="Max Tokens"
            )
        # --- End of UI Setup ---

        # --- User Function ---

        def user(user_message, history, session_id):
            # Append user message, placeholder for assistant message
            # Clear input textbox
            return (
                "",
                history + [[user_message, None]],
                session_id,
                history + [[user_message, None]],
            )

        # --- End of User Function ---

        # --- Bot Function ---

        def bot(history, temperature_value, top_p_value, max_tokens_value, session_id):
            if not history or not history[-1][0]:
                yield history, session_id, history  # No change if no user message
                return

            user_message = history[-1][0]
            # Ensure the placeholder for the assistant message exists
            if len(history[-1]) < 2:
                history[-1].append("")
            else:
                history[-1][1] = "*思考中...*"  # Initial placeholder

            # Stream updates for the chatbot history
            full_final_response = ""
            full_reasoning = ""

            # Start with the details element open to show reasoning live
            details_open = True

            # Use the chat generator which yields (response_chunk,
            # reasoning_chunk)
            for response_chunk, reasoning_chunk in chat(
                user_message,
                # Pass only user messages and *actual* assistant responses for API context
                # This requires careful history management if storing formatted HTML.
                # Simplification: Pass history as is, assuming API handles it or format is simple.
                [
                    h[:2] for h in history[:-1]
                ],  # Send previous turns' raw messages if stored differently
                temperature=temperature_value,
                top_p=top_p_value,
                max_tokens=max_tokens_value,
            ):
                full_final_response = response_chunk
                full_reasoning = reasoning_chunk

                # --- Format the combined message with collapsible HTML ---
                combined_message = full_final_response

                if reasoning_chunk:  # Even if it's just starting to generate
                    # Escape HTML characters within the reasoning content to prevent injection issues
                    # Use <pre> for better formatting of potentially code-like
                    # reasoning
                    escaped_reasoning = html.escape(full_reasoning)
                    formatted_reasoning = f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{escaped_reasoning}</pre>"

                    # Create details element with open attribute to show reasoning as it's being generated
                    # Once response is complete, next interaction will collapse
                    # it again
                    collapsible_reasoning_html = f"""
<details{' open' if details_open else ''} style="margin-top: 10px; border: 1px solid #eee; border-radius: 5px; padding: 5px;">
  <summary style="cursor: pointer; font-weight: bold; color: #555;">显示/隐藏 推理过程</summary>
  {formatted_reasoning}
</details>
"""
                    # Place reasoning above the response
                    combined_message = (
                        f"{collapsible_reasoning_html}\n\n{full_final_response}"
                    )
                # --- End of Formatting ---

                # Update the assistant's message in the *last* turn of the
                # history state
                history[-1][1] = combined_message

                # Yield updates: history state, session_id state, chatbot
                # display
                yield history, session_id, history

            # When generation is complete, update to close the details by
            # default for future displays
            if full_reasoning and full_reasoning.strip():
                # Create the final version with details closed by default
                escaped_reasoning = html.escape(full_reasoning)
                formatted_reasoning = f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{escaped_reasoning}</pre>"

                collapsible_reasoning_html = f"""
<details style="margin-top: 10px; border: 1px solid #eee; border-radius: 5px; padding: 5px;">
  <summary style="cursor: pointer; font-weight: bold; color: #555;">显示/隐藏 推理过程</summary>
  {formatted_reasoning}
</details>
"""
                # Place reasoning above the response with a separator
                combined_message = (
                    f"{collapsible_reasoning_html}\n{full_final_response}\n"
                )
                history[-1][1] = combined_message
                yield history, session_id, history

            # Log final results if needed
            logger.info(f"Session {session_id} - Final Response: {full_final_response}")
            logger.info(
                f"Session {session_id} - Final Reasoning (raw): {full_reasoning}"
            )

        # --- End of Bot Function ---

        # --- Event Handling ---
        # Define outputs - only history state, session state, and chatbot
        # component
        user_outputs = [msg, chat_history, session_id, chatbot]
        bot_outputs = [chat_history, session_id, chatbot]  # Removed reasoning_display

        # Connect text box submission
        msg.submit(
            user,
            [msg, chat_history, session_id],
            user_outputs,  # Outputs for user function
            queue=False,
        ).then(
            bot,
            # Inputs for bot function
            [chat_history, temperature, top_p, max_tokens, session_id],
            bot_outputs,  # Outputs for bot function
        )

        # Connect send button click
        send_btn.click(
            user,
            [msg, chat_history, session_id],
            user_outputs,  # Outputs for user function
            queue=False,
        ).then(
            bot,
            # Inputs for bot function
            [chat_history, temperature, top_p, max_tokens, session_id],
            bot_outputs,  # Outputs for bot function
        )
        # --- End of Event Handling ---

    demo.launch(
        debug=True,
        # Share=True # Uncomment if you need to share
    )
