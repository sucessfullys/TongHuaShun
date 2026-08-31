"""
调用本地 vLLM 服务 Qwen3.5-122B-A10B 的客户端程序
循环调用保证本地服务一直处于运行状态
"""

from openai import OpenAI
import time


def call_gemma4(prompt, messages=None, temperature=0.1, top_p=0.5):
    """
    调用本地 vLLM gemma4-31B 服务

    Args:
        prompt: 用户输入的提示词
        messages: 可选，消息列表格式 [{role: str, content: str}]
        temperature: 温度参数
        top_p: top_p 参数

    Returns:
        assistant_message: 模型返回的内容
    """
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8431/v1"
    model = "/mnt/image-edit/datasets/dingbaojin/models/google/gemma-4-31B-it"
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
    )

    return response.choices[0].message.content


def call_qwen(prompt, messages=None, temperature=0.1, top_p=0.5, presence_penalty=1.5):
    """
    调用本地 vLLM Qwen3.5-122B-A10B 服务

    Args:
        prompt: 用户输入的提示词
        messages: 可选，消息列表格式 [{role: str, content: str}]
        temperature: 温度参数
        top_p: top_p 参数
        presence_penalty: presence_penalty 参数

    Returns:
        assistant_message: 模型返回的内容
    """
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    model = "/mnt/model/Qwen3.5-122B-A10B"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    return response.choices[0].message.content


def main():
    # 固定的测试输入
    test_prompts = [
        "请介绍一下你自己",
        "1+1等于几?",
        "用一句话描述春天的景色",
    ]

    # 可用的模型服务
    # models = [
    #     {"name": "Qwen3.5-122B-A10B", "func": call_qwen},
    #     {"name": "Gemma4-31B", "func": call_gemma4},
    # ]
    models = [
        {"name": "Gemma4-31B", "func": call_gemma4},
    ]

    print("=" * 50)
    print("vLLM 客户端已启动")
    print(f"模型列表: {[m['name'] for m in models]}")
    print("=" * 50)

    while True:
        for model_info in models:
            model_name = model_info["name"]
            call_func = model_info["func"]

            for i, prompt in enumerate(test_prompts):
                print(f"\n[{model_name}] [{i+1}/{len(test_prompts)}] 调用中...")
                print(f"输入: {prompt}")

                try:
                    assistant_message = call_func(prompt)
                    print(f"输出: {assistant_message}")

                except Exception as e:
                    print(f"错误: {e}")
                    time.sleep(2)

                # time.sleep(1)

        print("\n一轮调用完成，开始下一轮...")
        time.sleep(1)


if __name__ == "__main__":
    main()
