"""
Message builder utilities for vLLM inference and evaluation.

Provides functions to construct chat message lists for Gemma (generation)
and Qwen (evaluation) pipelines.
"""


def build_gemma_messages(
    system_prompt: str,
    user_instruction: str,
    model_image_path: str,
    cloth_image_path: str,
) -> list[dict]:
    """Build the chat messages list for Gemma vLLM generation.

    Args:
        system_prompt: The system-level instruction text.
        user_instruction: The user-level instruction text appended after images.
        model_image_path: Absolute path to the model image on disk.
        cloth_image_path: Absolute path to the clothing image on disk.

    Returns:
        A list of message dicts compatible with the vLLM OpenAI-compatible
        chat completions API.
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Image 1: Model Image - Shows the model wearing current clothing",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{model_image_path}"},
                },
                {
                    "type": "text",
                    "text": "Image 2: Clothing Image - Shows the garment to be tried on",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{cloth_image_path}"},
                },
                {"type": "text", "text": user_instruction},
            ],
        },
    ]


def build_qwen_eval_messages(
    eval_instruction: str,
    model_image_path: str,
    cloth_image_path: str,
    result_image_path: str,
) -> list[dict]:
    """Build the three-image evaluation messages list for Qwen.

    Args:
        eval_instruction: The evaluation prompt text appended after images.
        model_image_path: Absolute path to the model image on disk.
        cloth_image_path: Absolute path to the clothing image on disk.
        result_image_path: Absolute path to the try-on result image on disk.

    Returns:
        A list of message dicts compatible with the vLLM OpenAI-compatible
        chat completions API.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{model_image_path}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{cloth_image_path}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{result_image_path}"},
                },
                {"type": "text", "text": eval_instruction},
            ],
        }
    ]
