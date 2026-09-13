import json
import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)


def _generate_response(prompt: str, max_new_tokens: int = 300) -> str:
    """
    Send a prompt to the Hugging Face model and return the generated text.
    """

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()


def get_raw_model_response(prompt: str, model: str) -> str:
    """
    Send a prompt to the model and return the raw text response.
    """

    return _generate_response(
        prompt=prompt,
        max_new_tokens=300,
    )


def get_json_model_response(prompt: str, model: str) -> str:
    """
    Send a judge prompt to the model and return the generated response.
    """

    return _generate_response(
        prompt=prompt,
        max_new_tokens=300,
    )


def normalize_judge_output(
    parsed_output: dict[str, Any],
    raw_output: str,
) -> dict[str, Any]:
    """
    Convert parsed model JSON into the normalized judge result format.
    """

    return {
        "predicted_label": parsed_output.get("verdict"),
        "explanation": parsed_output.get("explanation"),
        "corrected_verdict": parsed_output.get("corrected_verdict"),
        "corrected_explanation": parsed_output.get(
            "corrected_explanation"
        ),
        "raw_output": raw_output,
    }


def judge_response(prompt: str, model: str) -> dict[str, Any]:
    """
    Send a judge prompt to the model and parse its JSON response.
    """

    raw_output = get_json_model_response(
        prompt=prompt,
        model=model,
    )

    try:
        parsed_output = json.loads(raw_output)

        return normalize_judge_output(
            parsed_output=parsed_output,
            raw_output=raw_output,
        )

    except json.JSONDecodeError as e:
        logger.warning(
            f"Failed to parse model output as JSON: {e}"
        )

        logger.warning(
            f"RAW OUTPUT REPR: {repr(raw_output)}"
        )

        logger.warning(
            f"RAW OUTPUT END: {repr(raw_output[-300:])}"
        )

        return {
            "predicted_label": "parsing_error",
            "explanation": "Could not parse model output as JSON.",
            "corrected_verdict": None,
            "corrected_explanation": None,
            "raw_output": raw_output,
        }