import json
import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_tokenizer = None
_model = None


def load_hf_judge_model(model_name: str = DEFAULT_MODEL_NAME) -> None:
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )


def judge_response(prompt: str, model: str | None = None) -> dict[str, Any]:
    model_name = model or DEFAULT_MODEL_NAME
    load_hf_judge_model(model_name)

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(
        text,
        return_tensors="pt",
    ).to(_model.device)

    outputs = _model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]

    raw_output = _tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    try:
        parsed_output = json.loads(raw_output)

        return {
            "predicted_label": parsed_output.get("verdict"),
            "explanation": parsed_output.get("explanation"),
            "corrected_verdict": parsed_output.get("corrected_verdict"),
            "corrected_explanation": parsed_output.get("corrected_explanation"),
            "raw_output": raw_output,
        }

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse model output as JSON: {e}")
        logger.warning(f"Raw output: {raw_output}")

        return {
            "predicted_label": "parsing_error",
            "explanation": "Could not parse model output as JSON.",
            "raw_output": raw_output,
        }