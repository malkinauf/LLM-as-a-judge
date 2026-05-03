import json
from ollama import chat


def judge_response(prompt: str, model: str) -> dict:
    """
    Send a prepared prompt to the judge model and parse its JSON response.
    """

    response = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0
        }
    )

    raw_output = response["message"]["content"]

    try:
        parsed_output = json.loads(raw_output)

        return {
            "predicted_label": parsed_output.get("answer", "invalid_label"),
            "explanation": parsed_output.get("explanation", ""),
            "raw_output": raw_output
        }

    except json.JSONDecodeError:
        return {
            "predicted_label": "parsing_error",
            "explanation": "Could not parse model output as JSON.",
            "raw_output": raw_output
        }