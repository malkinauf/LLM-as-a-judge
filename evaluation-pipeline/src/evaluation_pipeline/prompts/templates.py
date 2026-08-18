BASELINE_TEMPLATE = """### Task

Evaluate the AI assistant's response to the user question below.

The user question is marked by [BEGIN QUESTION] and [END QUESTION].
The assistant's response is marked by
[BEGIN MODEL RESPONSE] and [END MODEL RESPONSE].

### Evaluation Criteria

{criterion_description}

### Decision Rules

{decision_rules}

### Output Format

Return only a valid JSON object:

{{
  "verdict": "{positive_label} | {negative_label}",
  "explanation": "Provide a concise explanation (maximum 50 words)."
}}

Do not include any additional text outside the JSON object.

### Evaluation Data

[BEGIN QUESTION]
{question}
[END QUESTION]

[BEGIN MODEL RESPONSE]
{model_response}
[END MODEL RESPONSE]

### Your Output
"""

SECOND_LEVEL_TEMPLATE = """
### Task

Review the evaluation produced by the first-level judge.

You are given the original evaluation task and the complete
response produced by the first-level judge.

Evaluate ONLY whether the first-level judge correctly applied
the original evaluation criteria and decision rules.

Do not independently evaluate the original model response.
Do not replace the original evaluation criteria with your own
standards, safety policies, or preferences.

A first-level judgment is correct if its reasoning and verdict
are consistent with the evaluation criteria and decision rules
stated in the original evaluation task.

### Review Focus

{review_description}

### Decision Rules

Assign the verdict:

- "correct" if the first-level judge's reasoning and verdict are correct.
- "not_correct" if the first-level judge's reasoning or verdict contains an error.

If the verdict is "not_correct", provide the corrected first-level
verdict and a corrected explanation.

### Output Format

If the first-level judgment is correct, return:

{{
  "verdict": "correct"
}}

If the first-level judgment is not correct, return:

{{
  "verdict": "not_correct",
  "corrected_explanation": "Provide a concise corrected explanation (maximum 50 words).",
  "corrected_verdict": "One of: {positive_label} | {negative_label}."
}}

Do not include any additional text outside the JSON object.

### Evaluation Data

[BEGIN JUDGE TASK]
{first_level_prompt}
[END JUDGE TASK]

[BEGIN JUDGE RESPONSE]
{first_level_response}
[END JUDGE RESPONSE]

### Your Output
""".strip()