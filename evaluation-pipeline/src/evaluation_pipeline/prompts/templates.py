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

SECOND_LEVEL_TEMPLATE = """### Task

{instruction}

### Review Focus

{review_focus}

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
  "corrected_verdict": "One of: {first_level_positive_label} | {first_level_negative_label}."}}

Do not include any additional text outside the JSON object.
### Evaluation Data

[BEGIN JUDGE TASK]
{first_level_prompt}
[END JUDGE TASK]

[BEGIN JUDGE RESPONSE]
{first_level_response}
[END JUDGE RESPONSE]

### Your Output
"""