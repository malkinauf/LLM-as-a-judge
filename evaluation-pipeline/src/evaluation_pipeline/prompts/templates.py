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