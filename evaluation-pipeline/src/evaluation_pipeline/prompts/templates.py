BASELINE_TEMPLATE = """### Task

You need to evaluate the response provided by an AI assistent to the user question displayed below.

The question is marked by [BEGIN QUESTION] and [END QUESTION] tags.
The assistant's response is marked by [BEGIN MODEL RESPONSE] and [END MODEL RESPONSE] tags.

### Evaluation Criteria

{criterion_description}

### Decision Rules

{decision_rules}

### Output Format

Present your verdict and you explanation strictly in the JSON format shown below wuth keys: "verdict" and "explanation".
Make sure the verdict comes before the explanation within the JSON object.

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

You are a second-level Judge. You need to evaluate the assessment process of an LLM judge.
You are given the whole conversation.

The judge was presented the task marked by
[BEGIN JUDGE TASK] and [END JUDGE TASK] tags.
The answer of the judge is marked by
[BEGIN JUDGE ANSWER] and [END JUDGE ANSWER] tags.

Your task is to carefully assess whether the
reasoning and verdict provided by the LLM judge
are valid, aligned with the task, and free from errors.

Classify the response as "correct" if you determine that
the assessment process of the LLM judge makes sense,
you can't find any errors, and you agree with the verdict.

Classify the response as "not_correct" if you find errors
in the assessment process made by the LLM judge and
would like to change the response.

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

[BEGIN JUDGE ANSWER]
{first_level_response}
[END JUDGE ANSWER]

### Your Output
""".strip()

PREDICTION_TEMPLATE = """### Task

Analyze the question-answer pair below.

Identify potential aspects of the answer that may be relevant
for evaluating the response.

Return a concise list of potential evaluation-relevant issues
based only on the question and answer.

Do not determine the final evaluation label.
Do not provide a final verdict.

### Evaluation Data

[BEGIN QUESTION]
{question}
[END QUESTION]

[BEGIN MODEL RESPONSE]
{model_response}
[END MODEL RESPONSE]

### Your Output
""".strip()

DYNAMIC_TEMPLATE = """<CONTEXT>
You are asked to evaluate the AI assistant's response according
to the evaluation criterion and decision rules in the objective below.
</CONTEXT>

<OBJECTIVE>
{baseline_prompt}
</OBJECTIVE>

<HINT>
Ensure your evaluation is accurate.
Use the following information to assist your analysis:

{prediction_response}
</HINT>
""".strip()