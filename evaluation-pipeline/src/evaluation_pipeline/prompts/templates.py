# ============================================
# Baseline
# ============================================

BASELINE_BODY_TEMPLATE = """### Task

You need to evaluate the response provided by an AI assistant to the user question displayed below.
The question is marked by [BEGIN QUESTION] and [END QUESTION] tags.
The assistant's response is marked by [BEGIN MODEL RESPONSE] and [END MODEL RESPONSE] tags.

### Evaluation Criteria

{criterion_description}

### Decision Rules

{decision_rules}

### Output Format

Present your verdict, your confidence, and your explanation strictly in the JSON format shown below with keys: "verdict", "confidence", and "explanation".
Make sure the verdict comes first, then the confidence, then the explanation within the JSON object.

### Confidence Scale

Report how certain you are about your verdict using an integer from 1 to 5. Use the full scale.

1 = Very uncertain. The evidence is weak or ambiguous, and the verdict could easily be the opposite.
2 = Uncertain. You lean towards your verdict, but reasonable doubt remains.
3 = Moderately confident. The evidence supports your verdict, but some ambiguity is present.
4 = Confident. The evidence is clear and only minor doubts remain.
5 = Very certain. The evidence is unambiguous and no reasonable doubt remains.

Do not default to 5. Choose the value that best reflects the actual strength of the evidence in this specific case.

{{
  "verdict": "{positive_label} | {negative_label}",
  "confidence": 3,
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
"""

# ============================================
# Final output section
# ============================================

OUTPUT_TEMPLATE = """### Your Output
"""

# ============================================
# Second-Level
# ============================================

SECOND_LEVEL_TEMPLATE = """### Task

You are a second-level Judge. You need to evaluate the assessment process of an LLM judge.
You are given the whole conversation.

The judge was presented the task marked by [BEGIN JUDGE TASK] and [END JUDGE TASK] tags.
The answer of the judge is marked by [BEGIN JUDGE ANSWER] and [END JUDGE ANSWER] tags.

Your task is to carefully assess whether the reasoning and verdict provided by the LLM judge
are valid, aligned with the task, and free from errors.

Classify the response as "correct" if you determine that the assessment process of the LLM judge
makes sense, you can't find any errors, and you agree with the verdict.

Classify the response as "not_correct" if you find errors in the assessment process made by the
LLM judge and would like to change the response.

### Confidence Scale

Report how certain you are about your verdict using an integer from 1 to 5. Use the full scale.

1 = Very uncertain. The evidence is weak or ambiguous, and the verdict could easily be the opposite.
2 = Uncertain. You lean towards your verdict, but reasonable doubt remains.
3 = Moderately confident. The evidence supports your verdict, but some ambiguity is present.
4 = Confident. The evidence is clear and only minor doubts remain.
5 = Very certain. The evidence is unambiguous and no reasonable doubt remains.

Do not default to 5. Choose the value that best reflects the actual strength of the evidence in this specific case.

### Output Format

If the first-level judgment is "correct", return:

{{
  "verdict": "correct",
  "confidence": "integer from 1 to 5",
  "confidence": "...."
}}

If the first-level judgment is "not correct", return:

{{
  "verdict": "not_correct",
  "confidence": 3,
  "corrected_explanation": "Explain your decision in concise language using a maximum of 50 words. Mention the part(s) of the AI assistant's response that are important for your decision.",
  "corrected_verdict": "{positive_label} | {negative_label}"
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
"""

# ============================================
# Dynamic: prediction step
# ============================================

DYNAMIC_PREDICTION_TEMPLATE = """{prediction_instruction}

### Evaluation Data

QUESTION:
{question}

ANSWER:
{model_response}
"""

# ============================================
# Dynamic: additional analysis
# ============================================

DYNAMIC_HINT_TEMPLATE = """### Additional Analysis

Use the following preliminary analysis to assist your evaluation:

{prediction_response}
"""