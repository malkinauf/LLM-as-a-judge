# Prompt System

The prompt system provides reusable prompts for different LLM-as-a-Judge
evaluation methods.

Currently, three evaluation methods are supported:

-   **Baseline** -- directly evaluates the model response according to a
    selected criterion.
-   **Second-level** -- reviews the result produced by the first-level
    judge.
-   **Dynamic** -- first generates evaluation-relevant information and
    then uses it as an additional hint for the final evaluation.

## Structure

``` text
src/evaluation_pipeline/prompts/
├── criteria/
│   ├── correctness.yaml
│   ├── harmbench.yaml
│   ├── safety.yaml
│   └── truthfulness.yaml
├── builder.py
├── loader.py
├── registry.py
└── templates.py
```

The files have the following responsibilities:

-   `criteria/` -- definitions of evaluation criteria
-   `templates.py` -- reusable prompt templates
-   `builder.py` -- builds the final prompts
-   `loader.py` -- loads criterion configurations
-   `registry.py` -- registers available criteria

## Evaluation Criteria

Each evaluation criterion is defined in a YAML file.

Example:

``` yaml
name: truthfulness

descriptions:
  minimal: |
    Short definition of the criterion.

  detailed: |
    More detailed definition of the criterion.

decision_rules: |
  Rules used by the judge to assign a label.

labels:
  positive: truthful
  negative: not_truthful
```

Currently supported criteria are:

-   `truthfulness`
-   `safety`
-   `correctness`
-   `harmbench`

The criterion and its detail level are selected in the experiment
configuration:

``` python
BASELINE_CRITERION = "truthfulness"
BASELINE_DETAIL = "minimal"
```

## Supported Prompt Methods

### Baseline

The baseline method evaluates the model response directly:

``` text
Question + Model Response
          ↓
Evaluation Criterion
          ↓
LLM Judge
          ↓
Verdict + Explanation
```

The criterion description, decision rules, and labels are taken from the
corresponding YAML file.

### Second-level

The second-level method checks the result of the first judge:

``` text
First-level Prompt
        +
First-level Response
        ↓
Second-level Judge
        ↓
correct / not_correct
```

If the first judgment is incorrect, the second-level judge can provide a
corrected verdict.

### Dynamic

The dynamic method consists of two steps:

``` text
Question + Model Response
          ↓
Prediction
          ↓
Evaluation-relevant information
          ↓
Baseline Prompt + Prediction Hint
          ↓
Final Judge
```

The prediction step does **not** make the final classification.

It only provides additional information that may help the final judge.

The same prediction approach is used across datasets so that the dynamic
method can be compared consistently.

## Adding a New Criterion

To add a new criterion, for example `helpfulness`:

### 1. Create a YAML file

``` text
criteria/helpfulness.yaml
```

Example:

``` yaml
name: helpfulness

descriptions:
  minimal: |
    Definition of helpfulness.

  detailed: |
    Detailed definition of helpfulness.

decision_rules: |
  - Use "helpful" if the response satisfies the criterion.
  - Use "not_helpful" otherwise.

labels:
  positive: helpful
  negative: not_helpful
```

### 2. Register the criterion

Add it to `registry.py`:

``` python
CRITERIA = {
    "truthfulness": load_criterion("truthfulness"),
    "safety": load_criterion("safety"),
    "correctness": load_criterion("correctness"),
    "harmbench": load_criterion("harmbench"),
    "helpfulness": load_criterion("helpfulness"),
}
```

### 3. Select it in the experiment

``` python
BASELINE_CRITERION = "helpfulness"
BASELINE_DETAIL = "minimal"
```

No new prompt template is required.

## When Should a New Prompt Template Be Added?

A new criterion normally **does not require a new prompt template**.

``` text
New criterion
→ add a YAML file

New criterion description
→ modify the YAML file

New prompt structure
→ modify templates.py

New evaluation method
→ add a new template and builder logic
```

## Dataset vs. Criterion

Datasets and evaluation criteria are kept separate.

A dataset provides:

``` text
question
model_response
true_label
```

The criterion defines **how the response should be evaluated**.

Therefore, a new dataset does not automatically require a new prompt.

Keeping the same prompt structure across datasets makes the experimental
comparison more consistent and avoids introducing dataset-specific
prompt variations.
