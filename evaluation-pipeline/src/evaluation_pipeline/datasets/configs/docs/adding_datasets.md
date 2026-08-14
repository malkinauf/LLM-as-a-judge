# Adding a Dataset

Datasets are integrated through YAML configuration files and converted to a common evaluation format:

```text
id
dataset
question
model_response
y_true
```

Dataset-specific logic should be described in the configuration whenever possible.

## 1. Create a Configuration

Create a YAML file in:

```text
src/evaluation_pipeline/datasets/configs/
```

Example:

```yaml
name: my_dataset
output_dataset_name: my_dataset

source:
  type: huggingface
  dataset: organization/dataset-name
  split: train

samples:
  - question:
      source: prompt

    model_response:
      source: response

    y_true:
      source: label
```

`question`, `model_response`, and `y_true` are mapped from fields of the original dataset.

`id` and `dataset` are added automatically.

## 2. Optional Configuration

### Label Mapping

Use `values` when source labels need to be converted:

```yaml
y_true:
  source: is_safe
  values:
    true: safe
    false: not_safe
```

### Multiple Samples per Row

One source row can produce multiple evaluation samples:

```yaml
samples:
  - question:
      source: question
    model_response:
      source: best_answer
    y_true:
      value: truthful

  - question:
      source: question
    model_response:
      source: incorrect_answers
      take: first
    y_true:
      value: not_truthful
```

### Balanced Sampling

For an equal number of samples from two labels:

```yaml
sampling:
  strategy: balanced
```

For example, `n_samples=100` produces 50 randomly selected samples from each label.

### Annotator Agreement

If multiple annotations must agree:

```yaml
agreement:
  fields:
    - human_0
    - human_1
    - human_2
```

Only rows with identical values in all specified fields are used.

## 3. Supported Sources

### Hugging Face

```yaml
source:
  type: huggingface
  dataset: organization/dataset-name
  subset: optional-subset
  split: train
```

### JSON

```yaml
source:
  type: json
  path: data/my_dataset.json
  url: https://example.com/my_dataset.json
```

If the local JSON file does not exist, it is downloaded from `url`.

## 4. Build the Dataset

No dataset-specific code is required in the notebook:

```python
dataset = build_dataset(
    name="my_dataset",
    n_samples=100,
    seed=42,
)
```

The resulting dataset is mapped, sampled, and validated automatically.

## Adding New Behavior

Prefer extending the YAML configuration over adding dataset-specific Python code.

Avoid:

```python
if name == "my_dataset":
    ...
```

Extend the generic pipeline only when the required behavior cannot be expressed by the existing configuration.