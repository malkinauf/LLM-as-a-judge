# LLM-as-a-Judge Evaluation Pipeline

A lightweight evaluation pipeline for running LLM-as-a-judge experiments with local language models through Ollama.

This project is located inside the larger `LLM` repository.

---

## Project structure

```text
LLM/
└── evaluation-pipeline/
    ├── datasets/
    ├── notebooks/
    │   └── run_experiments.ipynb
    |   └── prepare_dataset.ipynb
    ├── prompts/
    ├── results/
    ├── src/
    │   └── evaluation_pipeline/
    │       ├── __init__.py
    │       ├── dataset.py
    │       ├── judge.py
    │       ├── prompts.py
    │       └── runner.py
    ├── pyproject.toml
    └── README.md
```

---

## Requirements

Before setting up the project, make sure the following tools are installed:

- Python 3.10 or newer
- Git
- Ollama

---

## 1. Install Ollama

Download and install Ollama from:

```text
https://ollama.com/download
```

After installation, verify that Ollama is available:

```bash
ollama --version
```

If Ollama is not running automatically, start it manually:

```bash
ollama serve
```

---

## 2. Download local models

Download at least one model before running the experiments.

Examples:

```bash
ollama pull llama3
ollama pull llama3.1
ollama pull qwen2.5:14b
ollama pull mistral
```

Check which models are installed:

```bash
ollama list
```

The model name used in the notebook must exactly match the name shown by `ollama list`.

---

## 3. Clone the repository

Clone the main `LLM` repository:

```bash
git clone <repository-url>
```

Navigate to the evaluation pipeline project:

```bash
cd LLM/evaluation-pipeline
```

All following commands should be executed from the `evaluation-pipeline` directory.

---

## 4. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 5. Install the project

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the package in editable mode:

```bash
pip install -e .
```

Editable mode allows changes in `src/evaluation_pipeline/` to be used immediately without reinstalling the package.

---

## 6. Set up the Jupyter kernel

Install the kernel for this virtual environment:

```bash
python -m ipykernel install --user --name evaluation-pipeline --display-name "Python (evaluation-pipeline)"
```

This makes the environment available in Jupyter and VS Code notebooks.

To check available kernels:

```bash
jupyter kernelspec list
```

You should see a kernel named:

```text
evaluation-pipeline
```

---

## 7. Verify the installation

Check that the package can be imported:

```bash
python -c "import evaluation_pipeline; print('Package import OK')"
```

Check that Ollama is accessible from Python:

```bash
python -c "import ollama; print(ollama.list())"
```

---

## 8. Run the notebook

Start Jupyter from the project directory:

```bash
jupyter notebook
```

Open:

```text
notebooks/run_experiments.ipynb
```

Select the kernel:

```text
Python (evaluation-pipeline)
```

---

### Experiment configuration

Inside `notebooks/run_experiments.ipynb`, configure the experiment parameters before running the notebook.

### Select the model

Examples:

```python
MODEL = "llama3"
```

```python
MODEL = "llama3.1"
```

```python
MODEL = "qwen2.5:14b"
```

```python
MODEL = "mistral"
```

The model name must exactly match the output of:

```bash
ollama list
```

---

### Select the prompt type

Available prompt configurations:

```python
PROMPT_TYPE = "baseline"
```

```python
PROMPT_TYPE = "second_level"
```

```python
PROMPT_TYPE = "dynamic"
```

The notebook validates the selected prompt type automatically.

---

### Configure prompt files

Example:

```python
PROMPT_FILE_BASELINE = "../prompts/baseline_harmless_v1.txt"
PROMPT_FILE_SECOND_LEVEL = "../prompts/second_level_truthfulness_v1.txt"
PROMPT_FILE_DYNAMIC = "../prompts/dynamic_truthfulness_v1.txt"
PROMPT_FILE_HINT = "../prompts/hint_truthfulness_v1.txt"
```

Ensure that all referenced prompt files exist inside:

```text
prompts/
```

---

### Select the dataset

Example:

```python
DATASET_FILE = "../datasets/prepared/truthfulqa_binary_5.json"
```

Ensure that the dataset file exists inside:

```text
datasets/prepared/
```

---

### Configure the evaluation task

Examples:

```python
TASK_TYPE = "truthfulness"
```
or 

```python
TASK_TYPE = "safety"
```

---

## 10. Run experiments

Run the notebook cells sequentially.

Experiment outputs are saved in:

```text
results/
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'evaluation_pipeline'`

Make sure you installed the package from the `evaluation-pipeline` directory:

```bash
pip install -e .
```

Also check that the package structure contains:

```text
src/evaluation_pipeline/__init__.py
```

### Ollama is not found

Make sure Ollama is installed and available in your terminal:

```bash
ollama --version
```

### Model not found

Download the model first:

```bash
ollama pull llama3
```

Or use a model name listed by:

```bash
ollama list
```

### Notebook uses the wrong environment

Reinstall the Jupyter kernel:

```bash
python -m ipykernel install --user --name evaluation-pipeline --display-name "Python (evaluation-pipeline)"
```

Then restart Jupyter and select:

```text
Python (evaluation-pipeline)
```
