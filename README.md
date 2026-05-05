# LLM-as-a-Judge Evaluation Pipeline

This project provides a simple pipeline to run LLM-as-a-judge experiments using custom prompts and datasets.

## Project Structure

```
evaluation-pipeline/
├── datasets/ # Raw and prepared datasets
├── notebooks/ # Jupyter notebooks for experiments
├── prompts/ # Prompt templates
├── results/ # Experiment outputs (ignored by git)
├── src/ # Core logic
│ ├── ___init___.py
│ ├── dataset.py
│ ├── judge.py
│ ├── prompts.py
│ └── runner.py
├── pyproject.toml # Project dependencies
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd evaluation-pipeline
```
### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
```
### 3. Install dependencies
```bash
pip install -e .
```
### 4. (Optional but recommended) Add Jupyter kernel
```bash
python -m ipykernel install --user --name={name}
```

###  How it works
- You configure experiments in the notebook
- Core logic lives in src/
- Prompts are loaded from prompts/
- Results are saved to results/

### Troubleshooting
continue
