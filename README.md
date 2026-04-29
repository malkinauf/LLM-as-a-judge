# LLM-as-a-Judge Repository Guidelines

## Important Rule

**Do NOT push directly to `main`.**

The `main` branch must always stay clean and stable.  
All work should be done in personal branches and merged through Pull Requests.

---

## Branch Naming Convention

Use the following format:

```bash
type/name-task
```

### Examples

```bash
feature/tatiana-proposal
feature/esmo-code-bugfix
bugfix/reference-errors
docs/readme-update
```

### Branch Types

- `feature` = new work
- `bugfix` = fix errors
- `docs` = documentation updates
- `hotfix` = urgent fix

---

## How to Create a Branch

Always pull the newest version first:

```bash
git pull
```

Then create a new branch:

```bash
git checkout -b feature/yourname-task
```

### Example

```bash
git checkout -b feature/anastasia-proposal
```

---

## Daily Workflow

Check your current branch:

```bash
git branch
```

Add changes:

```bash
git add .
```

Create a commit:

```bash
git commit -m "Describe your changes"
```

Push your branch:

```bash
git push -u origin feature/yourname-task
```

### Example

```bash
git push -u origin feature/anastasia-proposal
```

---

## Good Commit Message Examples

```bash
git commit -m "Add proposal setup"
git commit -m "Update methodology section"
git commit -m "Fix bibliography"
git commit -m "Add evaluation script"
git commit -m "Update README guidelines"
```

---

## Pull Request Workflow

After pushing your branch:

1. Open GitHub
2. Create a Pull Request into `main`
3. Review changes
4. Merge only when everything is correct

---

## Project Structure

```text
proposal/   LaTeX proposal files
code/       Source code and notebooks
papers/     Literature notes and references
docs/       Planning and meeting notes
results/    Outputs, plots, and tables
```

---

## Team Rules

- Do not push directly to `main`
- Always work in your own branch
- Pull latest changes before starting work
- Use clear commit messages
- Do not upload temporary LaTeX files
- Ask before deleting or moving important files
