> [!NOTE]
> **AI Assistance Disclosure**  
> Figures were generated with **ChatGPT (GPT-5.6 Sol, OpenAI)** based on the authors' concepts and technical specifications. All content was reviewed and validated by the authors.

# 1. Prompting Architecture — Overview

![Prompting Architecture](images/prompt_architecture.png)

# 2. Criterion Configuration - TruthfulQA Example

![Criterion_Configuration](images/prompt_criterion_configuration.png)

# 3. Prompt Construction -Three Methods

![Prompt_Construction](images/prompt_construction.png)

### Baseline

The baseline prompt combines the shared baseline body with the output template.

[→ View full Baseline prompt example](prompt_examples.md#baseline-prompt-example)

### Second-Level

The second-level prompt receives the complete first-level task and the
first-level judge response.

[→ View full Second-Level prompt example](prompt_examples.md#second-level-prompt-example)

### Dynamic

Dynamic prompting consists of a preliminary prediction step followed by
the final evaluation prompt.

[→ View final Dynamic prompt example](prompt_examples.md#dynamic-prompt-example)