# Specialized Judge Models – Literature Notes

> Summaries follow the standard template:
> Paper title · Key idea · Judge models · Dataset · Prompt strategy · Experiment setup · Metrics · Important findings · Open questions

---

## 1. Prometheus: Inducing Fine-grained Evaluation Capability in Language Models

**Citation:** Kim, S., Shin, S., Cho, Y., Jang, J., Longpre, S., Lee, H., … & Hajishirzi, H. (2023). *Prometheus: Inducing Fine-grained Evaluation Capability in Language Models*. arXiv:2310.08491.

### Key Idea
Train an open-source judge model (based on LLaMA-2) that can evaluate other LLM outputs against fine-grained, user-defined rubrics—instead of relying on proprietary GPT-4 for evaluation. The model learns to produce both a score and a verbal feedback (critique) given a scoring rubric as part of the prompt.

### Judge Models
- **Prometheus-13B** (LLaMA-2-13B fine-tuned)
- Baselines: GPT-4, ChatGPT, Claude-2, LLaMA-2-13B-Chat, Vicuna-13B

### Dataset
- **Feedback Collection** – 1,000 instructions each with 5 candidate responses, fine-grained score rubrics (score 1–5), and GPT-4-generated reference answers and feedback (~100K training instances in total).

### Prompt Strategy
The prompt consists of five parts: (1) instruction, (2) reference answer, (3) candidate response, (4) fine-grained scoring rubric (user-defined, 5 criteria), and (5) request for score + verbal feedback. This rubric-conditioned prompting allows flexible, task-specific evaluation.

### Experiment Setup
- Evaluated on two held-out benchmarks: **MT-Bench** and **HHH Alignment** (yes/no preference).
- Pearson/Spearman correlation with human judgments.
- 4 human annotators per sample for ground truth.
- Compared against GPT-4, ChatGPT, Claude-2 and open LLMs as judges.

### Metrics
- Pearson correlation (r) and Spearman correlation (ρ) between model scores and human scores.
- Win rate in pairwise comparisons.

### Important Findings
- Prometheus-13B achieves human-correlation on par with GPT-4 when scoring rubrics are provided (r ≈ 0.897 vs. GPT-4 ≈ 0.895 on Vicuna Bench).
- Without rubrics, performance drops significantly, highlighting the critical role of fine-grained rubric conditioning.
- An open-source judge can match proprietary models when properly fine-tuned on high-quality feedback data.

### Open Questions
- Can rubric-conditioned judges generalize to unseen domains without human-written rubrics?
- How sensitive is performance to rubric quality?

---

## 2. Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models

**Citation:** Kim, S., Suk, J., Longpre, S., Lin, B. Y., Shin, J., Welleck, S., … & Hajishirzi, H. (2024). *Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models*. arXiv:2405.01535.

### Key Idea
Extend Prometheus to support **both absolute scoring (direct assessment)** and **pairwise ranking** in a single model, achieving better alignment with human judgments and GPT-4 preferences. A key novelty is a weight-merging technique to combine models trained on each task.

### Judge Models
- **Prometheus 2 – 7B** and **Prometheus 2 – 8×7B** (Mistral / Mixtral base)
- Baselines: GPT-4, GPT-3.5-Turbo, Llama-2-70B-Chat, Claude-3, Prometheus-13B (v1), Auto-J

### Dataset
- **Preference Collection** – 1,000 instructions, each with 2 candidate responses and a rubric for pairwise comparison (GPT-4 labelled).
- Combined with the original Feedback Collection from Prometheus 1 for absolute scoring.

### Prompt Strategy
Two prompt templates:
- *Direct Assessment*: same rubric-conditioned format as Prometheus 1.
- *Pairwise Ranking*: prompt provides two responses (A and B) and the rubric, asking the model to pick the better one with reasoning.

Model merging: weights from the absolute-scoring model and the pairwise-ranking model are linearly interpolated (ties broken by validation performance).

### Experiment Setup
- Four benchmarks: **MT-Bench**, **HHH Alignment**, **Flask Eval**, **Vicuna Bench**.
- Human agreement measured on both absolute and pairwise settings.
- Additional out-of-distribution evaluation on summarization and translation.

### Metrics
- Pearson/Spearman correlation (absolute scoring).
- Agreement rate and Cohen's κ (pairwise ranking).
- Win rate against GPT-4 judgments.

### Important Findings
- Prometheus 2 (7B) outperforms all open judge models and matches GPT-4 on pairwise ranking (agreement ≈ 80%).
- Weight merging outperforms single-task models and multi-task fine-tuning for combining both evaluation modes.
- Larger Prometheus 2 (8×7B) achieves near-GPT-4-level on absolute scoring too.

### Open Questions
- Does weight merging generalize to judges trained on more than two evaluation tasks?
- Can the approach scale to multimodal evaluation (text + image)?

---

## 3. JudgeLM: Fine-tuned Large Language Models are Scalable Judges

**Citation:** Zhu, L., Wang, X., Wang, X. (2023). *JudgeLM: Fine-tuned Large Language Models are Scalable Judges*. arXiv:2310.17631.

### Key Idea
Fine-tune LLMs as scalable judges using a large teacher-student framework: GPT-4 annotations on 100K+ diverse question-answer pairs serve as training labels. Investigates and mitigates known judge biases (position, knowledge, format).

### Judge Models
- **JudgeLM-7B, JudgeLM-13B, JudgeLM-33B** (Vicuna / LLaMA-2 base).
- Baselines: GPT-4, ChatGPT, Alpaca, Vicuna.

### Dataset
- **JudgeLM-100K** – 100,000 question/answer/judgment triplets covering 5 domains: reasoning, math, coding, STEM, humanities. GPT-4 annotations used as gold labels.

### Prompt Strategy
- Pairwise comparison prompt: given a question and two answers (A and B), produce a preference judgment with an explanation.
- **Swap augmentation**: both orderings (A-B, B-A) are presented during training to reduce position bias.
- **Reference answer injection**: reference solution appended to prompt to reduce knowledge-boundary issues.

### Experiment Setup
- Evaluated on **JudgeLM-Val-685** (held-out human-labelled subset).
- Human-agreement rate for pairwise comparisons.
- Ablation studies on swap, reference answer, and model size.

### Metrics
- Agreement rate with human judgments (%).
- Win/tie/loss rate when judging pairs of LLMs.
- Kendall's τ rank correlation for system-level ranking.

### Important Findings
- JudgeLM-33B approaches GPT-4-level agreement with humans (~82% vs GPT-4 ~84%).
- Swap augmentation reduces position bias by ~6 percentage points.
- Reference answers significantly help for knowledge-intensive questions (math, coding).
- Larger judge models are consistently better; 7B models already outperform ChatGPT as judge.

### Open Questions
- How should judges be trained to handle questions that exceed their own knowledge limits?
- Can the bias-mitigation techniques scale to multi-turn conversations?

---

## 4. PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization

**Citation:** Wang, Y., Yu, Z., Zeng, Z., Yang, L., Wang, C., Chen, H., … & Zhang, W. (2024). *PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization*. arXiv:2306.05087.

### Key Idea
Provide an open, reproducible automated evaluation pipeline for comparing instruction-tuned LLMs. PandaLM is a 7B judge model trained to judge pairwise outputs and provide natural language rationales, addressing reproducibility concerns with GPT-4-based evaluation.

### Judge Models
- **PandaLM-7B** (LLaMA-7B fine-tuned).
- Baselines: GPT-3.5-Turbo, human annotators.

### Dataset
- **PandaLM-Eval test set** – 2,967 samples covering conciseness, clarity, adherence, knowledge, creativity, and logic dimensions; annotated by 3 human raters per sample.
- Training set: ~300K GPT-3.5-generated pairwise judgments with rationale.

### Prompt Strategy
Pairwise judgment prompt that includes the instruction, two responses (Response 1 and Response 2), and asks for a structured JSON output: preferred response, rationale, and 6 aspect scores. The structured output enables quantitative analysis.

### Experiment Setup
- Human evaluation on PandaLM-Eval test set (3-way annotator agreement).
- Cohen's κ between PandaLM and human judgments.
- Downstream use case: hyperparameter selection for LLM fine-tuning.

### Metrics
- Cohen's κ (inter-annotator agreement between model and humans).
- Preference accuracy (% matching human majority vote).
- Win-rate shift when used to guide fine-tuning decisions.

### Important Findings
- PandaLM-7B achieves κ ≈ 0.56 with humans, comparable to GPT-3.5-Turbo's κ ≈ 0.59.
- The open, reproducible nature makes PandaLM suitable for iterative model development without API costs.
- Structured output (JSON) with aspect scores improves interpretability.

### Open Questions
- Can a small judge model trained on GPT-3.5 annotations match GPT-4-level judgment quality?
- How does evaluation quality degrade for instruction types unseen during training?

---

## 5. Auto-J: Generative Judge for Scalable LLM-as-a-Judge

**Citation:** Li, Z., Peng, B., He, P., Galley, M., Gao, J., & Yan, R. (2024). *Generative Judge for Evaluating Alignment*. ICLR 2024. arXiv:2310.05470.

### Key Idea
Train a single 13B model (Auto-J) that handles **both pairwise comparison and single-answer critique** tasks, generating natural language critiques with final verdicts. Focuses on alignment evaluation (helpfulness, safety, factuality) across diverse scenarios.

### Judge Models
- **Auto-J-13B** (LLaMA-2-13B fine-tuned).
- Baselines: GPT-4, Claude-2, GPT-3.5-Turbo, PandaLM-7B, JudgeLM-13B.

### Dataset
- **58,660 training instances** (pairwise + single-answer) covering 58 application scenarios (open QA, coding, math, writing, etc.).
- Training labels: GPT-4 for pairwise; human-written critiques for single-answer tasks.

### Prompt Strategy
- **Pairwise**: given query + two responses, produce a critique for each, then a comparative verdict.
- **Single-answer critique**: given query + one response, produce a detailed critique (strengths, weaknesses, score 1–10).
- Scenario-specific system prompts are prepended to tailor the judge to different task types.

### Experiment Setup
- Evaluated on **1,000 human-annotated pairwise test cases** and **100 human-written critique references**.
- Agreement with human pairwise preferences.
- Critique quality: human rating of correctness and helpfulness of critiques.

### Metrics
- Pairwise agreement rate (%).
- Critique quality score (human-rated, 1–5).
- ROUGE-L and BERTScore for critique text overlap.

### Important Findings
- Auto-J reaches 86.4% agreement with humans on pairwise judgments—surpassing GPT-3.5-Turbo (79.5%) and approaching GPT-4 (88.7%).
- Single-answer critiques are highly useful as feedback for model improvement (RLHF signal).
- Scenario-specific prompts are crucial; without them, performance drops >10%.

### Open Questions
- Can a single judge model reliably evaluate all 58 application scenarios without scenario-specific fine-tuning?
- How to systematically collect human critiques at scale for training?

---

## 6. CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Models

**Citation:** Ke, P., Wen, B., Feng, Z., Liu, X., Lei, X., Cheng, J., … & Huang, M. (2023). *CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Models*. arXiv:2311.18702.

### Key Idea
Rather than producing a scalar score, train a judge model to generate **detailed, informative critiques** that identify specific errors and improvement suggestions. These critiques can then be used directly as training signals for RLHF.

### Judge Models
- **CritiqueLLM-6B** (ChatGLM-6B fine-tuned).
- Baselines: GPT-4, ChatGPT, InstructGPT, LLaMA-13B-Chat, MOSS-7B.

### Dataset
- **CritiqueLLM-7K** training set – 7,000 (instruction, response, critique) triples across 6 evaluation dimensions (helpfulness, factuality, safety, logical coherence, creativity, code correctness). Critiques written by human experts and GPT-4.

### Prompt Strategy
- Dialogue-based prompting: the judge is placed in a conversation role where it asks clarifying questions about the response if needed (multi-turn).
- Final prompt requests a structured critique per dimension + overall assessment.

### Experiment Setup
- Evaluated by human raters on 200 held-out samples.
- Human raters score critiques on Informativeness, Correctness, and Relevance (1–5).
- Also tested as reward signal in downstream RLHF training loop.

### Metrics
- Human-rated Informativeness, Correctness, Relevance (1–5 scale).
- Downstream reward model accuracy when using CritiqueLLM feedback.

### Important Findings
- CritiqueLLM-6B produces significantly more informative critiques than scalar-only evaluators.
- Downstream RLHF trained with CritiqueLLM feedback outperforms RLHF with scalar reward on human preference by ~6%.
- Multi-turn clarification dialogue improves critique specificity for ambiguous instructions.

### Open Questions
- Is multi-turn critique dialogue always necessary, or are single-pass critiques sufficient for most cases?
- How does critique quality scale with model size?

---

## 7. OFFSETBIAS: Leveraging Debiased Data for Tuning Evaluators

**Citation:** Park, J., Jwa, A., Ren, M., Kim, D., & Choi, S. (2024). *OFFSETBIAS: Leveraging Debiased Data for Tuning Evaluators*. arXiv:2407.06551.

### Key Idea
Identify and systematically mitigate six types of evaluator bias (verbosity, sycophancy, position, instruction following, style, refusal) by constructing a debiased training dataset. Trains **Themis** – a compact but highly debiased evaluator model.

### Judge Models
- **Themis-7B** (Mistral-7B fine-tuned on debiased data).
- Baselines: GPT-4, GPT-4-Turbo, Claude-3, Prometheus 2, Auto-J, JudgeLM, PandaLM.

### Dataset
- **OFFSETBIAS dataset** – 8,000 pairwise examples specifically constructed so that six known biases cannot be exploited as shortcuts (e.g., longer ≠ better; early-position ≠ better).
- Each bias type has ~1,300 balanced contrast pairs.

### Prompt Strategy
Minimal pairwise prompt without heuristics that trigger bias; the model must evaluate based on content quality alone. Evaluation rubrics are deliberately kept abstract to avoid leaking quality signals.

### Experiment Setup
- Evaluated on **RewardBench** and a new **BiasEval benchmark** (one sub-benchmark per bias type).
- Comparison across 10+ judge models.

### Metrics
- Accuracy on RewardBench (chat, reasoning, safety categories).
- Bias score per type (% of cases where biased response is incorrectly preferred).

### Important Findings
- Themis-7B achieves state-of-the-art on RewardBench among open models ≤13B (accuracy 88.3%).
- Debiased training reduces position bias by 40% and verbosity bias by 35% compared to baseline judge models.
- Existing large models (GPT-4) still exhibit measurable bias; training-time debiasing is more effective than prompt-level corrections alone.

### Open Questions
- Are there other bias types not yet catalogued that affect LLM judges?
- How does bias change when moving from pairwise to absolute scoring?

---

## 8. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Citation:** Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., … & Stoica, I. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.05685.

### Key Idea
Establish the LLM-as-a-Judge paradigm as a scalable alternative to human evaluation. Introduces **MT-Bench** (multi-turn question set) and **Chatbot Arena** (crowd-sourced pairwise arena), and studies the reliability, biases, and agreement of GPT-4 as a judge.

### Judge Models
- **GPT-4** (primary judge), GPT-3.5-Turbo, Claude-v1 as judges.
- Models evaluated: Vicuna, Alpaca, LLaMA, GPT-3.5, GPT-4, and others.

### Dataset
- **MT-Bench** – 80 multi-turn questions across 8 categories (writing, roleplay, reasoning, math, coding, extraction, STEM, humanities).
- **Chatbot Arena** – 30K+ crowd-sourced pairwise battle results.

### Prompt Strategy
- *Single-answer grading*: judge rates a single response on a scale of 1–10.
- *Pairwise comparison*: judge compares two responses and picks the better one.
- **Position-swap** to detect position bias: each pair is judged in both orders; ties detected by inconsistency.

### Experiment Setup
- Human agreement study: 3 expert raters on 3,300 samples from MT-Bench.
- Bias analysis: position bias, verbosity bias, self-enhancement bias.
- Correlation with Chatbot Arena ELO rankings.

### Metrics
- Agreement rate with humans (%).
- Kendall's τ rank correlation (ELO vs. judge ranking).
- Bias rates for position, verbosity, self-enhancement.

### Important Findings
- GPT-4 as judge agrees with humans ~80% of the time—similar to human inter-annotator agreement (~81%).
- Position bias exists: GPT-4 prefers responses in first position ~10% more when quality is equal.
- Verbosity bias: longer responses are preferred, even when quality is equivalent.
- GPT-4 shows self-enhancement bias (favors its own outputs).

### Open Questions
- Can position and verbosity biases be fully eliminated through prompt engineering?
- How should ties (inconsistent judgments across orderings) be handled?

---

## 9. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment

**Citation:** Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. EMNLP 2023. arXiv:2303.16634.

### Key Idea
Use GPT-4 with a chain-of-thought (CoT) prompting framework to evaluate natural language generation (NLG) tasks. Introduces **form-filling** paradigm: the evaluator fills in evaluation steps generated by the LLM itself, improving consistency and human alignment.

### Judge Models
- **GPT-4** (primary), GPT-3.5 as judges.
- Task-specific evaluators; not a fine-tuned model.

### Dataset
- **SummEval** benchmark (summarization): 1,600 expert-annotated (consistency, coherence, fluency, relevance).
- **TOPICAL-CHAT** (dialogue): 270 open-domain conversations.

### Prompt Strategy
Two-stage:
1. *Auto-CoT generation*: prompt the LLM to generate evaluation steps for a given criterion.
2. *Form filling*: use those steps + the text to be evaluated to produce a final score.
Probability weighting: final score is a token-probability-weighted average over all score tokens to reduce discretization noise.

### Experiment Setup
- Spearman correlation with expert human judgments on SummEval and TOPICAL-CHAT.
- Compared against ROUGE, BERTScore, BARTScore, and UniEval.

### Metrics
- Spearman correlation (ρ) with human assessments per dimension.
- System-level Pearson correlation.

### Important Findings
- G-Eval with GPT-4 achieves Spearman ρ = 0.514 on SummEval coherence—far above prior metrics (UniEval: 0.472, BARTScore: 0.232).
- Probability weighting significantly improves correlation vs. raw score generation.
- GPT-4-based evaluation is sensitive to the framing of the evaluation form.

### Open Questions
- Does the auto-CoT step always generate reliable evaluation criteria, or can it introduce artifacts?
- How does G-Eval perform on long-document tasks?

---

## 10. FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets

**Citation:** Ye, S., Jang, J., Kim, S., Kim, S., Shin, S., Cha, S., … & Hajishirzi, H. (2023). *FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets*. arXiv:2307.10928.

### Key Idea
Decompose LLM evaluation into **12 fine-grained skills** organized under 4 meta-abilities (Reasoning, Role Keeping, Instruction Following, Usefulness). Each response is evaluated only on the skills it requires, enabling more reliable and interpretable fine-grained assessment.

### Judge Models
- **GPT-4** and **Claude** as primary automatic judges; humans as gold standard.
- No specialized fine-tuned judge; framework is evaluator-agnostic.

### Dataset
- **FLASK Evaluation Set** – 1,740 instructions sampled from 10 existing instruction datasets (Dolly, OpenAssistant, FLAN, WizardLM, etc.), each annotated with required skills, difficulty level, and human judgments.

### Prompt Strategy
- Skill-level prompting: separate prompt per skill, asking the evaluator to score only on the relevant skill (1–5 scale).
- **Difficulty filtering**: hard instances (where even GPT-4 struggles) are flagged for human annotation.
- Protocol: model output → skill identification → per-skill score request.

### Experiment Setup
- Human annotators score 1,740 instances per skill.
- Pearson correlation between automatic judges (GPT-4, Claude) and human scores per skill.
- Model comparison: GPT-4, Claude, LLaMA, Vicuna, WizardLM across skill dimensions.

### Metrics
- Pearson correlation per skill (automatic vs. human).
- Skill score distributions per model.
- Reliability: inter-annotator agreement per skill.

### Important Findings
- Fine-grained skill-level evaluation reveals significant variation that aggregate scores hide (e.g., a model can excel at instruction following but fail at factuality).
- GPT-4 agrees with humans best on low-complexity skills (ρ ≈ 0.8+); agreement drops for high-complexity reasoning tasks.
- Skill decomposition exposes systematic weaknesses in frontier models not visible in aggregate benchmarks.

### Open Questions
- How should skills be defined for domains beyond text (multimodal, code, math)?
- Can skill tagging be automated reliably without human annotation?
