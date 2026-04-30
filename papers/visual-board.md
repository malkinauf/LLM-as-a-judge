# Visual Board: Specialized Judge Models

---

## 1. Overview Map – Taxonomy of LLM-as-a-Judge Approaches

```
                        LLM-as-a-Judge
                              │
          ┌───────────────────┴──────────────────┐
          │                                       │
   Prompting-based                         Fine-tuned / Specialized
   (general LLMs)                            (dedicated judge LLMs)
          │                                       │
   ┌──────┴──────┐                    ┌───────────┴───────────┐
   │             │                    │                       │
 Single-      Pairwise          Score-based             Critique-based
 answer       comparison        (absolute)              (generative)
   │             │                    │                       │
 G-Eval       MT-Bench /       Prometheus (v1/v2)       Auto-J
 FLASK         Chatbot Arena    JudgeLM                  CritiqueLLM
               OFFSETBIAS /
               Themis
               PandaLM
```

---

## 2. Model Comparison Table

| Model | Base Model | Size | Evaluation Mode | Open Source | Training Data | Rubric Support |
|-------|-----------|------|-----------------|-------------|---------------|----------------|
| **Prometheus 1** | LLaMA-2 | 13B | Absolute scoring | ✅ | Feedback Collection (100K) | ✅ Fine-grained rubrics |
| **Prometheus 2** | Mistral / Mixtral | 7B / 8×7B | Absolute + Pairwise | ✅ | Feedback + Preference Collection | ✅ Fine-grained rubrics |
| **JudgeLM** | LLaMA-2 / Vicuna | 7B / 13B / 33B | Pairwise | ✅ | JudgeLM-100K | ❌ |
| **PandaLM** | LLaMA | 7B | Pairwise | ✅ | 300K GPT-3.5-labeled | ❌ |
| **Auto-J** | LLaMA-2 | 13B | Pairwise + Critique | ✅ | 58K (pairwise + critique) | ✅ Scenario-specific |
| **CritiqueLLM** | ChatGLM | 6B | Critique | ✅ | CritiqueLLM-7K | ✅ Dimension-based |
| **Themis** (OFFSETBIAS) | Mistral | 7B | Pairwise | ✅ | OFFSETBIAS-8K (debiased) | ❌ |
| **GPT-4** (MT-Bench) | GPT-4 | — | Pairwise + Absolute | ❌ | Proprietary | ❌ (prompt-only) |
| **G-Eval (GPT-4)** | GPT-4 | — | Absolute | ❌ | Proprietary | ✅ Auto-CoT |
| **FLASK (GPT-4)** | GPT-4 | — | Absolute per skill | ❌ | Proprietary | ✅ Skill-level |

---

## 3. Dataset Comparison

| Dataset | Papers | Size | Task Coverage | Labels | Access |
|---------|--------|------|---------------|--------|--------|
| **Feedback Collection** | Prometheus 1 | ~100K | General instructions | GPT-4 feedback + scores | Public |
| **Preference Collection** | Prometheus 2 | ~1K rubrics | General instructions | GPT-4 preferences | Public |
| **JudgeLM-100K** | JudgeLM | 100K | Reasoning, Math, Code, STEM, Humanities | GPT-4 pairwise labels | Public |
| **PandaLM-Eval** | PandaLM | ~300K train / ~3K test | General instructions | GPT-3.5 train / Human test | Public |
| **Auto-J Dataset** | Auto-J | 58,660 | 58 application scenarios | GPT-4 + human | Public |
| **CritiqueLLM-7K** | CritiqueLLM | 7,000 | 6 evaluation dimensions | Human expert + GPT-4 | Partial |
| **OFFSETBIAS** | Themis | 8,000 | Bias-targeted pairs | GPT-4 (debiased) | Public |
| **MT-Bench** | Zheng et al. | 80 questions | 8 categories, multi-turn | Human + GPT-4 | Public |
| **SummEval** | G-Eval | 1,600 | Summarization | Human expert | Public |
| **FLASK Eval Set** | FLASK | 1,740 | 10 instruction datasets | Human per skill | Public |

---

## 4. Prompt Strategy Comparison

| Model | Prompt Type | Rubric / Criteria | Reference Answer | Output Format |
|-------|------------|-------------------|-----------------|---------------|
| Prometheus 1 | Single-response + rubric | ✅ User-defined 5-level rubric | ✅ Required | Score + verbal feedback |
| Prometheus 2 | Dual-mode (absolute / pairwise) | ✅ User-defined rubric | ✅ / ❌ (optional) | Score + feedback / Preference + rationale |
| JudgeLM | Pairwise with swap augmentation | ❌ | ✅ Injected | Preference + explanation |
| PandaLM | Pairwise structured JSON | ❌ | ❌ | JSON: preference + 6 aspect scores |
| Auto-J | Pairwise + single-critique | ✅ Scenario-specific system prompt | ❌ | Critique + verdict |
| CritiqueLLM | Multi-turn dialogue | ✅ Per-dimension | ❌ | Structured critique per dimension |
| Themis | Minimal pairwise | ❌ Deliberately abstract | ❌ | Preference |
| G-Eval | Form-filling + CoT | ✅ Auto-generated CoT steps | ❌ | Token-probability-weighted score |
| FLASK | Per-skill prompting | ✅ Per skill (12 skills) | ❌ | Score per skill |

---

## 5. Evaluation Metrics Comparison

| Metric | Papers Using It | Notes |
|--------|----------------|-------|
| **Pearson correlation (r)** | Prometheus, G-Eval, FLASK | Between model scores and human scores |
| **Spearman correlation (ρ)** | Prometheus, G-Eval | Rank-order correlation |
| **Agreement rate (%)** | JudgeLM, Auto-J, MT-Bench | % of judgments matching human preference |
| **Cohen's κ** | PandaLM, MT-Bench | Inter-annotator agreement |
| **Kendall's τ** | JudgeLM | System-level ranking correlation |
| **Win rate** | Prometheus 2, MT-Bench | % times model A preferred over model B |
| **RewardBench accuracy** | Themis | Chat, reasoning, safety sub-categories |
| **ROUGE-L / BERTScore** | Auto-J | For critique text quality |
| **Human-rated critique quality** | CritiqueLLM, Auto-J | Informativeness, Correctness, Relevance |

---

## 6. Bias Landscape

| Bias Type | Papers Discussing It | Mitigation Strategy |
|-----------|---------------------|---------------------|
| **Position bias** | MT-Bench, JudgeLM, Themis | Swap augmentation; balanced training (OFFSETBIAS) |
| **Verbosity bias** | MT-Bench, Themis | Debiased training data (OFFSETBIAS) |
| **Self-enhancement bias** | MT-Bench | Noted; no strong mitigation proposed |
| **Sycophancy** | Themis | Debiased training data |
| **Knowledge-boundary** | JudgeLM | Reference answer injection |
| **Format/style bias** | Themis | Debiased training data |
| **Refusal bias** | Themis | Debiased training data |

---

## 7. Performance Summary (Human Agreement on Pairwise Tasks)

```
Agreement with Human Judgments (Pairwise) – Approximate Values

GPT-4            ████████████████████████████████████████████  ~88%
Auto-J-13B       ████████████████████████████████████████      ~86%
Prometheus 2-8x7B ██████████████████████████████████████████   ~85%
JudgeLM-33B      ████████████████████████████████████████      ~82%
Prometheus 2-7B  █████████████████████████████████████████     ~80%
Themis-7B        ████████████████████████████████████████      ~80% (RewardBench)
GPT-3.5-Turbo    ███████████████████████████████████           ~75%
JudgeLM-13B      █████████████████████████████████████         ~78%
PandaLM-7B       ██████████████████████████████████████        ~79% (w/ human)
Prometheus 1-13B ████████████████████████████████████████████  ~90% (on Vicuna Bench, correlation)

Note: Values are approximate and not directly comparable across papers (different benchmarks / settings).
```

---

## 8. Open Research Questions

| # | Open Question | Relevant Papers |
|---|---------------|-----------------|
| 1 | Can rubric-conditioned judges generalize to **unseen domains** without human-written rubrics? | Prometheus 1, 2 |
| 2 | How do biases (position, verbosity, self-enhancement) **interact**, and can all be mitigated simultaneously? | MT-Bench, Themis |
| 3 | Is a **single universal judge** model feasible, or do specialized judges per task domain work better? | Auto-J, JudgeLM |
| 4 | How should judges handle **questions exceeding their own knowledge limits**? | JudgeLM |
| 5 | Can **weight merging** generalize to combining more than two evaluation task models? | Prometheus 2 |
| 6 | How does judge quality degrade for **instruction types unseen** during training? | PandaLM, CritiqueLLM |
| 7 | Can fine-grained skill decomposition (FLASK) be **automated** without manual skill tagging? | FLASK |
| 8 | How does judge performance scale with **model size** (small vs. large judges)? | JudgeLM, Prometheus 2 |
| 9 | Can LLM judges be reliably applied to **multimodal** evaluation (text + image)? | Prometheus 2 |
| 10 | Is multi-turn **dialogue-based critique** always better than single-pass critique generation? | CritiqueLLM |

---

## 9. Timeline of Key Papers

```
2023 ─────────────────────────────────────────────────────── 2024

Mar     Jun         Oct             Nov     Jan     Apr     Jul
 │       │           │               │       │       │       │
G-Eval  MT-Bench   Prometheus 1    Critique Auto-J  Prometheus2  Themis
(GPT-4) Chatbot    JudgeLM         LLM      (ICLR   (v2)     (OFFSETBIAS)
        Arena      FLASK                    2024)
```
