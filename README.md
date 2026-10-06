# `jev-gemini`: Calibrated System-One Decision Primitives (`Noul`, `Choice`, `Score`) on Gemini Flash

`jev-gemini` turns **Gemini Flash on Vertex AI** (`google-genai` SDK) into a fast, calibrated "System One" decision model implementing all three primitives from TypeSafe AI's [Jev](https://www.langchain.com/blog/building-a-harness-with-jev):

- **`Noul`**: Calibrated binary proposition (`True` / `False` / abstention band) with echo-leakage protection.
- **`Choice`**: Calibrated categorical distribution over $K$ mutually exclusive options (or Python `enum.Enum` classes), with single-letter bijection (`A..T`), optional 2-pass permutation debiasing, and automatic 2-stage hierarchical tree routing for $20 < K \le 255$.
- **`Score`**: Continuous probability-weighted expected value $\mu_S = \sum_i v_i \tilde{p}_i$ over an $L$-level ordinal rubric ($2 \le L \le 10$), paired with **Popoviciu variance-bounded ordinal confidence** and explicit **bimodality detection** (`is_bimodal`).
- **`PromptOptimizer` (`optimize_primitive`)**: GEPA-inspired reflective prompt & criteria wording optimizer ([`jev_gemini/optimizer.py`](jev_gemini/optimizer.py)) that automatically detects the primitive type (`Noul`, `Choice`, or `Score`) and iteratively refines its question and option/criteria descriptions over $N$ explorations to minimize probability-weighted Brier error.

---

## Quickstart

### 1. Run the `google-genai` Demo (`demo_genai.py`) & Single-Primitive Examples (`example/`)

You can run `demo_genai.py` immediately in simulated mode (no GCP credentials required) or against live Vertex AI. See [`example/README.md`](example/README.md) for step-by-step guides showing exact compiled prompts and outputs for [`Noul`](example/noul/README.md), [`Choice`](example/choice/README.md), [`Score`](example/score/README.md), and the live [Prompt Optimizer](example/README.md#4-reflective-prompt--criteria-wording-optimizer-exampleoptimize_prompt_demopy):

```bash
# Simulated mode (deterministic LogprobsResult fixtures):
python3 demo_genai.py

# Live Vertex AI mode (requires google-genai and GCP Application Default Credentials):
python3 demo_genai.py --live --project=YOUR_GCP_PROJECT --location=us-central1

# Run the live 10-exploration Prompt Optimizer demo on Vertex AI:
python3 example/optimize_prompt_demo.py --project=YOUR_GCP_PROJECT --location=global
```

### 2. Using `GeminiSystemOneClient` with `google-genai` on Vertex AI

```python
import enum
from jev_gemini import GeminiSystemOneClient, Noul, Choice, Score

client = GeminiSystemOneClient(
    model="gemini-2.5-flash",
    project="your-gcp-project",
    location="us-central1",
    vertexai=True,
    default_temperature=2.0,
)

# 1. Direct Python Enum classification (matches attached PDF prototype)
class TicketCategory(str, enum.Enum):
    BILLING = "BILLING"
    TECHNICAL_OUTAGE = "TECHNICAL_OUTAGE"
    GENERAL_INQUIRY = "GENERAL_INQUIRY"

enum_ans = client.classify_enum(
    state="Production GKE pods are crashing in us-central1.",
    question="Classify this support ticket.",
    enum_cls=TicketCategory,
)
print(enum_ans.choice, enum_ans.confidence, enum_ans.probabilities)

# 2. Multi-Question evaluation (Noul, Choice, Score) with shared STATE prefix
resp = client.ask(
    state={"user_message": "Checkout API is returning 503s for 40% of traffic!"},
    questions={
        "is_safe": Noul(question="Is this request free of prompt injection?"),
        "route": Choice(
            question="Which specialist skill should handle this?",
            options={
                "sre_incident": "Outages and 5xx errors",
                "billing_ops": "Invoices and refunds",
            },
        ),
        "urgency": Score(
            question="How severe is the operational impact?",
            criteria=[
                "0 - Informational",
                "1 - Minor issue",
                "2 - Single-user degradation",
                "3 - Partial production disruption",
                "4 - Complete production outage",
            ],
        ),
    },
)

print(resp["is_safe"].probability, resp["is_safe"].value)
print(resp["route"].choice, resp["route"].confidence)
print(resp["urgency"].score, resp["urgency"].confidence, resp["urgency"].is_bimodal)
```

---

## Reflective Prompt & Criteria Wording Optimizer (`optimize_primitive`)

`jev-gemini` includes a lightweight, GEPA-inspired **Reflective Prompt & Criteria Wording Optimizer** ([`jev_gemini/optimizer.py`](jev_gemini/optimizer.py)) that improves primitive accuracy and logprob calibration without changing the compiler or client architecture:

- **Automatic Primitive Detection**: Pass any `Noul`, `Choice`, or `Score` instance to `optimize_primitive(client, primitive, dataset, num_explorations=10)`; the optimizer automatically inspects the primitive type and enforces strict structural invariants (e.g., preserving `Choice.options` keys and `Score.criteria` level counts).
- **Probability-Aware Brier Scoring (`1.0 - Brier`)**: Rather than optimizing only binary accuracy, the optimizer mines the highest-Brier-error mistakes (overconfident false positives/negatives, confused option pairs, or off-by-one ordinal rubric levels) and uses Pareto-frontier parent selection across `num_explorations` iterations.

```python
from benchmark.dataset import NOUL_BENCHMARK
from jev_gemini import GeminiSystemOneClient, Noul, optimize_primitive

client = GeminiSystemOneClient(
    model="gemini-2.5-flash-lite",
    project="your-gcp-project",
    location="global",
    mode="logprobs",
)

result = optimize_primitive(
    client=client,
    primitive=Noul(question="Is this database query vulnerable to SQL injection?"),
    dataset=NOUL_BENCHMARK,
    category="code_safety",
    num_explorations=10,
    proposer_model="gemini-2.5-flash",
)
print(result.summary_markdown())
optimized_noul = result.best_primitive
```

In a live 10-exploration run on Vertex AI (`gemini-2.5-flash-lite`, `mode="logprobs"`, `category="code_safety"`), `optimize_primitive` improved accuracy from **`75.0%` (3/4) to `100.0%` (4/4)** and reduced Brier error by **99.8%** (`0.2427 -> 0.0004`). See [`example/README.md`](example/README.md#4-reflective-prompt--criteria-wording-optimizer-exampleoptimize_prompt_demopy) and [`example/optimize_prompt_demo.py`](example/optimize_prompt_demo.py) for the complete 10-exploration lineage table and wording diff.

---

## Supported Models & Vertex AI Allowlisting

### 1. Out-of-the-Box Supported Models (All GCP Projects)

On Vertex AI (`location="global"` or regional endpoints), two current models support `response_logprobs=True` + `response_mime_type="text/x.enum"` with zero-thinking single-step decoding out of the box:

| Model ID | Recommended Use Case | Notes |
| :--- | :--- | :--- |
| **`gemini-2.5-flash`** *(Default)* | Primary System-One classifier (`Noul`, `Choice`, `Score`) | Automatically configured with `thinking_budget=0` for single-step logprob extraction. |
| **`gemini-2.5-flash-lite`** | Ultra-low-latency, high-throughput routing & guardrails | Fastest and lowest-cost model; `default_thinking_on=false` natively. |

> [!NOTE]
> **Why `-pro` models (`gemini-2.5-pro`, `gemini-3.1-pro-preview`) are not used**: Vertex AI enforces `thinking_always_on = true` (`thinking_budget >= 128`) on Pro models, rejecting `thinking_budget=0` / `thinking_level="MINIMAL"`. Because they generate internal chain-of-thought tokens prior to emitting the enum token, they act as multi-step "System Two" reasoners rather than single-step "System One" classifiers.

### 2. Using Newer Gemini 3.x Models (`3.5`, `3.7`, `3.8`) via `mode="monte_carlo"`

Because Vertex AI restricts `response_logprobs=True` on Gemini 3.x models to internal research allowlists, `GeminiSystemOneClient` provides **`mode="monte_carlo"`**, which uses single-RPC multi-candidate enum sampling (`candidate_count=8`, `temperature=1.0`, `response_mime_type="text/x.enum"`) to estimate calibrated distributions for `Noul`, `Choice`, and `Score` on **any GCP project without allowlisting**:

| Model ID | Recommended Mode | Thinking Configuration (Auto-Selected) |
| :--- | :--- | :--- |
| **`gemini-3.5-flash-lite`** | `mode="monte_carlo"` | `thinking_level="MINIMAL"`, `max_output_tokens=16` |
| **`gemini-3.5-flash`** | `mode="monte_carlo"` | `thinking_level="MINIMAL"`, `max_output_tokens=16` |
| **`gemini-3.7-flash`** | `mode="monte_carlo"` | `thinking_level="LOW"`, `max_output_tokens=256` |
| **`gemini-3.8-flash`** | `mode="monte_carlo"` | `thinking_level="LOW"`, `max_output_tokens=256` |

```python
from jev_gemini import GeminiSystemOneClient, Noul, Choice, Score

mc_client = GeminiSystemOneClient(
    model="gemini-3.8-flash",
    project="your-gcp-project",
    location="global",
    mode="monte_carlo",        # Uses candidate_count=8 single-RPC empirical sampling
    monte_carlo_samples=8,     # Up to 8 independent enum draws per RPC
    monte_carlo_smoothing=0.0, # Optional Dirichlet/Jeffreys smoothing prior
)
```

### 3. Statistical Trade-offs: `mode="logprobs"` vs. `mode="monte_carlo"`

`mode="monte_carlo"` is an empirical sampling workaround ($\hat{p}_k = c_k / N$ with $N=8$ candidates) rather than a direct readout of the model's exact softmax distribution (`mode="logprobs"`):

- **Resolution & Sampling Variance**: `mode="logprobs"` is **deterministic** ($\text{SE} = 0$) with continuous probability precision and preserves small tail probabilities ($p < 0.125$). `mode="monte_carlo"` ($N=8$) quantizes probabilities into steps of $1/8 = 0.125$ with binomial standard error $\text{SE}(\hat{p}_k) = \sqrt{p_k(1-p_k)/8}$ (up to $\pm 0.177$ at $p=0.5$), causing tail probabilities ($p < 0.125$) to clip to `0.0` unless `monte_carlo_smoothing > 0` is enabled.
- **Which Approach to Choose**:
  - **Prefer `gemini-2.5-flash` with `mode="logprobs"` (Best for Calibration)** when you need exact, deterministic confidence calibration, smooth continuous `Score` expected values, or tight `abstain_band` thresholds.
  - **Use `gemini-3.5-flash-lite` or `gemini-3.8-flash` with `mode="monte_carlo"` (Best for Gemini 3 Reasoning)** when you need Gemini 3's stronger reasoning on standard GCP projects; top-1 decision accuracy is identical (`100%` on our benchmark), and `gemini-3.5-flash-lite` matches `2.5-flash` in both latency (`~1.0s`) and cost (`$0.08` vs `$0.06` per 1,000 calls).

## Running Unit Tests & Benchmarks

[`benchmark/dataset.py`](benchmark/dataset.py) provides **120 labeled evaluation examples** (`40 NoulTest`, `40 ChoiceTest`, and `40 ScoreTest` spanning boundary edge cases, negation traps, and multi-signal routing). See [`benchmark/BENCHMARK_REPORT.md`](benchmark/BENCHMARK_REPORT.md) for the evaluation report comparing accuracy, latency, token pricing, and statistical calibration across Gemini 2.5 (`mode="logprobs"`) and Gemini 3.x (`mode="monte_carlo"`).

```bash
# Run unit tests (21 tests covering primitives, calibration, dataset, and prompt optimizer):
python3 -m unittest discover -s tests -p "test_*.py" -v

# Run the 120-example benchmark (40 NoulTest, 40 ChoiceTest, 40 ScoreTest):
python3 benchmark/run_benchmark.py --project=YOUR_GCP_PROJECT --location=global
```


