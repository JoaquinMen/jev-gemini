# `jev-gemini`: Calibrated System-One Decision Primitives (`Noul`, `Choice`, `Score`) on Gemini Flash

`jev-gemini` turns **Gemini Flash on Vertex AI** (`google-genai` SDK) into a fast, calibrated "System One" decision model implementing all three primitives from TypeSafe AI's [Jev](https://www.langchain.com/blog/building-a-harness-with-jev):

- **`Noul`**: Calibrated binary proposition (`True` / `False` / abstention band) with echo-leakage protection.
- **`Choice`**: Calibrated categorical distribution over $K$ mutually exclusive options (or Python `enum.Enum` classes), with single-letter bijection (`A..T`), optional 2-pass permutation debiasing, and automatic 2-stage hierarchical tree routing for $20 < K \le 255$.
- **`Score`**: Continuous probability-weighted expected value $\mu_S = \sum_i v_i \tilde{p}_i$ over an $L$-level ordinal rubric ($2 \le L \le 10$), paired with **Popoviciu variance-bounded ordinal confidence** and explicit **bimodality detection** (`is_bimodal`).

---

## Quickstart

### 1. Run the Pure `google-genai` Demo (`demo_genai.py`)

No LangChain required! You can run `demo_genai.py` immediately in simulated mode (no GCP credentials required) or against live Vertex AI:

```bash
# Simulated mode (deterministic LogprobsResult fixtures):
python3 demo_genai.py

# Live Vertex AI mode (requires google-genai and GCP Application Default Credentials):
python3 demo_genai.py --live --project=YOUR_GCP_PROJECT --location=us-central1
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

## Running Unit Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
