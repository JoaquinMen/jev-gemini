# `Choice` Primitive Example (`choice_01` — Support Ticket Routing)

`Choice` evaluates a calibrated **categorical probability distribution** over $K$ mutually exclusive options (or a Python `enum.Enum`), mapping each option key to a single uppercase letter (`A..T`).

---

## 1. Benchmark Case (`choice_01` from [`benchmark/dataset.py`](../../benchmark/dataset.py))

- **Test ID**: `choice_01`
- **Category**: `support_routing`
- **Ground-Truth Label**: `expected_choice = "sre_incident"`

---

## 2. How to Set Up the Prompt & Evaluate `Choice`

```python
from jev_gemini import GeminiSystemOneClient, Choice
from jev_gemini.compiler import compile_primitive_prompt

# 1. Define the STATE and the Choice options
state = "Our GKE ingress controller in europe-west1 is returning HTTP 502 for all traffic."

question = Choice(
    question="Route this support ticket:",
    options={
        "sre_incident": "Production outages, 5xx errors, GKE/compute crashes, latency spikes",
        "billing_ops": "Invoices, credit card charges, refunds, subscription tier billing",
        "iam_security": "Permission denied errors, SSO/SAML login, API key rotation, IAM roles",
        "feature_request": "Product roadmap requests, new UI features, enhancement suggestions",
    },
)

# 2. Inspect the compiled prompt and single-letter bijection
compiled = compile_primitive_prompt(state, question)
print("Symbols:", compiled.symbols)          # ('A', 'B', 'C', 'D')
print("Bijection:", compiled.symbol_to_key)  # {'A': 'sre_incident', 'B': 'billing_ops', 'C': 'iam_security', 'D': 'feature_request'}
print(compiled.prompt)

# 3. Evaluate on Vertex AI (mode="logprobs" on Gemini 2.5 or mode="monte_carlo" on Gemini 3.x)
client = GeminiSystemOneClient(
    model="gemini-2.5-flash",
    project="flex-hr-chatbot",
    location="global",
    mode="logprobs",
)

answer = client.evaluate_one(state=state, question=question)
print(answer.to_dict())
```

---

## 3. Compiled Prompt Sent to Gemini

### System Instruction
```text
You are a calibrated System-One state evaluator.
Evaluate the provided STATE and answer the QUESTION by selecting the single uppercase letter corresponding to the best option.
Do not explain your reasoning. Output only the single option letter.
```

### User Prompt (`contents`)
```text
=== STATE ===
Our GKE ingress controller in europe-west1 is returning HTTP 502 for all traffic.

=== QUESTION (CHOICE) ===
Route this support ticket:

=== OPTIONS ===
[A]: (sre_incident) Production outages, 5xx errors, GKE/compute crashes, latency spikes
[B]: (billing_ops) Invoices, credit card charges, refunds, subscription tier billing
[C]: (iam_security) Permission denied errors, SSO/SAML login, API key rotation, IAM roles
[D]: (feature_request) Product roadmap requests, new UI features, enhancement suggestions

Select the single best letter (A, B, C, D):
```

### Constrained Decoding Schema (`GenerateContentConfig`)
- `response_mime_type`: `"text/x.enum"`
- `response_schema`: `{"type": "STRING", "enum": ["A", "B", "C", "D"]}`
- `symbol_to_key`: `{"A": "sre_incident", "B": "billing_ops", "C": "iam_security", "D": "feature_request"}`

---

## 4. Output (`ChoiceAnswer`)

```json
{
  "type": "choice",
  "choice": "sre_incident",
  "confidence": 0.9966,
  "normalized_confidence": 0.9955,
  "margin": 0.9950,
  "certainty": 0.9818,
  "probabilities": {
    "sre_incident": 0.9966,
    "iam_security": 0.0016,
    "billing_ops": 0.0009,
    "feature_request": 0.0009
  },
  "raw_probabilities": {
    "sre_incident": 0.999997,
    "iam_security": 0.000002,
    "billing_ops": 0.000001,
    "feature_request": 0.000001
  },
  "diagnostics": {
    "selected_step_index": 0,
    "total_decoding_steps": 1,
    "thought_tokens_stripped": 0,
    "matched_enum_mass": 1.0,
    "unobserved_options": [],
    "calibration_temperature": 2.0,
    "echo_leakage_warning": null,
    "jensen_shannon_divergence": null,
    "is_degraded_fallback": false
  }
}
```

### Field Breakdown
| Field | Value | Meaning |
| :--- | :--- | :--- |
| `choice` | `"sre_incident"` | Winning option key ($\arg\max_k p_k$). |
| `confidence` | `0.9966` | Calibrated probability of the winning option ($99.66\%$). |
| `normalized_confidence` | `0.9955` | Chance-adjusted peak margin $(K \cdot p_{\max} - 1)/(K - 1) \in [0, 1]$. |
| `margin` | `0.9950` | Probability gap $p_{(1)} - p_{(2)}$ between the top-1 and top-2 options. |
| `certainty` | `0.9818` | Normalized inverse Shannon entropy $1 - H(p)/\ln K \in [0, 1]$. |
| `probabilities` | `{"sre_incident": 0.9966, ...}` | Full calibrated probability distribution sorted descending. |
