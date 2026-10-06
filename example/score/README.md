# `Score` Primitive Example (`score_05` — Incident Operational Severity)

`Score` evaluates a continuous **probability-weighted expected value** $\mu_S = \sum_{i=0}^{L-1} v_i p_i$ over an $L$-level ordinal rubric ($2 \le L \le 10$), paired with **Popoviciu variance-bounded ordinal confidence** and **bimodality detection** (`is_bimodal`).

---

## 1. Benchmark Case (`score_05` from [`benchmark/dataset.py`](../../benchmark/dataset.py))

- **Test ID**: `score_05`
- **Category**: `incident_severity`
- **Ground-Truth Label**: `expected_level = 1`

---

## 2. How to Set Up the Prompt & Evaluate `Score`

```python
from jev_gemini import GeminiSystemOneClient, Score
from jev_gemini.compiler import compile_primitive_prompt

# 1. Define the STATE and the 5-level Score rubric (0..4)
state = "Tooltip alignment on the secondary settings modal is shifted by 2 pixels on Safari; all buttons work."

question = Score(
    question="Rate the operational severity of this incident report from 0 to 4:",
    criteria=[
        "0 - Informational / Routine healthy operation or cosmetic doc typo",
        "1 - Minor issue / Slight latency increase or single non-critical UI glitch with easy workaround",
        "2 - Moderate degradation / Non-core feature impaired or partial error rate in a single zone",
        "3 - Major disruption / Core production workflow failing for a large fraction of users",
        "4 - Critical catastrophe / Total global production outage or active data breach / data loss",
    ],
)

# 2. Inspect the compiled prompt and single-letter bijection
compiled = compile_primitive_prompt(state, question)
print("Symbols:", compiled.symbols)          # ('A', 'B', 'C', 'D', 'E')
print("Bijection:", compiled.symbol_to_key)  # {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': '4'}
print(compiled.prompt)

# 3. Evaluate on Vertex AI (mode="logprobs" on Gemini 2.5 or mode="monte_carlo" on Gemini 3.x)
client = GeminiSystemOneClient(
    model="gemini-2.5-flash",
    project="your-gcp-project",
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
Tooltip alignment on the secondary settings modal is shifted by 2 pixels on Safari; all buttons work.

=== QUESTION (SCORE) ===
Rate the operational severity of this incident report from 0 to 4:

=== RUBRIC LEVELS (Ordered Lowest to Highest) ===
[A] (Level 0): 0 - Informational / Routine healthy operation or cosmetic doc typo
[B] (Level 1): 1 - Minor issue / Slight latency increase or single non-critical UI glitch with easy workaround
[C] (Level 2): 2 - Moderate degradation / Non-core feature impaired or partial error rate in a single zone
[D] (Level 3): 3 - Major disruption / Core production workflow failing for a large fraction of users
[E] (Level 4): 4 - Critical catastrophe / Total global production outage or active data breach / data loss

Select the single best letter (A, B, C, D, E):
```

### Constrained Decoding Schema (`GenerateContentConfig`)
- `response_mime_type`: `"text/x.enum"`
- `response_schema`: `{"type": "STRING", "enum": ["A", "B", "C", "D", "E"]}`
- `symbol_to_key`: `{"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}`

---

## 4. Output (`ScoreAnswer`)

```json
{
  "type": "score",
  "score": 0.7541,
  "normalized_score": 0.1885,
  "confidence": 0.7508,
  "std_dev": 0.4306,
  "entropy": 0.3463,
  "is_bimodal": false,
  "mode_level": 1,
  "distribution": [
    0.2459,
    0.7541,
    0.0,
    0.0,
    0.0
  ],
  "level_values": [
    0.0,
    1.0,
    2.0,
    3.0,
    4.0
  ],
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
| `score` | `0.7541` | Continuous expected value $\mu_S = \sum_{i=0}^{4} v_i p_i = 0(0.2459) + 1(0.7541) = 0.7541$. |
| `normalized_score` | `0.1885` | Expected value rescaled to $[0.0, 1.0]$ ($\mu_S / 4.0$). |
| `mode_level` | `1` | Highest-probability discrete rubric level ($p_1 = 0.7541$, `round(score) = 1`). |
| `confidence` | `0.7508` | Popoviciu variance-bounded ordinal confidence $1 - 2\sigma_S / (v_{\max} - v_{\min})$. |
| `std_dev` | `0.4306` | Ordinal standard deviation $\sigma_S = \sqrt{\sum_i p_i (v_i - \mu_S)^2}$. |
| `is_bimodal` | `false` | `False` because probability mass is shared between adjacent levels `0` and `1` rather than split across non-adjacent extremes. |
| `distribution` | `[0.2459, 0.7541, 0.0, 0.0, 0.0]` | Calibrated probability mass $[p_0, p_1, p_2, p_3, p_4]$ across levels `0..4`. |
