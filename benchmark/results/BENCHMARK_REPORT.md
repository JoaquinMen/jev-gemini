# `jev-gemini` Benchmark Report: Accuracy, Latency, and Cost (`NoulTest`, `ChoiceTest`, `ScoreTest`)

## 1. Executive Summary & Benchmark Architecture

We created the `benchmark/` suite to evaluate **time spent**, **prediction accuracy**, and **token cost** across all three System-One primitives—testing **one primitive at a time** via `GeminiSystemOneClient.evaluate_one`:

- **`NOUL_BENCHMARK` (`20` `NoulTest` examples)**: 10 `True` and 10 `False` binary propositions spanning security guardrails (prompt injection), destructive operations, PII detection, incident detection, refund policy eligibility, human escalation triggers, factual consistency, HR leave rules, SQL injection (`code_safety`), and numerical SLA breaches (`sla_breach`).
- **`CHOICE_BENCHMARK` (`20` `ChoiceTest` examples)**: 4-way categorical routing across Cloud Support (`sre_incident`, `billing_ops`, `iam_security`, `feature_request`), HR Chatbot (`benefits_leave`, `payroll_tax`, `it_equipment`, `performance_career`), E-Commerce Triage (`order_tracking`, `returns_damaged`, `payment_declined`, `account_profile`), and GCP Architecture Selection (`bigquery`, `cloud_spanner`, `cloud_run`, `vertex_ai`).
- **`SCORE_BENCHMARK` (`20` `ScoreTest` examples)**: 5-level ordinal rubrics (`0..4`, 4 examples per level) across Incident Operational Severity and Customer Frustration / Urgency.

---

## 2. Overall Benchmark Comparison (60 Single-Primitive Evaluations per Model)

| Model | Mode | Overall Accuracy (`60` tests) | `Noul` Acc (`20`) | `Choice` Acc (`20`) | `Score` Exact Acc (`20`) | `Score` MAE (`0..4`) | Total Cumulative Time (`60` calls) | Mean Call Latency | Cost for `60` Calls (USD) | Cost per `1,000` Calls (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemini-2.5-flash`** | `logprobs` | **100.0%** (60/60) | **100.0%** (20/20) | **100.0%** (20/20) | **100.0%** (20/20) | `0.018` | **`59.49s`** | `991.6 ms` | **`$0.0038`** | **`$0.0625`** |
| **`gemini-2.5-flash-lite`** | `logprobs` | **96.7%** (58/60) | 90.0% (18/20) | **100.0%** (20/20) | **100.0%** (20/20) | `0.001` | **`56.21s`** | `936.8 ms` | **`$0.0012`** | **`$0.0204`** |
| **`gemini-3.5-flash-lite`** | `monte_carlo` | **100.0%** (60/60) | **100.0%** (20/20) | **100.0%** (20/20) | **100.0%** (20/20) | **`0.000`** | **`61.33s`** | `1,022.1 ms` | **`$0.0048`** | **`$0.0803`** |
| **`gemini-3.8-flash`** | `monte_carlo` | **100.0%** (60/60) | **100.0%** (20/20) | **100.0%** (20/20) | **100.0%** (20/20) | `0.013` | **`184.57s`** | `3,076.1 ms` | `$0.0986` | `$1.6433` |
| **`gemini-3.7-flash`** | `monte_carlo` | **100.0%** (60/60) | **100.0%** (20/20) | **100.0%** (20/20) | **100.0%** (20/20) | **`0.000`** | **`389.34s`** | `6,488.9 ms` | `$0.0907` | `$1.5120` |

---

## 3. Model Pricing & Measured Token Cost Breakdown

| Model | Mode | Standard Input Price (`$ / 1M` tok) | Standard Output + Thinking Price (`$ / 1M` tok) | Avg Input Tokens / Call | Avg Candidate Output Tokens / Call | Avg Billed Thinking Tokens / Call | Total Billed Output Tokens / Call | Cost per Single Call (USD) | Cost for 60-Call Benchmark (USD) | Cost per `1,000` Calls (USD) | Cost per `10,000` Calls (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemini-2.5-flash-lite`** | `logprobs` | `$0.10` | `$0.40` | `199.3` | `1.0` | `0.0` | `1.0` | `$0.000020` | **`$0.0012`** | **`$0.0204`** | **`$0.204`** |
| **`gemini-2.5-flash`** | `logprobs` | `$0.30` | `$2.50` | `199.3` | `1.0` | `0.0` | `1.0` | `$0.000063` | **`$0.0038`** | **`$0.0625`** | **`$0.625`** |
| **`gemini-3.5-flash-lite`** | `monte_carlo` ($N=8$) | `$0.30` | `$2.50` | `201.0` | `8.0` | **`0.0`** | **`8.0`** | `$0.000080` | **`$0.0048`** | **`$0.0803`** | **`$0.803`** |
| **`gemini-3.7-flash`** | `monte_carlo` ($N=8$) | `$0.75`* | `$3.75`* | `201.0` | `7.3` | `355.3` | `362.7` | `$0.001512` | `$0.0907` | `$1.5120` | `$15.120` |
| **`gemini-3.8-flash`** | `monte_carlo` ($N=8$) | `$0.75`* | `$3.75`* | `201.0` | `7.7` | `389.7` | `397.3` | `$0.001643` | `$0.0986` | `$1.6433` | `$16.433` |

*\*Note: `gemini-3.7-flash` and `gemini-3.8-flash` are priced at promotional rates of `$0.75 / 1M` input and `$3.75 / 1M` output (including thinking tokens) through December 31, 2026 (`$1.50 / 1M` input and `$7.50 / 1M` output starting January 1, 2027).*

---

## 4. Per-Primitive Results (Tested One Primitive at a Time)

### 4.1 Primitive 1: `NoulTest` (20 Binary Proposition Examples, `~150` Input Tokens/Call)

| Model | Mode | Accuracy (`20` tests) | Brier Score (↓ better) | Mean Confidence | Cumulative Time (`20` calls) | Mean Latency | Cost for `20` Calls (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemini-2.5-flash`** | `logprobs` | **100.0%** (20/20) | `0.0037` | `0.9762` | `20.61s` | `1,030.5 ms` | `$0.00095` |
| **`gemini-2.5-flash-lite`** | `logprobs` | **90.0%** (18/20) | `0.0891` | `0.9909` | `18.92s` | `946.2 ms` | `$0.00031` |
| **`gemini-3.5-flash-lite`** | `monte_carlo` | **100.0%** (20/20) | **`0.0000`** | `1.0000` | `20.37s` | `1,018.4 ms` | `$0.00130` |
| **`gemini-3.8-flash`** | `monte_carlo` | **100.0%** (20/20) | **`0.0000`** | `1.0000` | `69.03s` | `3,451.4 ms` | `$0.03533` |
| **`gemini-3.7-flash`** | `monte_carlo` | **100.0%** (20/20) | **`0.0000`** | `1.0000` | `143.93s` | `7,196.3 ms` | `$0.03068` |

### 4.2 Primitive 2: `ChoiceTest` (20 Multi-Class Routing Examples, `~200` Input Tokens/Call)

| Model | Mode | Top-1 Accuracy (`20` tests) | Mean True-Class Prob (↑ better) | Multi-Class Brier (↓ better) | Cumulative Time (`20` calls) | Mean Latency | Cost for `20` Calls (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemini-2.5-flash`** | `logprobs` | **100.0%** (20/20) | `0.9928` | `0.0001` | `19.55s` | `977.2 ms` | `$0.00124` |
| **`gemini-2.5-flash-lite`** | `logprobs` | **100.0%** (20/20) | `0.9926` | `0.0001` | `18.73s` | `936.3 ms` | `$0.00041` |
| **`gemini-3.5-flash-lite`** | `monte_carlo` | **100.0%** (20/20) | **`1.0000`** | **`0.0000`** | `20.46s` | `1,023.1 ms` | `$0.00161` |
| **`gemini-3.8-flash`** | `monte_carlo` | **100.0%** (20/20) | **`1.0000`** | **`0.0000`** | `57.72s` | `2,886.0 ms` | `$0.03999` |
| **`gemini-3.7-flash`** | `monte_carlo` | **100.0%** (20/20) | **`1.0000`** | **`0.0000`** | `69.79s` | `3,489.4 ms` | `$0.02979` |

### 4.3 Primitive 3: `ScoreTest` (20 Ordinal Rubric `0..4` Examples, `~250` Input Tokens/Call)

| Model | Mode | Exact Level Acc (`20`) | Within ±1 Level Acc | MAE (↓ better) | RMSE (↓ better) | Cumulative Time (`20` calls) | Mean Latency | Cost for `20` Calls (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemini-2.5-flash`** | `logprobs` | **100.0%** (20/20) | **100.0%** (20/20) | `0.0178` | `0.0599` | `19.34s` | `967.0 ms` | `$0.00154` |
| **`gemini-2.5-flash-lite`** | `logprobs` | **100.0%** (20/20) | **100.0%** (20/20) | `0.0010` | `0.0043` | `18.56s` | `928.0 ms` | `$0.00051` |
| **`gemini-3.5-flash-lite`** | `monte_carlo` | **100.0%** (20/20) | **100.0%** (20/20) | **`0.0000`** | **`0.0000`** | `20.50s` | `1,024.8 ms` | `$0.00191` |
| **`gemini-3.8-flash`** | `monte_carlo` | **100.0%** (20/20) | **100.0%** (20/20) | `0.0125` | `0.0559` | `57.82s` | `2,890.9 ms` | `$0.02333` |
| **`gemini-3.7-flash`** | `monte_carlo` | **100.0%** (20/20) | **100.0%** (20/20) | **`0.0000`** | **`0.0000`** | `175.62s` | `8,781.1 ms` | `$0.03026` |
