#!/usr/bin/env python3
"""Benchmark runner for `jev_gemini` across `Noul`, `Choice`, and `Score`.

Evaluates 20 `NoulTest`, 20 `ChoiceTest`, and 20 `ScoreTest` examples one
primitive at a time across Gemini 2.5 (`mode="logprobs"`) and Gemini 3.x
(`mode="monte_carlo"`) models on Vertex AI, recording accuracy, calibration
metrics, and wall-clock latency.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import json
import math
import pathlib
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Literal
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from benchmark.dataset import CHOICE_BENCHMARK
from benchmark.dataset import NOUL_BENCHMARK
from benchmark.dataset import SCORE_BENCHMARK
from jev_gemini import ChoiceAnswer
from jev_gemini import GeminiSystemOneClient
from jev_gemini import NoulAnswer
from jev_gemini import ScoreAnswer
from jev_gemini._types_compat import HAS_GOOGLE_GENAI
from jev_gemini._types_compat import types


DEFAULT_MODELS: list[tuple[str, Literal["logprobs", "monte_carlo"]]] = [
    ("gemini-2.5-flash", "logprobs"),
    ("gemini-2.5-flash-lite", "logprobs"),
    ("gemini-3.5-flash-lite", "monte_carlo"),
    ("gemini-3.7-flash", "monte_carlo"),
    ("gemini-3.8-flash", "monte_carlo"),
]

# Official Standard On-Demand pricing (USD per 1M tokens) and average billed
# token profile per call (`(input_price, output_price, avg_in_tok, avg_out_tok)`).
MODEL_PRICING_PROFILE: dict[str, tuple[float, float, float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40, 199.3, 1.0),
    "gemini-2.5-flash": (0.30, 2.50, 199.3, 1.0),
    "gemini-3.5-flash-lite": (0.30, 2.50, 201.0, 8.0),
    "gemini-3.5-flash": (1.50, 9.00, 201.0, 8.0),
    "gemini-3.7-flash": (0.75, 3.75, 201.0, 362.7),
    "gemini-3.8-flash": (0.75, 3.75, 201.0, 397.3),
}


def get_gcloud_access_token() -> str:
  """Retrieves an OAuth2 bearer token from `gcloud auth`."""
  for cmd in (
      ["gcloud", "auth", "application-default", "print-access-token"],
      ["gcloud", "auth", "print-access-token"],
  ):
    try:
      token = subprocess.check_output(
          cmd, text=True, stderr=subprocess.DEVNULL
      ).strip()
      if token:
        return token
    except Exception:  # pylint: disable=broad-except
      continue
  raise RuntimeError(
      "Could not obtain GCP access token via `gcloud auth print-access-token`."
  )


class VertexRESTModelsAdapter:
  """Lightweight REST adapter for Vertex AI `generateContent` when `google-genai` is absent."""

  def __init__(self, project: str, location: str) -> None:
    self.project = project
    self.location = location
    self._token_lock = threading.Lock()
    self._token = get_gcloud_access_token()
    self._token_ts = time.monotonic()

  def _get_token(self, force_refresh: bool = False) -> str:
    with self._token_lock:
      if force_refresh or (time.monotonic() - self._token_ts > 1800):
        self._token = get_gcloud_access_token()
        self._token_ts = time.monotonic()
      return self._token

  def raw_call(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    host = (
        "aiplatform.googleapis.com"
        if self.location == "global"
        else f"{self.location}-aiplatform.googleapis.com"
    )
    url = (
        f"https://{host}/v1beta1/projects/{self.project}/locations/{self.location}"
        f"/publishers/google/models/{model}:generateContent"
    )
    body_bytes = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(4):
      token = self._get_token(force_refresh=(attempt > 0))
      req = urllib.request.Request(
          url,
          data=body_bytes,
          headers={
              "Authorization": f"Bearer {token}",
              "Content-Type": "application/json",
          },
          method="POST",
      )
      try:
        with urllib.request.urlopen(req, timeout=60) as resp:
          return json.loads(resp.read().decode("utf-8"))
      except urllib.error.HTTPError as e:
        last_err = e
        if e.code in (401, 403, 429, 500, 502, 503, 504) and attempt < 3:
          time.sleep(1.5 * (2**attempt))
          continue
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {model}: {err_body}") from e
      except Exception as e:  # pylint: disable=broad-except
        last_err = e
        if attempt < 3:
          time.sleep(1.5 * (2**attempt))
          continue
        raise
    raise RuntimeError(f"Exhausted retries on {model}: {last_err}")

  def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
    gen_cfg: dict[str, Any] = {
        "temperature": config.temperature,
        "maxOutputTokens": config.max_output_tokens,
    }
    if config.response_mime_type:
      gen_cfg["responseMimeType"] = config.response_mime_type
    if config.response_schema:
      if isinstance(config.response_schema, dict):
        gen_cfg["responseSchema"] = config.response_schema
      else:
        gen_cfg["responseSchema"] = {
            "type": "STRING",
            "enum": list(config.response_schema),
        }
    if getattr(config, "candidate_count", None) is not None:
      gen_cfg["candidateCount"] = config.candidate_count
    if config.response_logprobs:
      gen_cfg["responseLogprobs"] = True
    if config.logprobs is not None:
      gen_cfg["logprobs"] = config.logprobs
    if config.thinking_config is not None:
      tc: dict[str, Any] = {}
      if getattr(config.thinking_config, "thinking_budget", None) is not None:
        tc["thinkingBudget"] = config.thinking_config.thinking_budget
      if getattr(config.thinking_config, "thinking_level", None) is not None:
        tc["thinkingLevel"] = config.thinking_config.thinking_level
      if tc:
        gen_cfg["thinkingConfig"] = tc

    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": str(contents)}]}],
        "generationConfig": gen_cfg,
    }
    if config.system_instruction:
      payload["systemInstruction"] = {
          "parts": [{"text": str(config.system_instruction)}]
      }

    data = self.raw_call(model, payload)
    candidates: list[Any] = []
    for c in data.get("candidates", []):
      parts = [
          types.Part(
              text=p.get("text", ""),
              thought=bool(p.get("thought", False)),
          )
          for p in c.get("content", {}).get("parts", [])
      ]
      lp_raw = c.get("logprobsResult")
      lp_obj = None
      if lp_raw:
        chosen = [
            types.LogprobsResultCandidate(
                token=ch.get("token", ""),
                log_probability=float(ch.get("logProbability", 0.0)),
                token_id=ch.get("tokenId"),
            )
            for ch in lp_raw.get("chosenCandidates", [])
        ]
        top_steps = []
        for ts in lp_raw.get("topCandidates", []):
          top_cands = [
              types.LogprobsResultCandidate(
                  token=tc.get("token", ""),
                  log_probability=float(tc.get("logProbability", 0.0)),
                  token_id=tc.get("tokenId"),
              )
              for tc in ts.get("candidates", [])
          ]
          top_steps.append(
              types.LogprobsResultTopCandidates(candidates=top_cands)
          )
        lp_obj = types.LogprobsResult(
            chosen_candidates=chosen, top_candidates=top_steps
        )
      candidates.append(
          types.Candidate(
              content=types.Content(parts=parts),
              logprobs_result=lp_obj,
              finish_reason=c.get("finishReason"),
          )
      )
    return types.GenerateContentResponse(candidates=candidates)


class VertexRESTClientShim:
  """Drop-in shim matching `google.genai.Client` for Vertex AI REST calls."""

  def __init__(self, project: str, location: str) -> None:
    self._api_client = type("ApiClient", (), {"vertexai": True})()
    self.models = VertexRESTModelsAdapter(project, location)


def _percentile(sorted_vals: list[float], q: float) -> float:
  if not sorted_vals:
    return 0.0
  idx = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals)) - 1)))
  return sorted_vals[idx]


def _compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
  if not latencies_ms:
    return {
        "total_time_s": 0.0,
        "mean_ms": 0.0,
        "median_ms": 0.0,
        "p95_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
    }
  s = sorted(latencies_ms)
  return {
      "total_time_s": round(sum(s) / 1000.0, 3),
      "mean_ms": round(statistics.mean(s), 1),
      "median_ms": round(statistics.median(s), 1),
      "p95_ms": round(_percentile(s, 0.95), 1),
      "min_ms": round(s[0], 1),
      "max_ms": round(s[-1], 1),
  }


def evaluate_model_on_benchmarks(
    *,
    model: str,
    mode: Literal["logprobs", "monte_carlo"],
    project: str,
    location: str,
    monte_carlo_samples: int = 8,
    intra_primitive_workers: int = 1,
    genai_client: Any | None = None,
) -> dict[str, Any]:
  """Evaluates one model on 20 NoulTest, 20 ChoiceTest, and 20 ScoreTest examples.

  Tests each primitive separately (`Noul` -> `Choice` -> `Score`).
  """
  client = GeminiSystemOneClient(
      model=model,
      project=project,
      location=location,
      mode=mode,
      monte_carlo_samples=monte_carlo_samples,
      genai_client=genai_client,
  )

  # 1. Evaluate 20 NoulTest examples (Primitive 1: Noul)
  noul_wall_start = time.perf_counter()

  def _eval_noul_case(case: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    ans = client.evaluate_one(state=case.state, question=case.question)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(ans, NoulAnswer)
    predicted_bool = (
        ans.value if ans.value is not None else bool(ans.probability >= 0.5)
    )
    correct = predicted_bool == case.expected_value
    y_true = 1.0 if case.expected_value else 0.0
    brier = (ans.probability - y_true) ** 2
    return {
        "test_id": case.test_id,
        "category": case.category,
        "expected": case.expected_value,
        "predicted": predicted_bool,
        "probability": round(ans.probability, 4),
        "confidence": round(ans.confidence, 4),
        "correct": correct,
        "brier": round(brier, 6),
        "latency_ms": round(dt_ms, 1),
    }

  if intra_primitive_workers > 1:
    with futures.ThreadPoolExecutor(max_workers=intra_primitive_workers) as ex:
      noul_details = list(ex.map(_eval_noul_case, NOUL_BENCHMARK))
  else:
    noul_details = [_eval_noul_case(c) for c in NOUL_BENCHMARK]
  noul_wall_s = round(time.perf_counter() - noul_wall_start, 3)

  noul_correct = sum(1 for d in noul_details if d["correct"])
  noul_latencies = [d["latency_ms"] for d in noul_details]
  noul_summary = {
      "num_examples": len(noul_details),
      "correct": noul_correct,
      "accuracy": round(noul_correct / len(noul_details), 4),
      "brier_score": round(
          statistics.mean(d["brier"] for d in noul_details), 4
      ),
      "mean_confidence": round(
          statistics.mean(d["confidence"] for d in noul_details), 4
      ),
      "wall_clock_s": noul_wall_s,
      "latency": _compute_latency_stats(noul_latencies),
      "details": noul_details,
  }
  print(
      f"[{model} ({mode})] Finished 20 NoulTest: "
      f"acc={noul_summary['accuracy']*100:.1f}%, "
      f"sum_time={noul_summary['latency']['total_time_s']}s "
      f"(mean={noul_summary['latency']['mean_ms']}ms)",
      file=sys.stderr,
      flush=True,
  )

  # 2. Evaluate 20 ChoiceTest examples (Primitive 2: Choice)
  choice_wall_start = time.perf_counter()

  def _eval_choice_case(case: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    ans = client.evaluate_one(state=case.state, question=case.question)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(ans, ChoiceAnswer)
    correct = ans.choice == case.expected_choice
    true_prob = float(ans.probabilities.get(case.expected_choice, 0.0))
    brier = sum(
        (p - (1.0 if k == case.expected_choice else 0.0)) ** 2
        for k, p in ans.probabilities.items()
    )
    return {
        "test_id": case.test_id,
        "category": case.category,
        "expected": case.expected_choice,
        "predicted": ans.choice,
        "true_option_prob": round(true_prob, 4),
        "confidence": round(ans.confidence, 4),
        "correct": correct,
        "brier": round(brier, 6),
        "latency_ms": round(dt_ms, 1),
    }

  if intra_primitive_workers > 1:
    with futures.ThreadPoolExecutor(max_workers=intra_primitive_workers) as ex:
      choice_details = list(ex.map(_eval_choice_case, CHOICE_BENCHMARK))
  else:
    choice_details = [_eval_choice_case(c) for c in CHOICE_BENCHMARK]
  choice_wall_s = round(time.perf_counter() - choice_wall_start, 3)

  choice_correct = sum(1 for d in choice_details if d["correct"])
  choice_latencies = [d["latency_ms"] for d in choice_details]
  choice_summary = {
      "num_examples": len(choice_details),
      "correct": choice_correct,
      "accuracy": round(choice_correct / len(choice_details), 4),
      "mean_true_prob": round(
          statistics.mean(d["true_option_prob"] for d in choice_details), 4
      ),
      "brier_score": round(
          statistics.mean(d["brier"] for d in choice_details), 4
      ),
      "wall_clock_s": choice_wall_s,
      "latency": _compute_latency_stats(choice_latencies),
      "details": choice_details,
  }
  print(
      f"[{model} ({mode})] Finished 20 ChoiceTest: "
      f"acc={choice_summary['accuracy']*100:.1f}%, "
      f"sum_time={choice_summary['latency']['total_time_s']}s "
      f"(mean={choice_summary['latency']['mean_ms']}ms)",
      file=sys.stderr,
      flush=True,
  )

  # 3. Evaluate 20 ScoreTest examples (Primitive 3: Score)
  score_wall_start = time.perf_counter()

  def _eval_score_case(case: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    ans = client.evaluate_one(state=case.state, question=case.question)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(ans, ScoreAnswer)
    nearest_level = int(round(ans.score))
    exact_match = nearest_level == case.expected_level
    abs_err = abs(ans.score - float(case.expected_level))
    within_1 = abs_err <= 1.0
    return {
        "test_id": case.test_id,
        "category": case.category,
        "expected_level": case.expected_level,
        "predicted_score": round(ans.score, 4),
        "nearest_level": nearest_level,
        "mode_level": ans.mode_level,
        "exact_match": exact_match,
        "within_1": within_1,
        "abs_error": round(abs_err, 4),
        "sq_error": round(abs_err**2, 6),
        "confidence": round(ans.confidence, 4),
        "latency_ms": round(dt_ms, 1),
    }

  if intra_primitive_workers > 1:
    with futures.ThreadPoolExecutor(max_workers=intra_primitive_workers) as ex:
      score_details = list(ex.map(_eval_score_case, SCORE_BENCHMARK))
  else:
    score_details = [_eval_score_case(c) for c in SCORE_BENCHMARK]
  score_wall_s = round(time.perf_counter() - score_wall_start, 3)

  score_exact = sum(1 for d in score_details if d["exact_match"])
  score_within_1 = sum(1 for d in score_details if d["within_1"])
  score_latencies = [d["latency_ms"] for d in score_details]
  score_mae = statistics.mean(d["abs_error"] for d in score_details)
  score_rmse = math.sqrt(statistics.mean(d["sq_error"] for d in score_details))
  score_summary = {
      "num_examples": len(score_details),
      "exact_correct": score_exact,
      "exact_accuracy": round(score_exact / len(score_details), 4),
      "within_1_correct": score_within_1,
      "within_1_accuracy": round(score_within_1 / len(score_details), 4),
      "mae": round(score_mae, 4),
      "rmse": round(score_rmse, 4),
      "wall_clock_s": score_wall_s,
      "latency": _compute_latency_stats(score_latencies),
      "details": score_details,
  }
  print(
      f"[{model} ({mode})] Finished 20 ScoreTest: "
      f"exact_acc={score_summary['exact_accuracy']*100:.1f}%, "
      f"MAE={score_summary['mae']:.3f}, "
      f"sum_time={score_summary['latency']['total_time_s']}s "
      f"(mean={score_summary['latency']['mean_ms']}ms)",
      file=sys.stderr,
      flush=True,
  )

  all_latencies = noul_latencies + choice_latencies + score_latencies
  total_correct = noul_correct + choice_correct + score_exact
  total_examples = len(noul_details) + len(choice_details) + len(score_details)

  return {
      "model": model,
      "mode": mode,
      "overall": {
          "total_examples": total_examples,
          "total_correct": total_correct,
          "overall_accuracy": round(total_correct / total_examples, 4),
          "wall_clock_s": round(
              noul_wall_s + choice_wall_s + score_wall_s, 3
          ),
          "latency": _compute_latency_stats(all_latencies),
      },
      "noul": noul_summary,
      "choice": choice_summary,
      "score": score_summary,
  }


def format_markdown_report(results: list[dict[str, Any]]) -> str:
  """Generates a comprehensive Markdown benchmark report table."""
  lines = [
      "# `jev-gemini` Benchmark Report: 20 `NoulTest`, 20 `ChoiceTest`, 20 `ScoreTest`",
      "",
      "Each primitive (`Noul`, `Choice`, `Score`) was evaluated independently one primitive at a time across 20 test cases per primitive (60 total single-primitive evaluations per model) on Vertex AI (`project=\"flex-hr-chatbot\"`, `location=\"global\"`).",
      "",
      "## 1. Overall Model Comparison (60 Single-Primitive Evaluations per Model)",
      "",
      "| Model | Mode | Overall Accuracy (`60` tests) | `Noul` Acc (`20`) | `Choice` Acc (`20`) | `Score` Exact Acc (`20`) | `Score` MAE (`0..4`) | Total Cumulative Time (s) | Mean Call Latency (ms) | Median Latency (ms) | p95 Latency (ms) |",
      "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ]
  for r in results:
    ov = r["overall"]
    nl = r["noul"]
    ch = r["choice"]
    sc = r["score"]
    lines.append(
        f"| **`{r['model']}`** | `{r['mode']}` | "
        f"**{ov['overall_accuracy']*100:.1f}%** ({ov['total_correct']}/60) | "
        f"{nl['accuracy']*100:.1f}% ({nl['correct']}/20) | "
        f"{ch['accuracy']*100:.1f}% ({ch['correct']}/20) | "
        f"{sc['exact_accuracy']*100:.1f}% ({sc['exact_correct']}/20) | "
        f"`{sc['mae']:.3f}` | "
        f"**`{ov['latency']['total_time_s']:.2f}s`** | "
        f"`{ov['latency']['mean_ms']:.1f} ms` | "
        f"`{ov['latency']['median_ms']:.1f} ms` | "
        f"`{ov['latency']['p95_ms']:.1f} ms` |"
    )

  lines.extend([
      "",
      "## 2. Per-Primitive Breakdown: `NoulTest` (20 Binary Proposition Examples)",
      "",
      "| Model | Mode | Accuracy (`20` tests) | Brier Score (↓ better) | Mean Confidence | Total Time Spent (20 calls) | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) |",
      "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ])
  for r in results:
    nl = r["noul"]
    lines.append(
        f"| **`{r['model']}`** | `{r['mode']}` | "
        f"**{nl['accuracy']*100:.1f}%** ({nl['correct']}/20) | "
        f"`{nl['brier_score']:.4f}` | "
        f"`{nl['mean_confidence']:.4f}` | "
        f"`{nl['latency']['total_time_s']:.2f}s` | "
        f"`{nl['latency']['mean_ms']:.1f} ms` | "
        f"`{nl['latency']['median_ms']:.1f} ms` | "
        f"`{nl['latency']['p95_ms']:.1f} ms` |"
    )

  lines.extend([
      "",
      "## 3. Per-Primitive Breakdown: `ChoiceTest` (20 Multi-Class Routing Examples)",
      "",
      "| Model | Mode | Top-1 Accuracy (`20` tests) | Mean True-Class Prob (↑ better) | Multi-Class Brier (↓ better) | Total Time Spent (20 calls) | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) |",
      "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ])
  for r in results:
    ch = r["choice"]
    lines.append(
        f"| **`{r['model']}`** | `{r['mode']}` | "
        f"**{ch['accuracy']*100:.1f}%** ({ch['correct']}/20) | "
        f"`{ch['mean_true_prob']:.4f}` | "
        f"`{ch['brier_score']:.4f}` | "
        f"`{ch['latency']['total_time_s']:.2f}s` | "
        f"`{ch['latency']['mean_ms']:.1f} ms` | "
        f"`{ch['latency']['median_ms']:.1f} ms` | "
        f"`{ch['latency']['p95_ms']:.1f} ms` |"
    )

  lines.extend([
      "",
      "## 4. Per-Primitive Breakdown: `ScoreTest` (20 Ordinal Rubric `0..4` Examples)",
      "",
      "| Model | Mode | Exact Level Acc (`20`) | Within ±1 Level Acc | MAE (↓ better) | RMSE (↓ better) | Total Time Spent (20 calls) | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) |",
      "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ])
  for r in results:
    sc = r["score"]
    lines.append(
        f"| **`{r['model']}`** | `{r['mode']}` | "
        f"**{sc['exact_accuracy']*100:.1f}%** ({sc['exact_correct']}/20) | "
        f"{sc['within_1_accuracy']*100:.1f}% ({sc['within_1_correct']}/20) | "
        f"`{sc['mae']:.4f}` | "
        f"`{sc['rmse']:.4f}` | "
        f"`{sc['latency']['total_time_s']:.2f}s` | "
        f"`{sc['latency']['mean_ms']:.1f} ms` | "
        f"`{sc['latency']['median_ms']:.1f} ms` | "
        f"`{sc['latency']['p95_ms']:.1f} ms` |"
    )
  lines.extend([
      "",
      "## 5. Model Pricing & Measured Token Cost Breakdown",
      "",
      "| Model | Mode | Input Price (`$ / 1M` tok) | Output + Thinking Price (`$ / 1M` tok) | Avg Input Tokens / Call | Avg Billed Output + Thinking Tokens / Call | Cost per Call (USD) | Cost for 60-Call Benchmark (USD) | Cost per `1,000` Calls (USD) |",
      "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ])
  for r in results:
    m_name = r["model"]
    in_price, out_price, avg_in, avg_out = MODEL_PRICING_PROFILE.get(
        m_name, (0.30, 2.50, 200.0, 8.0)
    )
    cost_per_call = (avg_in * in_price + avg_out * out_price) / 1_000_000.0
    cost_60 = cost_per_call * 60.0
    cost_1k = cost_per_call * 1000.0
    lines.append(
        f"| **`{m_name}`** | `{r['mode']}` | "
        f"`${in_price:.2f}` | `${out_price:.2f}` | "
        f"`{avg_in:.1f}` | `{avg_out:.1f}` | "
        f"`${cost_per_call:.6f}` | **`${cost_60:.4f}`** | **`${cost_1k:.4f}`** |"
    )
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Run jev_gemini benchmark suite.")
  parser.add_argument("--project", default="flex-hr-chatbot", help="GCP project ID.")
  parser.add_argument("--location", default="global", help="Vertex AI location.")
  parser.add_argument(
      "--intra-workers",
      type=int,
      default=2,
      help="Concurrency within a single primitive (default: 2).",
  )
  parser.add_argument(
      "--parallel-models",
      action="store_true",
      default=True,
      help="Evaluate different models in parallel across model endpoints.",
  )
  parser.add_argument(
      "--output-dir",
      default=str(pathlib.Path(__file__).resolve().parent / "results"),
      help="Directory to write JSON and Markdown benchmark reports.",
  )
  args = parser.parse_args()

  out_dir = pathlib.Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  shared_shim = (
      None
      if HAS_GOOGLE_GENAI
      else VertexRESTClientShim(project=args.project, location=args.location)
  )

  def _run_one(item: tuple[str, Literal["logprobs", "monte_carlo"]]) -> dict[str, Any]:
    model_id, mode_name = item
    res = evaluate_model_on_benchmarks(
        model=model_id,
        mode=mode_name,
        project=args.project,
        location=args.location,
        monte_carlo_samples=8,
        intra_primitive_workers=args.intra_workers,
        genai_client=shared_shim,
    )
    ckpt_path = out_dir / f"{model_id}_{mode_name}.json"
    ckpt_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res

  if args.parallel_models:
    with futures.ThreadPoolExecutor(max_workers=len(DEFAULT_MODELS)) as pool:
      all_results = list(pool.map(_run_one, DEFAULT_MODELS))
  else:
    all_results = [_run_one(item) for item in DEFAULT_MODELS]

  json_path = out_dir / "benchmark_results.json"
  md_path = out_dir / "BENCHMARK_REPORT.md"
  json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
  md_report = format_markdown_report(all_results)
  md_path.write_text(md_report, encoding="utf-8")
  print(md_report)


if __name__ == "__main__":
  main()
