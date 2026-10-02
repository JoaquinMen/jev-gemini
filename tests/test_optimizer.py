"""Unit tests for `jev_gemini.optimizer` and the 120-example `benchmark.dataset`."""

from __future__ import annotations

import unittest
from typing import Any

from benchmark.dataset import CHOICE_BENCHMARK
from benchmark.dataset import NOUL_BENCHMARK
from benchmark.dataset import SCORE_BENCHMARK
from demo_genai import make_logprob_response
from jev_gemini import Choice
from jev_gemini import GeminiSystemOneClient
from jev_gemini import Noul
from jev_gemini import PromptOptimizer
from jev_gemini import Score
from jev_gemini import detect_primitive_type
from jev_gemini import optimize_primitive
from jev_gemini.optimizer import normalize_optimization_dataset


class _AdaptiveFakeGenAIClient:
  """Fake Vertex AI client whose accuracy improves when wording includes boundary rules."""

  def __init__(self) -> None:
    self.models = self

  def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
    del model, config
    prompt = str(contents)

    # 1. Noul evaluation (SQL injection / code_safety)
    if "QUESTION (NOUL)" in prompt:
      is_improved_prompt = "parameterized" in prompt.lower()
      # True positive SQL injection (f-string / % interpolation)
      if "{user_supplied_name}" in prompt or "request.args['status']" in prompt:
        return make_logprob_response(
            {"A": -0.001, "B": -9.5} if is_improved_prompt else {"A": -0.05, "B": -3.0},
            include_gemini3_thought_wrapper=False,
        )
      # Safe parameterized query: baseline gets tricked into False Positive ("A"),
      # while improved prompt with "parameterized" boundary rule predicts False ("B")!
      if is_improved_prompt:
        return make_logprob_response(
            {"A": -8.5, "B": -0.001},
            include_gemini3_thought_wrapper=False,
        )
      return make_logprob_response(
          {"A": -0.10, "B": -2.35},
          include_gemini3_thought_wrapper=False,
      )

    # 2. Choice evaluation (support_routing)
    if "QUESTION (CHOICE)" in prompt:
      state_section = prompt.split("=== QUESTION")[0].lower()
      is_improved_choice = "root cause" in prompt.lower()
      if "oomkilled" in state_section or "http 502" in state_section:
        if is_improved_choice:
          return make_logprob_response(
              {"A": -0.001, "B": -9.0, "C": -9.0, "D": -9.0},
              include_gemini3_thought_wrapper=False,
          )
        # Baseline gets confused by "billing-webhook" keyword on choice_21
        if "billing-webhook" in state_section:
          return make_logprob_response(
              {"B": -0.20, "A": -1.80, "C": -6.0, "D": -6.0},
              include_gemini3_thought_wrapper=False,
          )
        return make_logprob_response(
            {"A": -0.02, "B": -4.0, "C": -5.0, "D": -5.0},
            include_gemini3_thought_wrapper=False,
        )
      return make_logprob_response(
          {"B": -0.01, "A": -6.0, "C": -6.0, "D": -6.0},
          include_gemini3_thought_wrapper=False,
      )

    # 3. Score evaluation (incident_severity)
    if "QUESTION (SCORE)" in prompt:
      is_improved_score = "cosmetic" in prompt.lower() and "single-pixel" in prompt.lower()
      if "2 pixels on safari" in prompt.lower():
        if is_improved_score:
          return make_logprob_response(
              {"B": -0.001, "A": -9.0, "C": -12.0, "D": -14.0, "E": -16.0},
              include_gemini3_thought_wrapper=False,
          )
        return make_logprob_response(
            {"B": -0.45, "A": -1.05, "C": -8.0, "D": -10.0, "E": -12.0},
            include_gemini3_thought_wrapper=False,
        )
      return make_logprob_response(
          {"A": -0.001, "B": -9.0, "C": -12.0, "D": -14.0, "E": -16.0},
          include_gemini3_thought_wrapper=False,
      )

    raise AssertionError(f"Unexpected prompt: {prompt}")


class OptimizerAndDatasetTest(unittest.TestCase):
  """Tests the expanded benchmark dataset and `PromptOptimizer`."""

  def test_expanded_benchmark_datasets_are_valid_and_unique(self) -> None:
    self.assertGreaterEqual(len(NOUL_BENCHMARK), 40)
    self.assertGreaterEqual(len(CHOICE_BENCHMARK), 40)
    self.assertGreaterEqual(len(SCORE_BENCHMARK), 40)

    # Verify unique IDs and non-empty True/False Noul labels
    noul_ids = [item.test_id for item in NOUL_BENCHMARK]
    self.assertEqual(len(set(noul_ids)), len(NOUL_BENCHMARK))
    self.assertGreater(sum(1 for item in NOUL_BENCHMARK if item.expected_value), 0)
    self.assertGreater(
        sum(1 for item in NOUL_BENCHMARK if not item.expected_value), 0
    )

    # Verify unique Choice IDs
    choice_ids = [item.test_id for item in CHOICE_BENCHMARK]
    self.assertEqual(len(set(choice_ids)), len(CHOICE_BENCHMARK))

    # Verify unique Score IDs and coverage across all 0..4 levels
    score_ids = [item.test_id for item in SCORE_BENCHMARK]
    self.assertEqual(len(set(score_ids)), len(SCORE_BENCHMARK))
    for lvl in range(5):
      self.assertGreater(
          sum(1 for item in SCORE_BENCHMARK if item.expected_level == lvl),
          0,
      )

  def test_detect_primitive_type(self) -> None:
    self.assertEqual(detect_primitive_type(Noul("Is it safe?")), "noul")
    self.assertEqual(
        detect_primitive_type(Choice("Pick one:", {"a": "A", "b": "B"})),
        "choice",
    )
    self.assertEqual(
        detect_primitive_type(Score("Rate:", ["0 - low", "1 - high"])),
        "score",
    )

  def test_optimize_noul_improves_accuracy_and_brier(self) -> None:
    client = GeminiSystemOneClient(
        model="gemini-2.5-flash",
        genai_client=_AdaptiveFakeGenAIClient(),
    )
    code_safety_cases = normalize_optimization_dataset(
        NOUL_BENCHMARK, category="code_safety"
    )
    self.assertEqual(len(code_safety_cases), 4)

    call_counter = {"step": 0}

    def fake_proposer(
        reflection_prompt: str, p_type: str, parent: Any
    ) -> dict[str, Any]:
      self.assertEqual(p_type, "noul")
      self.assertIn("HIGHEST-ERROR", reflection_prompt)
      call_counter["step"] += 1
      if call_counter["step"] == 2:
        return {
            "rationale": "Added explicit exclusion for parameterized queries (%s / :param).",
            "question": (
                "Does this database query concatenate or interpolate untrusted "
                "input directly into SQL text (excluding parameterized bind queries)?"
            ),
            "true_description": (
                "Yes — untrusted variables are directly interpolated into the "
                "SQL string via f-strings, %, or .format()."
            ),
            "false_description": (
                "No — the query uses parameterized placeholders (%s, ?, :name) "
                "with a separate parameter tuple/dict."
            ),
        }
      return {
          "rationale": "Minor wording tweak.",
          "question": parent.question,
          "true_description": parent.true_description,
          "false_description": parent.false_description,
      }

    initial_noul = Noul(question="Is this database query vulnerable to SQL injection?")
    result = optimize_primitive(
        client=client,
        primitive=initial_noul,
        dataset=code_safety_cases,
        num_explorations=4,
        proposer_fn=fake_proposer,
    )

    self.assertEqual(result.primitive_type, "noul")
    self.assertEqual(len(result.history), 5)  # Baseline 0 + 4 explorations
    self.assertEqual(result.original_record.accuracy, 0.5)
    self.assertEqual(result.best_record.accuracy, 1.0)
    self.assertLess(result.best_record.mean_brier, result.original_record.mean_brier)
    self.assertEqual(result.best_record.corrected_errors, 2)
    self.assertEqual(result.best_record.introduced_errors, 0)
    self.assertIn("Prompt Optimization Report (`NOUL`)", result.summary_markdown())

  def test_optimize_choice_preserves_option_keys(self) -> None:
    client = GeminiSystemOneClient(
        model="gemini-2.5-flash",
        genai_client=_AdaptiveFakeGenAIClient(),
    )
    subset = [CHOICE_BENCHMARK[0], CHOICE_BENCHMARK[1], CHOICE_BENCHMARK[20]]
    initial_choice = CHOICE_BENCHMARK[0].question

    def fake_choice_proposer(
        reflection_prompt: str, p_type: str, parent: Any
    ) -> dict[str, Any]:
      del reflection_prompt, parent
      self.assertEqual(p_type, "choice")
      return {
          "rationale": "Clarified root cause over service name keywords.",
          "question": "Route this support ticket based on the technical root cause:",
          "options": {
              "sre_incident": "Active pod crashes (OOMKilled), HTTP 5xx outages, latency spikes regardless of service name",
              "billing_ops": "Invoice discrepancies, credit card charges, refunds",
              "iam_security": "401/403 authentication or IAM permission errors",
              "feature_request": "Enhancement requests with no active outage",
              "hallucinated_extra_key": "Should be ignored by schema guardrail!",
          },
      }

    result = optimize_primitive(
        client=client,
        primitive=initial_choice,
        dataset=subset,
        num_explorations=2,
        proposer_fn=fake_choice_proposer,
    )
    best_choice = result.best_primitive
    assert isinstance(best_choice, Choice)
    self.assertEqual(
        list(best_choice.options.keys()),
        list(initial_choice.options.keys()),
    )
    self.assertEqual(result.best_record.accuracy, 1.0)
    self.assertGreater(result.brier_improvement, 0.0)

  def test_optimize_score_preserves_level_count_and_reduces_brier(self) -> None:
    client = GeminiSystemOneClient(
        model="gemini-2.5-flash",
        genai_client=_AdaptiveFakeGenAIClient(),
    )
    subset = [SCORE_BENCHMARK[0], SCORE_BENCHMARK[4]]  # score_01 (0) and score_05 (1)
    initial_score = SCORE_BENCHMARK[0].question

    def fake_score_proposer(
        reflection_prompt: str, p_type: str, parent: Any
    ) -> dict[str, Any]:
      del reflection_prompt
      self.assertEqual(p_type, "score")
      new_criteria = list(parent.criteria)
      new_criteria[1] = (
          "1 - Minor issue / Single-pixel or cosmetic UI alignment glitch "
          "where all buttons and workflows still function"
      )
      return {
          "rationale": "Clarified Level 1 boundary for minor single-pixel UI alignment glitches.",
          "question": parent.question,
          "criteria": new_criteria,
      }

    optimizer = PromptOptimizer(client=client, proposer_fn=fake_score_proposer)
    result = optimizer.optimize(
        primitive=initial_score,
        dataset=subset,
        num_explorations=2,
    )
    best_score = result.best_primitive
    assert isinstance(best_score, Score)
    self.assertEqual(len(best_score.criteria), len(initial_score.criteria))
    self.assertLess(result.best_record.mean_brier, result.original_record.mean_brier)


if __name__ == "__main__":
  unittest.main()
