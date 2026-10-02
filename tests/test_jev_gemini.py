"""Comprehensive unit test suite for `jev_gemini`."""

from __future__ import annotations

import asyncio
import enum
import math
import unittest
from typing import Any, Mapping, Sequence

from jev_gemini import Choice
from jev_gemini import ChoiceAnswer
from jev_gemini import compute_score_metrics
from jev_gemini import extract_decision_step_distribution
from jev_gemini import extract_monte_carlo_distribution
from jev_gemini import fit_temperature
from jev_gemini import GeminiJevClassifier
from jev_gemini import GeminiSystemOneClient
from jev_gemini import MissingLogprobsError
from jev_gemini import Noul
from jev_gemini import NoulAnswer
from jev_gemini import Score
from jev_gemini import ScoreAnswer
from jev_gemini import ThoughtTruncationError
from jev_gemini import UnsupportedBackendError
from jev_gemini._types_compat import types
from jev_gemini.extractor import CTRL94_START_THOUGHT
from jev_gemini.extractor import CTRL95_END_THOUGHT


def build_synthetic_response(
    steps: Sequence[tuple[str, float, Mapping[str, float]]],
) -> Any:
  """Constructs a synthetic `GenerateContentResponse` with `LogprobsResult`."""
  chosen = []
  top_per_step = []
  last_token = ""
  for chosen_tok, chosen_lp, top_dict in steps:
    last_token = chosen_tok
    chosen.append(
        types.LogprobsResultCandidate(
            token=chosen_tok, log_probability=chosen_lp
        )
    )
    cands = [
        types.LogprobsResultCandidate(token=tok, log_probability=lp)
        for tok, lp in top_dict.items()
    ]
    top_per_step.append(types.LogprobsResultTopCandidates(candidates=cands))

  return types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(
                  role="model", parts=[types.Part.from_text(text=last_token)]
              ),
              logprobs_result=types.LogprobsResult(
                  chosen_candidates=chosen,
                  top_candidates=top_per_step,
              ),
          )
      ]
  )


def build_monte_carlo_response(
    sampled_symbols: Sequence[str],
    *,
    include_thought_parts: bool = False,
) -> Any:
  """Constructs a synthetic multi-candidate `GenerateContentResponse` without logprobs."""
  candidates = []
  for sym in sampled_symbols:
    parts = []
    if include_thought_parts:
      parts.append(types.Part(text="Thinking about options...", thought=True))
    parts.append(types.Part.from_text(text=sym))
    candidates.append(
        types.Candidate(
            content=types.Content(role="model", parts=parts),
            logprobs_result=None,
            finish_reason="STOP",
        )
    )
  return types.GenerateContentResponse(candidates=candidates)


class FakeVertexGenAIClient:
  """Deterministic fake `google.genai.Client` recording calls and prompts."""

  def __init__(
      self,
      handler: Any,
  ) -> None:
    self.prompts: list[str] = []
    self.configs: list[Any] = []
    self._handler = handler
    self.models = self

  def generate_content(
      self, *, model: str, contents: Any, config: Any = None
  ) -> Any:
    del model
    prompt_str = str(contents)
    self.prompts.append(prompt_str)
    self.configs.append(config)
    return self._handler(prompt_str, config)


class ExtractorTest(unittest.TestCase):
  """Tier 1: Thought-stripping, last-step selection, and subword summing tests."""

  def test_strips_gemini3_4_token_thought_wrapper_and_leaked_article_a(self) -> None:
    # Step 2 inside thought block leaks English word "A" with logprob=0.0!
    # A naive `pick="max"` extractor would pick step 2 instead of step 5.
    response = build_synthetic_response([
        (CTRL94_START_THOUGHT, -0.001, {CTRL94_START_THOUGHT: -0.001}),
        ("thought", -0.001, {"thought": -0.001}),
        ("A", 0.0, {"A": 0.0}),  # Leaked English article inside thought!
        (CTRL95_END_THOUGHT, -0.001, {CTRL95_END_THOUGHT: -0.001}),
        ("B", math.log(0.85), {"B": math.log(0.85), "A": math.log(0.15)}),
    ])
    res = extract_decision_step_distribution(response, ["A", "B"])
    self.assertEqual(res.selected_step_index, 4)
    self.assertEqual(res.thought_tokens_stripped, 4)
    self.assertAlmostEqual(res.symbol_linear_probs["B"], 0.85, places=6)
    self.assertAlmostEqual(res.symbol_linear_probs["A"], 0.15, places=6)

  def test_raises_thought_truncation_error_when_ctrl95_missing(self) -> None:
    response = build_synthetic_response([
        (CTRL94_START_THOUGHT, -0.001, {CTRL94_START_THOUGHT: -0.001}),
        ("thought", -0.001, {"thought": -0.001}),
    ])
    with self.assertRaises(ThoughtTruncationError):
      extract_decision_step_distribution(response, ["A", "B"])

  def test_sums_subword_surface_variants_and_records_unobserved(self) -> None:
    # "A" (0.60) and " A" (0.20) must sum to 0.80; "C" is unobserved (0.0)
    response = build_synthetic_response([
        (
            "A",
            math.log(0.60),
            {
                "A": math.log(0.60),
                " A": math.log(0.20),
                "B": math.log(0.15),
            },
        )
    ])
    res = extract_decision_step_distribution(response, ["A", "B", "C"])
    self.assertAlmostEqual(res.symbol_linear_probs["A"], 0.80, places=6)
    self.assertAlmostEqual(res.symbol_linear_probs["B"], 0.15, places=6)
    self.assertEqual(res.symbol_linear_probs["C"], 0.0)
    self.assertEqual(res.unobserved_symbols, ["C"])

  def test_missing_logprobs_raises_error_unless_fallback_enabled(self) -> None:
    empty_resp = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="B")]
                ),
                logprobs_result=None,
            )
        ]
    )
    with self.assertRaises(MissingLogprobsError):
      extract_decision_step_distribution(empty_resp, ["A", "B"])

    fallback_res = extract_decision_step_distribution(
        empty_resp, ["A", "B"], allow_text_fallback=True
    )
    self.assertTrue(fallback_res.is_degraded_fallback)
    self.assertEqual(fallback_res.symbol_linear_probs["B"], 1.0)


class ScoreAndCalibrationMathTest(unittest.TestCase):
  """Tier 1: Mathematical invariants for Score, Popoviciu confidence, and NLL."""

  def test_score_concentrated_vs_adjacent_vs_polarized_bimodal(self) -> None:
    levels = [0.0, 1.0, 2.0, 3.0, 4.0]

    # Case 1: Point mass at middle level 2 -> score=2.0, confidence=1.0, not bimodal
    s1, ns1, conf1, std1, _, bim1, mode1 = compute_score_metrics(
        [0.0, 0.0, 1.0, 0.0, 0.0], levels
    )
    self.assertAlmostEqual(s1, 2.0)
    self.assertAlmostEqual(ns1, 0.5)
    self.assertAlmostEqual(conf1, 1.0)
    self.assertAlmostEqual(std1, 0.0)
    self.assertFalse(bim1)
    self.assertEqual(mode1, 2)

    # Case 2: Adjacent low split [0.5, 0.5, 0, 0, 0] -> score=0.5, confidence=0.75
    s2, _, conf2, std2, _, bim2, _ = compute_score_metrics(
        [0.5, 0.5, 0.0, 0.0, 0.0], levels
    )
    self.assertAlmostEqual(s2, 0.5)
    self.assertAlmostEqual(std2, 0.5)
    self.assertAlmostEqual(conf2, 0.75)
    self.assertFalse(bim2)

    # Case 3: Polarized bimodal split [0.5, 0, 0, 0, 0.5] -> score=2.0, confidence=0.0, bimodal=True
    s3, _, conf3, std3, _, bim3, _ = compute_score_metrics(
        [0.5, 0.0, 0.0, 0.0, 0.5], levels
    )
    self.assertAlmostEqual(s3, 2.0)
    self.assertAlmostEqual(std3, 2.0)
    self.assertAlmostEqual(conf3, 0.0)
    self.assertTrue(bim3)

  def test_fit_temperature_reduces_nll_and_ece_on_overconfident_logits(self) -> None:
    # Over-sharp distribution: 70% accuracy but model predicts 99% confidence
    raw_dists = [{"A": 0.99, "B": 0.01} for _ in range(10)]
    labels = ["A"] * 7 + ["B"] * 3
    cal = fit_temperature(raw_dists, labels)
    self.assertGreater(cal.optimal_temperature, 1.5)
    self.assertLess(cal.calibrated_nll, cal.raw_nll)
    self.assertLess(cal.calibrated_ece, cal.raw_ece)


class ClientAndPrimitivesTest(unittest.TestCase):
  """Tier 2 & 3: Vertex AI enforcement, Noul/Choice/Score, hierarchy, and LangChain."""

  def test_rejects_non_vertex_ai_backend(self) -> None:
    with self.assertRaises(UnsupportedBackendError):
      GeminiSystemOneClient(vertexai=False)

  def test_noul_abstain_band_and_echo_leakage_warning(self) -> None:
    fake = FakeVertexGenAIClient(
        lambda prompt, cfg: build_synthetic_response(
            [("A", math.log(0.55), {"A": math.log(0.55), "B": math.log(0.45)})]
        )
    )
    client = GeminiSystemOneClient(
        default_temperature=1.0, genai_client=fake
    )
    ans = client.evaluate_one(
        state={"allow_escalation": False, "user": "alice"},
        question=Noul(
            question="Should we allow escalation?",
            abstain_band=(0.40, 0.60),
        ),
    )
    self.assertIsInstance(ans, NoulAnswer)
    assert isinstance(ans, NoulAnswer)
    self.assertIsNone(ans.value)
    self.assertAlmostEqual(ans.probability, 0.55, places=4)
    self.assertIsNotNone(ans.diagnostics.echo_leakage_warning)

  def test_choice_permute_debias_and_shared_state_prefix_in_async_ask(self) -> None:
    def router_handler(prompt: str, cfg: Any) -> Any:
      del cfg
      if "QUESTION (NOUL)" in prompt:
        return build_synthetic_response(
            [("A", math.log(0.90), {"A": math.log(0.90), "B": math.log(0.10)})]
        )
      if "QUESTION (SCORE)" in prompt:
        return build_synthetic_response([
            (
                "C",
                math.log(0.70),
                {"A": math.log(0.10), "B": math.log(0.20), "C": math.log(0.70)},
            )
        ])
      return build_synthetic_response(
          [("A", math.log(0.80), {"A": math.log(0.80), "B": math.log(0.20)})]
      )

    fake = FakeVertexGenAIClient(router_handler)
    client = GeminiSystemOneClient(
        model="gemini-3-flash-preview",
        default_temperature=1.0,
        genai_client=fake,
    )
    state = {"ticket_id": 101, "body": "System down"}
    resp = asyncio.run(
        client.ask_async(
            state=state,
            questions={
                "safe": Noul(question="Is this safe?"),
                "route": Choice(
                    question="Which team?",
                    options={"sre": "SRE team", "billing": "Billing team"},
                    permute_debias=True,
                ),
                "sev": Score(
                    question="Severity?",
                    criteria=["Low", "Medium", "High"],
                    level_values=[1.0, 5.0, 10.0],
                ),
            },
        )
    )
    # Verify all prompts start with the exact same `=== STATE ===` prefix
    state_headers = [p.split("=== QUESTION")[0] for p in fake.prompts]
    self.assertEqual(len(set(state_headers)), 1)

    # Verify Gemini 3 model selected thinking_level="MINIMAL"
    self.assertEqual(
        fake.configs[0].thinking_config.thinking_level, "MINIMAL"
    )
    self.assertIsNone(fake.configs[0].thinking_config.thinking_budget)

    route_ans = resp["route"]
    assert isinstance(route_ans, ChoiceAnswer)
    self.assertIsNotNone(route_ans.diagnostics.jensen_shannon_divergence)

    sev_ans = resp["sev"]
    assert isinstance(sev_ans, ScoreAnswer)
    # Expected value: 0.1*1 + 0.2*5 + 0.7*10 = 8.1
    self.assertAlmostEqual(sev_ans.score, 8.1, places=5)

  def test_hierarchical_choice_supports_more_than_20_options(self) -> None:
    fake = FakeVertexGenAIClient(
        lambda prompt, cfg: build_synthetic_response(
            [("A", math.log(0.90), {"A": math.log(0.90), "B": math.log(0.10)})]
        )
    )
    client = GeminiSystemOneClient(default_temperature=1.0, genai_client=fake)
    many_options = {f"intent_{i}": f"Description {i}" for i in range(35)}
    ans = client.evaluate_one(
        state="User wants intent 0",
        question=Choice(question="Select intent", options=many_options),
    )
    assert isinstance(ans, ChoiceAnswer)
    self.assertEqual(ans.choice, "intent_0")
    self.assertEqual(len(ans.probabilities), 35)
    self.assertAlmostEqual(sum(ans.probabilities.values()), 1.0, places=5)

  def test_classify_enum_and_langchain_runnable_adapter(self) -> None:
    class Priority(str, enum.Enum):
      HIGH = "HIGH"
      LOW = "LOW"

    fake = FakeVertexGenAIClient(
        lambda prompt, cfg: build_synthetic_response(
            [("A", math.log(0.88), {"A": math.log(0.88), "B": math.log(0.12)})]
        )
    )
    client = GeminiSystemOneClient(default_temperature=1.0, genai_client=fake)

    enum_ans = client.classify_enum(
        state="Database offline",
        question="What priority?",
        enum_cls=Priority,
    )
    self.assertEqual(enum_ans.choice, "HIGH")
    self.assertAlmostEqual(enum_ans.probabilities["HIGH"], 0.88, places=5)

    # Test optional LangChain Runnable wrapper
    classifier = GeminiJevClassifier(
        client=client,
        questions={"priority": Choice.from_enum("What priority?", Priority)},
        output_key="jev_out",
    )
    out_state = classifier.invoke({"message": "Database offline"})
    self.assertIn("jev_out", out_state)
    self.assertEqual(out_state["jev_out"]["priority"].choice, "HIGH")


class MonteCarloModeTest(unittest.TestCase):
  """Tests for `mode="monte_carlo"` (`candidate_count=8` empirical sampling)."""

  def test_extract_monte_carlo_distribution_ignores_thought_parts_and_counts_candidates(
      self,
  ) -> None:
    response = build_monte_carlo_response(
        ["A", "A", "A", "A", "A", "A", "B", "B"],
        include_thought_parts=True,
    )
    res = extract_monte_carlo_distribution(response, ["A", "B", "C"])
    self.assertEqual(res.thought_tokens_stripped, 8)
    self.assertEqual(res.total_decoding_steps, 8)
    self.assertAlmostEqual(res.symbol_linear_probs["A"], 6.0 / 8.0, places=6)
    self.assertAlmostEqual(res.symbol_linear_probs["B"], 2.0 / 8.0, places=6)
    self.assertAlmostEqual(res.symbol_linear_probs["C"], 0.0, places=6)
    self.assertEqual(res.unobserved_symbols, ["C"])
    self.assertAlmostEqual(res.matched_enum_mass, 1.0, places=6)

  def test_extract_monte_carlo_distribution_applies_dirichlet_smoothing(
      self,
  ) -> None:
    response = build_monte_carlo_response(["A"] * 8)
    res = extract_monte_carlo_distribution(
        response, ["A", "B"], smoothing_alpha=0.5
    )
    # With N=8, K=2, alpha=0.5: P(A) = (8 + 0.5) / (8 + 1.0) = 8.5 / 9.0
    self.assertAlmostEqual(res.symbol_linear_probs["A"], 8.5 / 9.0, places=6)
    self.assertAlmostEqual(res.symbol_linear_probs["B"], 0.5 / 9.0, places=6)
    self.assertEqual(res.unobserved_symbols, ["B"])

  def test_monte_carlo_client_evaluates_noul_choice_and_score_across_3x_models(
      self,
  ) -> None:
    def mc_handler(prompt: str, cfg: Any) -> Any:
      del cfg
      if "QUESTION (NOUL)" in prompt:
        # 6/8 True ("A"), 2/8 False ("B") -> 0.75
        return build_monte_carlo_response(
            ["A", "A", "A", "A", "A", "A", "B", "B"],
            include_thought_parts=True,
        )
      if "QUESTION (SCORE)" in prompt:
        # 4/8 level C (2.0), 4/8 level D (3.0) -> expected score = 2.5
        return build_monte_carlo_response(
            ["C", "C", "C", "C", "D", "D", "D", "D"],
            include_thought_parts=True,
        )
      # Choice: 7/8 option A, 1/8 option B
      return build_monte_carlo_response(
          ["A", "A", "A", "A", "A", "A", "A", "B"],
          include_thought_parts=True,
      )

    fake = FakeVertexGenAIClient(mc_handler)
    client_38 = GeminiSystemOneClient(
        model="gemini-3.8-flash",
        mode="monte_carlo",
        monte_carlo_samples=8,
        genai_client=fake,
    )
    resp = client_38.ask(
        state={"incident": "Intermittent 502 errors on API gateway"},
        questions={
            "is_outage": Noul(question="Is this an active outage?"),
            "team": Choice(
                question="Route to which team?",
                options={"sre": "Site Reliability", "billing": "Billing"},
            ),
            "severity": Score(
                question="Rate severity",
                criteria=["0-Info", "1-Low", "2-Medium", "3-High", "4-Crit"],
            ),
        },
    )

    # Verify config omits response_logprobs and sets candidate_count=8, temperature=1.0
    cfg = fake.configs[0]
    self.assertIsNone(cfg.response_logprobs)
    self.assertIsNone(cfg.logprobs)
    self.assertEqual(cfg.candidate_count, 8)
    self.assertEqual(cfg.temperature, 1.0)
    # Verify gemini-3.8-flash selects thinking_level="LOW" and max_output_tokens >= 256
    self.assertEqual(cfg.thinking_config.thinking_level, "LOW")
    self.assertGreaterEqual(cfg.max_output_tokens, 256)

    # Verify gemini-3.5-flash-lite selects thinking_level="MINIMAL" and max_output_tokens=16
    client_35 = GeminiSystemOneClient(
        model="gemini-3.5-flash-lite",
        mode="monte_carlo",
        genai_client=fake,
    )
    _ = client_35.evaluate_one(
        state="Test", question=Noul(question="Is this a test?")
    )
    cfg_35 = fake.configs[-1]
    self.assertEqual(cfg_35.thinking_config.thinking_level, "MINIMAL")
    self.assertEqual(cfg_35.max_output_tokens, 16)

    # Verify Noul, Choice, and Score outputs
    noul_ans = resp["is_outage"]
    assert isinstance(noul_ans, NoulAnswer)
    self.assertTrue(noul_ans.value)
    self.assertAlmostEqual(noul_ans.probability, 0.75, places=5)

    choice_ans = resp["team"]
    assert isinstance(choice_ans, ChoiceAnswer)
    self.assertEqual(choice_ans.choice, "sre")
    self.assertAlmostEqual(choice_ans.probabilities["sre"], 0.875, places=5)
    self.assertAlmostEqual(choice_ans.probabilities["billing"], 0.125, places=5)

    score_ans = resp["severity"]
    assert isinstance(score_ans, ScoreAnswer)
    self.assertAlmostEqual(score_ans.score, 2.5, places=5)
    self.assertFalse(score_ans.is_bimodal)

  def test_validates_monte_carlo_parameters(self) -> None:
    with self.assertRaises(ValueError):
      GeminiSystemOneClient(mode="invalid_mode")  # type: ignore[arg-type]
    with self.assertRaises(ValueError):
      GeminiSystemOneClient(mode="monte_carlo", monte_carlo_samples=0)
    with self.assertRaises(ValueError):
      GeminiSystemOneClient(mode="monte_carlo", monte_carlo_samples=9)
    with self.assertRaises(ValueError):
      GeminiSystemOneClient(mode="monte_carlo", monte_carlo_smoothing=-0.1)


class BenchmarkSuiteTest(unittest.TestCase):
  """Tests for `benchmark.dataset` (40 Noul, 40 Choice, 40 Score) and runner."""

  def test_benchmark_datasets_have_40_examples_each_and_run_cleanly(self) -> None:
    from benchmark import CHOICE_BENCHMARK
    from benchmark import NOUL_BENCHMARK
    from benchmark import SCORE_BENCHMARK
    from benchmark.run_benchmark import evaluate_model_on_benchmarks

    self.assertEqual(len(NOUL_BENCHMARK), 40)
    self.assertEqual(len(CHOICE_BENCHMARK), 40)
    self.assertEqual(len(SCORE_BENCHMARK), 40)

    fake = FakeVertexGenAIClient(
        lambda prompt, cfg: build_monte_carlo_response(["A"] * 8)
    )
    res = evaluate_model_on_benchmarks(
        model="gemini-3.5-flash-lite",
        mode="monte_carlo",
        project="test-project",
        location="global",
        intra_primitive_workers=1,
        genai_client=fake,
    )
    self.assertEqual(res["overall"]["total_examples"], 120)
    self.assertEqual(res["noul"]["num_examples"], 40)
    self.assertEqual(res["choice"]["num_examples"], 40)
    self.assertEqual(res["score"]["num_examples"], 40)


if __name__ == "__main__":
  unittest.main()

