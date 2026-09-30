"""Runs 1 benchmark example per primitive (`noul_01`, `choice_01`, `score_05`).

Usage:
  # Simulated mode (deterministic fixtures matching the benchmark outputs):
  python3 example/run_examples.py

  # Live Vertex AI mode:
  python3 example/run_examples.py --live --project=flex-hr-chatbot --location=global
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from benchmark.dataset import CHOICE_BENCHMARK
from benchmark.dataset import NOUL_BENCHMARK
from benchmark.dataset import SCORE_BENCHMARK
from demo_genai import make_logprob_response
from jev_gemini import GeminiSystemOneClient
from jev_gemini.compiler import compile_primitive_prompt


class _SimulatedQueueClient:
  """Minimal fake `genai.Client` returning queued `GenerateContentResponse` objects."""

  def __init__(self, responses: list[Any]) -> None:
    self._responses = list(responses)
    self.models = self

  def generate_content(self, **kwargs: Any) -> Any:
    del kwargs
    return self._responses.pop(0)


def _build_simulated_client() -> GeminiSystemOneClient:
  """Builds an offline client returning the exact logprobs from `noul_01`, `choice_01`, `score_05`."""
  responses = [
      # 1. noul_01: P_cal(True)=0.9946 at T=2.0 -> logit_gap = 2 * ln(0.9946 / 0.0054) = 10.4362
      make_logprob_response(
          {"A": -0.00003, "B": -10.43623},
          include_gemini3_thought_wrapper=False,
      ),
      # 2. choice_01: P_cal(sre_incident)=0.9966 at T=2.0
      make_logprob_response(
          {"A": -0.000003, "C": -12.87, "B": -14.02, "D": -14.02},
          include_gemini3_thought_wrapper=False,
      ),
      # 3. score_05: P_cal(Level 0)=0.2459, P_cal(Level 1)=0.7541 at T=2.0
      make_logprob_response(
          {"B": -0.1120, "A": -2.3542, "C": -22.0, "D": -24.0, "E": -26.0},
          include_gemini3_thought_wrapper=False,
      ),
  ]
  return GeminiSystemOneClient(
      model="gemini-2.5-flash",
      mode="logprobs",
      default_temperature=2.0,
      genai_client=_SimulatedQueueClient(responses),
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run 1 benchmark example per primitive (noul_01, choice_01, score_05)."
  )
  parser.add_argument(
      "--live",
      action="store_true",
      help="Call live Vertex AI instead of offline deterministic fixtures.",
  )
  parser.add_argument(
      "--project",
      default="flex-hr-chatbot",
      help="GCP project ID for live Vertex AI mode.",
  )
  parser.add_argument(
      "--location",
      default="global",
      help="Vertex AI location (default: global).",
  )
  parser.add_argument(
      "--model",
      default="gemini-2.5-flash",
      help="Gemini model ID (default: gemini-2.5-flash).",
  )
  parser.add_argument(
      "--mode",
      choices=["logprobs", "monte_carlo"],
      default="logprobs",
      help="Probability estimation mode (default: logprobs).",
  )
  args = parser.parse_args()

  if args.live:
    client = GeminiSystemOneClient(
        model=args.model,
        project=args.project,
        location=args.location,
        mode=args.mode,
    )
  else:
    client = _build_simulated_client()

  # Select exactly one example per primitive from benchmark/dataset.py:
  noul_case = NOUL_BENCHMARK[0]    # noul_01
  choice_case = CHOICE_BENCHMARK[0]  # choice_01
  score_case = SCORE_BENCHMARK[4]    # score_05

  examples: list[tuple[str, Any, Any, Any]] = [
      ("Noul (noul_01)", noul_case.state, noul_case.question, noul_case.expected_value),
      ("Choice (choice_01)", choice_case.state, choice_case.question, choice_case.expected_choice),
      ("Score (score_05)", score_case.state, score_case.question, score_case.expected_level),
  ]

  for label, state, question, expected in examples:
    compiled = compile_primitive_prompt(state, question)
    answer = client.evaluate_one(state=state, question=question)
    print("=" * 80)
    print(f"PRIMITIVE EXAMPLE: {label} (Expected: {expected!r})")
    print("-" * 80)
    print("COMPILED PROMPT:")
    print(compiled.prompt)
    print("-" * 80)
    print("OUTPUT:")
    print(json.dumps(answer.to_dict(), indent=2))
    print()


if __name__ == "__main__":
  main()
