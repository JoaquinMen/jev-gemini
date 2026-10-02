"""Live Vertex AI demo script showing how to optimize a `Noul`, `Choice`, or `Score` prompt over N explorations.

Usage:
  # Run with default real Vertex AI parameters (flex-hr-chatbot, global, gemini-2.5-flash-lite, logprobs, noul, code_safety, 10 explorations):
  python3 example/optimize_prompt_demo.py

  # Or customize CLI arguments:
  python3 example/optimize_prompt_demo.py --project=flex-hr-chatbot --location=global --model=gemini-2.5-flash-lite --mode=logprobs --primitive=noul --category=code_safety --explorations=10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from benchmark.dataset import CHOICE_BENCHMARK
from benchmark.dataset import NOUL_BENCHMARK
from benchmark.dataset import SCORE_BENCHMARK
from benchmark.run_benchmark import VertexRESTClientShim
from jev_gemini import GeminiSystemOneClient
from jev_gemini import Noul
from jev_gemini import optimize_primitive
from jev_gemini._types_compat import HAS_GOOGLE_GENAI


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Optimize Noul, Choice, or Score wording live on Vertex AI over N explorations."
  )
  parser.add_argument(
      "--project",
      default="flex-hr-chatbot",
      help="GCP project ID for Vertex AI (default: flex-hr-chatbot).",
  )
  parser.add_argument(
      "--location",
      default="global",
      help="Vertex AI location (default: global).",
  )
  parser.add_argument(
      "--model",
      default="gemini-2.5-flash-lite",
      help="Evaluator model ID (default: gemini-2.5-flash-lite).",
  )
  parser.add_argument(
      "--mode",
      choices=["logprobs", "monte_carlo"],
      default="logprobs",
      help="Evaluation mode (default: logprobs).",
  )
  parser.add_argument(
      "--proposer-model",
      default="gemini-2.5-flash",
      help="Gemini model used to propose revised wordings (default: gemini-2.5-flash).",
  )
  parser.add_argument(
      "--primitive",
      choices=["noul", "choice", "score"],
      default="noul",
      help="Which primitive type to optimize (default: noul).",
  )
  parser.add_argument(
      "--category",
      default="code_safety",
      help="Category filter (default: code_safety).",
  )
  parser.add_argument(
      "--explorations",
      type=int,
      default=10,
      help="Number of candidate wording explorations (default: 10).",
  )
  args = parser.parse_args()

  genai_client = (
      None
      if HAS_GOOGLE_GENAI
      else VertexRESTClientShim(args.project, args.location)
  )
  client = GeminiSystemOneClient(
      model=args.model,
      project=args.project,
      location=args.location,
      mode=args.mode,
      genai_client=genai_client,
  )

  if args.primitive == "noul":
    dataset = NOUL_BENCHMARK
    category = args.category
    primitive = (
        Noul(question="Is this database query vulnerable to SQL injection?")
        if category == "code_safety"
        else NOUL_BENCHMARK[0].question
    )
  elif args.primitive == "choice":
    dataset = CHOICE_BENCHMARK
    category = args.category or "support_routing"
    primitive = CHOICE_BENCHMARK[0].question
  else:
    dataset = SCORE_BENCHMARK
    category = args.category or "incident_severity"
    primitive = SCORE_BENCHMARK[0].question

  result = optimize_primitive(
      client=client,
      primitive=primitive,
      dataset=dataset,
      category=category,
      num_explorations=args.explorations,
      proposer_model=args.proposer_model,
      verbose=True,
  )
  print("\n" + result.summary_markdown())


if __name__ == "__main__":
  main()

