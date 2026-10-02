"""GEPA-inspired Reflective Prompt & Criteria Wording Optimizer for `jev_gemini`.

Iteratively evolves the wording (`question` and option/criteria descriptions) of
any single `Noul`, `Choice`, or `Score` primitive over `num_explorations`
candidates guided by per-example probability error (`Brier` score) and accuracy
feedback—without requiring any modifications to existing client or compiler code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent import futures
import dataclasses
import json
import re
import time
from typing import Any, Literal

from jev_gemini._types_compat import types
from jev_gemini.client import GeminiSystemOneClient
from jev_gemini.primitives import Choice
from jev_gemini.primitives import Noul
from jev_gemini.primitives import Question
from jev_gemini.primitives import Score
from jev_gemini.responses import ChoiceAnswer
from jev_gemini.responses import NoulAnswer
from jev_gemini.responses import ScoreAnswer

PrimitiveType = Literal["noul", "choice", "score"]
OptimizationObjective = Literal["brier", "accuracy", "hybrid"]

EXPLORATION_STRATEGIES: tuple[tuple[str, str], ...] = (
    (
        "false_positive_exclusion",
        (
            "Focus on eliminating FALSE POSITIVES and overconfident wrong "
            "predictions. Add explicit 'Excludes:' or 'Do not infer X merely "
            "because Y is present' boundary clauses, and check the direction "
            "of relationships (e.g., parameterized queries, dry-run commands, "
            "sandbox vs production, or negated requests)."
        ),
    ),
    (
        "necessary_elements_checklist",
        (
            "Decompose the decision into an explicit checklist of necessary "
            "elements that must all be supported by the supplied STATE. "
            "Instruct the evaluator not to extrapolate unstated context."
        ),
    ),
    (
        "edge_case_disambiguation",
        (
            "Sharpen the boundary between adjacent or easily confused options "
            "(such as cross-domain keywords, multi-condition numeric/policy "
            "thresholds, or adjacent ordinal rubric levels)."
        ),
    ),
    (
        "concise_calibration_refinement",
        (
            "Distill the question and option/criteria descriptions into crisp, "
            "self-contained, mutually exclusive decision rules that maximize "
            "probability concentration on the true label."
        ),
    ),
)


@dataclasses.dataclass(frozen=True)
class OptimizationExample:
  """Normalized single labeled example for primitive wording optimization."""

  example_id: str
  state: Any
  expected: Any
  category: str = "default"
  item_question: Question | None = None


@dataclasses.dataclass(frozen=True)
class ExampleFeedback:
  """Per-example evaluation and error diagnostics for reflection."""

  example_id: str
  category: str
  state: Any
  expected: Any
  predicted: Any
  probability_of_expected: float
  confidence: float
  brier_error: float
  score: float
  correct: bool


@dataclasses.dataclass(frozen=True)
class CandidateRecord:
  """Evaluation record for one candidate primitive in the exploration lineage."""

  candidate_id: int
  parent_id: int | None
  strategy: str
  rationale: str
  primitive: Question
  mean_brier: float
  mean_score: float
  accuracy: float
  num_correct: int
  num_examples: int
  mean_confidence: float
  corrected_errors: int
  introduced_errors: int
  latency_s: float
  feedback: list[ExampleFeedback]

  def to_dict(self) -> dict[str, Any]:
    return {
        "candidate_id": self.candidate_id,
        "parent_id": self.parent_id,
        "strategy": self.strategy,
        "rationale": self.rationale,
        "primitive": dataclasses.asdict(self.primitive),
        "mean_brier": round(self.mean_brier, 6),
        "mean_score": round(self.mean_score, 6),
        "accuracy": round(self.accuracy, 4),
        "num_correct": self.num_correct,
        "num_examples": self.num_examples,
        "mean_confidence": round(self.mean_confidence, 4),
        "corrected_errors": self.corrected_errors,
        "introduced_errors": self.introduced_errors,
        "latency_s": round(self.latency_s, 3),
    }


@dataclasses.dataclass(frozen=True)
class PromptOptimizationResult:
  """Complete output of `PromptOptimizer.optimize` / `optimize_primitive`."""

  primitive_type: PrimitiveType
  original_primitive: Question
  best_primitive: Question
  best_candidate_id: int
  original_record: CandidateRecord
  best_record: CandidateRecord
  history: list[CandidateRecord]
  total_explorations: int
  total_time_s: float

  @property
  def brier_improvement(self) -> float:
    """Absolute reduction in Brier score (`original_brier - best_brier`)."""
    return self.original_record.mean_brier - self.best_record.mean_brier

  @property
  def accuracy_improvement(self) -> float:
    """Absolute gain in accuracy (`best_accuracy - original_accuracy`)."""
    return self.best_record.accuracy - self.original_record.accuracy

  def to_dict(self) -> dict[str, Any]:
    return {
        "primitive_type": self.primitive_type,
        "best_candidate_id": self.best_candidate_id,
        "brier_improvement": round(self.brier_improvement, 6),
        "accuracy_improvement": round(self.accuracy_improvement, 4),
        "total_explorations": self.total_explorations,
        "total_time_s": round(self.total_time_s, 3),
        "original_primitive": dataclasses.asdict(self.original_primitive),
        "best_primitive": dataclasses.asdict(self.best_primitive),
        "history": [rec.to_dict() for rec in self.history],
    }

  def summary_markdown(self) -> str:
    """Renders a Markdown report with the exploration lineage and prompt diff."""
    lines = [
        f"## Prompt Optimization Report (`{self.primitive_type.upper()}`)",
        "",
        f"- **Explorations Evaluated**: `{self.total_explorations}` (+ Baseline Candidate `0`)",
        f"- **Selected Winner**: Candidate `{self.best_candidate_id}` (Parent `{self.best_record.parent_id}`)",
        (
            f"- **Accuracy**: `{self.original_record.accuracy * 100:.1f}%` "
            f"({self.original_record.num_correct}/{self.original_record.num_examples}) -> "
            f"**`{self.best_record.accuracy * 100:.1f}%`** "
            f"({self.best_record.num_correct}/{self.best_record.num_examples})"
        ),
        (
            f"- **Brier Error (↓ better)**: `{self.original_record.mean_brier:.4f}` -> "
            f"**`{self.best_record.mean_brier:.4f}`** "
            f"(`{self.brier_improvement:+.4f}` reduction)"
        ),
        (
            f"- **Error Transitions vs. Baseline**: `{self.best_record.corrected_errors}` "
            f"corrected, `{self.best_record.introduced_errors}` introduced"
        ),
        "",
        "### 1. Exploration Lineage Table",
        "",
        "| Candidate | Parent | Strategy | Brier Error (↓) | GEPA Score `1-Brier` (↑) | Accuracy (↑) | Corrected / Introduced |",
        "| :---: | :---: | :--- | :---: | :---: | :---: | :---: |",
    ]
    for rec in self.history:
      marker = " **★**" if rec.candidate_id == self.best_candidate_id else ""
      parent_str = "None" if rec.parent_id is None else str(rec.parent_id)
      lines.append(
          f"| `{rec.candidate_id}`{marker} | `{parent_str}` | `{rec.strategy}` | "
          f"`{rec.mean_brier:.4f}` | `{rec.mean_score:.4f}` | "
          f"`{rec.accuracy * 100:.1f}%` ({rec.num_correct}/{rec.num_examples}) | "
          f"`+{rec.corrected_errors} / -{rec.introduced_errors}` |"
      )

    lines.extend([
        "",
        "### 2. Original vs. Optimized Primitive Wording",
        "",
        "#### Original (`Candidate 0`)",
        "```json",
        json.dumps(dataclasses.asdict(self.original_primitive), indent=2),
        "```",
        "",
        f"#### Optimized (`Candidate {self.best_candidate_id}` — {self.best_record.rationale})",
        "```json",
        json.dumps(dataclasses.asdict(self.best_primitive), indent=2),
        "```",
    ])
    return "\n".join(lines)


def detect_primitive_type(primitive: Question) -> PrimitiveType:
  """Automatically detects whether a primitive is `noul`, `choice`, or `score`."""
  if isinstance(primitive, Noul):
    return "noul"
  if isinstance(primitive, Choice):
    return "choice"
  if isinstance(primitive, Score):
    return "score"
  raise TypeError(
      f"Expected Noul, Choice, or Score instance, got {type(primitive).__name__}."
  )


def normalize_optimization_dataset(
    dataset: Sequence[Any],
    *,
    category: str | None = None,
) -> list[OptimizationExample]:
  """Normalizes benchmark test cases, tuples, or dicts into `OptimizationExample`s."""
  normalized: list[OptimizationExample] = []
  for idx, item in enumerate(dataset):
    if isinstance(item, OptimizationExample):
      if category is not None and item.category != category:
        continue
      normalized.append(item)
    elif hasattr(item, "state") and hasattr(item, "question"):
      item_cat = getattr(item, "category", "default")
      if category is not None and item_cat != category:
        continue
      if hasattr(item, "expected_value"):
        expected = item.expected_value
      elif hasattr(item, "expected_choice"):
        expected = item.expected_choice
      elif hasattr(item, "expected_level"):
        expected = item.expected_level
      else:
        raise ValueError(f"Unrecognized benchmark test item at index {idx}: {item!r}")
      test_id = getattr(item, "test_id", f"ex_{idx + 1:02d}")
      normalized.append(
          OptimizationExample(
              example_id=str(test_id),
              state=item.state,
              expected=expected,
              category=str(item_cat),
              item_question=item.question,
          )
      )
    elif isinstance(item, tuple) and len(item) >= 2:
      normalized.append(
          OptimizationExample(
              example_id=f"ex_{idx + 1:02d}",
              state=item[0],
              expected=item[1],
          )
      )
    elif isinstance(item, Mapping) and "state" in item and "expected" in item:
      item_cat = str(item.get("category", "default"))
      if category is not None and item_cat != category:
        continue
      normalized.append(
          OptimizationExample(
              example_id=str(item.get("test_id", f"ex_{idx + 1:02d}")),
              state=item["state"],
              expected=item["expected"],
              category=item_cat,
          )
      )
    else:
      raise TypeError(
          f"Unsupported dataset item at index {idx}: expected NoulTest/ChoiceTest/"
          f"ScoreTest, (state, expected) tuple, or dict, got {type(item).__name__}."
      )
  if not normalized:
    raise ValueError("Optimization dataset is empty after filtering.")
  return normalized


def _extract_json_object(raw_text: str) -> dict[str, Any]:
  """Parses a JSON object from an LLM response, stripping optional markdown fences."""
  cleaned = raw_text.strip()
  if cleaned.startswith("```"):
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
  start = cleaned.find("{")
  end = cleaned.rfind("}")
  if start != -1 and end != -1 and end >= start:
    cleaned = cleaned[start : end + 1]
  parsed = json.loads(cleaned)
  if not isinstance(parsed, dict):
    raise ValueError(f"Expected JSON object from proposer, got {type(parsed).__name__}.")
  return parsed


def _apply_candidate_to_example(
    candidate: Question,
    example: OptimizationExample,
    *,
    has_heterogeneous_questions: bool,
) -> Question:
  """Combines the candidate wording with an example (preserving per-item tasks if heterogeneous)."""
  if not has_heterogeneous_questions or example.item_question is None:
    return candidate

  # When a user passes a multi-task benchmark suite (e.g., all 40 NOUL_BENCHMARK items
  # which ask 10 different questions), we preserve each item's specific question core
  # while applying the candidate's evolved rubric guidelines and option descriptions!
  item_q = example.item_question
  if isinstance(candidate, Noul) and isinstance(item_q, Noul):
    guideline_suffix = ""
    if candidate.question.strip() != item_q.question.strip():
      guideline_suffix = f"\nEvaluation Guideline: {candidate.question.strip()}"
    return dataclasses.replace(
        candidate,
        question=f"{item_q.question.strip()}{guideline_suffix}",
    )
  if isinstance(candidate, Choice) and isinstance(item_q, Choice):
    if set(candidate.options.keys()) == set(item_q.options.keys()):
      return candidate
    guideline_suffix = f"\nRouting Guideline: {candidate.question.strip()}"
    return dataclasses.replace(
        item_q,
        question=f"{item_q.question.strip()}{guideline_suffix}",
    )
  if isinstance(candidate, Score) and isinstance(item_q, Score):
    if item_q.question.strip() == candidate.question.strip():
      return candidate
    return dataclasses.replace(
        item_q,
        criteria=candidate.criteria
        if len(candidate.criteria) == len(item_q.criteria)
        else item_q.criteria,
    )
  return candidate


class PromptOptimizer:
  """Iterative GEPA-style Prompt & Criteria Wording Optimizer for `jev_gemini`.

  Automatically detects the primitive type (`Noul`, `Choice`, or `Score`),
  evaluates candidate wordings against a labeled dataset, inspects the highest
  probability-error examples (`Brier` error), and proposes improved question
  and option/criteria wording across `num_explorations` iterations.
  """

  def __init__(
      self,
      client: GeminiSystemOneClient,
      *,
      proposer_model: str = "gemini-2.5-flash",
      proposer_fn: Callable[[str, PrimitiveType, Question], dict[str, Any]] | None = None,
      max_feedback_examples: int = 8,
      concurrency: int = 4,
      objective: OptimizationObjective = "hybrid",
  ) -> None:
    """Initializes `PromptOptimizer`.

    Args:
      client: `GeminiSystemOneClient` used to evaluate each candidate primitive.
      proposer_model: Vertex AI Gemini model ID used to reflect on errors and
        propose improved question/criteria wording (default `"gemini-2.5-flash"`).
      proposer_fn: Optional custom callable `(reflection_prompt, primitive_type,
        parent_primitive) -> dict` for deterministic unit testing or custom LLMs.
      max_feedback_examples: Maximum number of high-error examples included in
        each reflection prompt (`default = 8`).
      concurrency: Number of parallel evaluation workers (`default = 4`).
      objective: Selection criterion (`"brier"`, `"accuracy"`, or `"hybrid"`
        which maximizes accuracy first and breaks ties via lowest Brier score,
        or vice-versa).
    """
    self._client = client
    self._proposer_model = proposer_model
    self._proposer_fn = proposer_fn
    self._max_feedback_examples = max(1, max_feedback_examples)
    self._concurrency = max(1, concurrency)
    self._objective: OptimizationObjective = objective

  def _evaluate_single_example(
      self,
      candidate: Question,
      example: OptimizationExample,
      *,
      has_heterogeneous_questions: bool,
  ) -> ExampleFeedback:
    effective_q = _apply_candidate_to_example(
        candidate,
        example,
        has_heterogeneous_questions=has_heterogeneous_questions,
    )
    ans = self._client.evaluate_one(state=example.state, question=effective_q)

    if isinstance(ans, NoulAnswer):
      expected_bool = bool(example.expected)
      y = 1.0 if expected_bool else 0.0
      brier = (ans.probability - y) ** 2
      prob_expected = ans.probability if expected_bool else (1.0 - ans.probability)
      correct = ans.value == expected_bool
      return ExampleFeedback(
          example_id=example.example_id,
          category=example.category,
          state=example.state,
          expected=expected_bool,
          predicted=ans.value,
          probability_of_expected=round(prob_expected, 4),
          confidence=round(ans.confidence, 4),
          brier_error=round(brier, 6),
          score=round(1.0 - brier, 6),
          correct=correct,
      )

    if isinstance(ans, ChoiceAnswer):
      expected_choice = str(example.expected)
      k_count = max(1, len(ans.probabilities))
      brier = (
          sum(
              (prob - (1.0 if opt == expected_choice else 0.0)) ** 2
              for opt, prob in ans.probabilities.items()
          )
          / k_count
      )
      prob_expected = ans.probabilities.get(expected_choice, 0.0)
      correct = ans.choice == expected_choice
      return ExampleFeedback(
          example_id=example.example_id,
          category=example.category,
          state=example.state,
          expected=expected_choice,
          predicted=ans.choice,
          probability_of_expected=round(prob_expected, 4),
          confidence=round(ans.confidence, 4),
          brier_error=round(brier, 6),
          score=round(1.0 - brier, 6),
          correct=correct,
      )

    if isinstance(ans, ScoreAnswer):
      expected_level = int(example.expected)
      nearest_level = int(round(ans.score))
      span = max(1.0, ans.level_values[-1] - ans.level_values[0])
      norm_diff = (ans.score - float(expected_level)) / span
      brier = norm_diff**2
      prob_expected = (
          ans.distribution[expected_level]
          if 0 <= expected_level < len(ans.distribution)
          else 0.0
      )
      correct = nearest_level == expected_level
      return ExampleFeedback(
          example_id=example.example_id,
          category=example.category,
          state=example.state,
          expected=expected_level,
          predicted=round(ans.score, 4),
          probability_of_expected=round(prob_expected, 4),
          confidence=round(ans.confidence, 4),
          brier_error=round(brier, 6),
          score=round(1.0 - brier, 6),
          correct=correct,
      )

    raise TypeError(f"Unsupported answer type: {type(ans).__name__}")

  def evaluate_candidate(
      self,
      candidate: Question,
      examples: Sequence[OptimizationExample],
      *,
      candidate_id: int = 0,
      parent_id: int | None = None,
      strategy: str = "baseline",
      rationale: str = "Original baseline wording",
      baseline_feedback: Sequence[ExampleFeedback] | None = None,
      has_heterogeneous_questions: bool = False,
  ) -> CandidateRecord:
    """Evaluates a candidate primitive across `examples` and computes GEPA metrics."""
    t_start = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=self._concurrency) as pool:
      feedback_list = list(
          pool.map(
              lambda ex: self._evaluate_single_example(
                  candidate,
                  ex,
                  has_heterogeneous_questions=has_heterogeneous_questions,
              ),
              examples,
          )
      )
    elapsed_s = time.perf_counter() - t_start

    n = len(feedback_list)
    num_correct = sum(1 for f in feedback_list if f.correct)
    mean_brier = sum(f.brier_error for f in feedback_list) / n
    mean_score = 1.0 - mean_brier
    mean_conf = sum(f.confidence for f in feedback_list) / n

    corrected = 0
    introduced = 0
    if baseline_feedback is not None and len(baseline_feedback) == n:
      for base_f, cur_f in zip(baseline_feedback, feedback_list, strict=True):
        if not base_f.correct and cur_f.correct:
          corrected += 1
        elif base_f.correct and not cur_f.correct:
          introduced += 1

    return CandidateRecord(
        candidate_id=candidate_id,
        parent_id=parent_id,
        strategy=strategy,
        rationale=rationale,
        primitive=candidate,
        mean_brier=mean_brier,
        mean_score=mean_score,
        accuracy=num_correct / n,
        num_correct=num_correct,
        num_examples=n,
        mean_confidence=mean_conf,
        corrected_errors=corrected,
        introduced_errors=introduced,
        latency_s=elapsed_s,
        feedback=feedback_list,
    )

  def _build_reflection_prompt(
      self,
      primitive_type: PrimitiveType,
      parent_record: CandidateRecord,
      strategy_name: str,
      strategy_instruction: str,
      exploration_idx: int,
  ) -> str:
    """Builds the GEPA reflective prompt containing the hardest feedback examples."""
    sorted_fb = sorted(
        parent_record.feedback,
        key=lambda f: (not f.correct, f.brier_error),
        reverse=True,
    )
    top_fb = sorted_fb[: self._max_feedback_examples]
    fb_records = []
    for f in top_fb:
      status = (
          "WRONG_PREDICTION"
          if not f.correct
          else ("LOW_CONFIDENCE_CORRECT" if f.brier_error > 0.005 else "CALIBRATED_CORRECT")
      )
      fb_records.append({
          "example_id": f.example_id,
          "category": f.category,
          "status": status,
          "state": f.state,
          "expected": f.expected,
          "predicted": f.predicted,
          "probability_of_expected": f.probability_of_expected,
          "brier_error": f.brier_error,
      })

    parent_p = parent_record.primitive
    if isinstance(parent_p, Noul):
      current_spec = {
          "question": parent_p.question,
          "true_description": parent_p.true_description,
          "false_description": parent_p.false_description,
      }
      schema_instructions = (
          "Return a JSON object with keys:\n"
          '  "rationale": (string) 1-sentence summary of what boundary/rule you improved,\n'
          '  "question": (string) improved binary proposition wording,\n'
          '  "true_description": (string) explicit criteria when option [A] (True) applies,\n'
          '  "false_description": (string) explicit criteria and exclusions when option [B] (False) applies.'
      )
    elif isinstance(parent_p, Choice):
      current_spec = {
          "question": parent_p.question,
          "options": dict(parent_p.options),
      }
      option_keys = list(parent_p.options.keys())
      schema_instructions = (
          "Return a JSON object with keys:\n"
          '  "rationale": (string) 1-sentence summary of what boundary/rule you improved,\n'
          '  "question": (string) improved routing/classification question,\n'
          f'  "options": (object) mapping the EXACT option keys {option_keys!r} to '
          "clearer, mutually exclusive descriptions (with explicit inclusions and exclusions)."
      )
    elif isinstance(parent_p, Score):
      current_spec = {
          "question": parent_p.question,
          "criteria": list(parent_p.criteria),
      }
      num_levels = len(parent_p.criteria)
      schema_instructions = (
          "Return a JSON object with keys:\n"
          '  "rationale": (string) 1-sentence summary of what boundary/rule you improved,\n'
          '  "question": (string) improved ordinal scoring question,\n'
          f'  "criteria": (array of {num_levels} strings) improved level descriptions '
          f"from Level 0 to Level {num_levels - 1} with crisp boundary rules between adjacent levels."
      )
    else:
      raise TypeError(f"Unsupported primitive: {type(parent_p).__name__}")

    return (
        "You are a GEPA (Genetic-Pareto) Reflective Prompt Optimizer for a "
        "System-One single-token classifier.\n"
        f"Exploration Step: #{exploration_idx}\n"
        f"Primitive Type: {primitive_type.upper()}\n"
        f"Optimization Strategy ({strategy_name}): {strategy_instruction}\n\n"
        "=== CURRENT PRIMITIVE WORDING ===\n"
        f"{json.dumps(current_spec, indent=2)}\n\n"
        "=== CURRENT PERFORMANCE ===\n"
        f"Accuracy: {parent_record.accuracy * 100:.1f}% "
        f"({parent_record.num_correct}/{parent_record.num_examples}), "
        f"Mean Brier Error: {parent_record.mean_brier:.4f} (lower is better)\n\n"
        "=== HIGHEST-ERROR & BOUNDARY FEEDBACK EXAMPLES ===\n"
        f"{json.dumps(fb_records, indent=2)}\n\n"
        "=== INSTRUCTIONS ===\n"
        "1. Inspect the `WRONG_PREDICTION` and `LOW_CONFIDENCE_CORRECT` examples above.\n"
        "2. Revise the wording so the classifier evaluates ONLY the supplied STATE, "
        "checks the exact direction of relationships, and avoids surface-keyword traps "
        "(without overfitting to specific example IDs or names).\n"
        f"3. {schema_instructions}\n"
        "Output ONLY valid JSON."
    )

  def _build_proposer_thinking_config(self) -> Any:
    model_lower = self._proposer_model.lower()
    if any(t in model_lower for t in ("gemini-3.7", "gemini-3.8", "gemini-4")):
      return types.ThinkingConfig(thinking_level="LOW")
    if "gemini-3" in model_lower:
      return types.ThinkingConfig(thinking_level="MINIMAL")
    return types.ThinkingConfig(thinking_budget=0)

  def _call_proposer(
      self,
      reflection_prompt: str,
      primitive_type: PrimitiveType,
      parent_primitive: Question,
  ) -> dict[str, Any]:
    if self._proposer_fn is not None:
      return self._proposer_fn(reflection_prompt, primitive_type, parent_primitive)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.7,
        max_output_tokens=4096,
        thinking_config=self._build_proposer_thinking_config(),
    )
    response = self._client.genai_client.models.generate_content(
        model=self._proposer_model,
        contents=reflection_prompt,
        config=config,
    )
    raw_text = ""
    if getattr(response, "candidates", None):
      parts = getattr(response.candidates[0].content, "parts", None) or []
      raw_text = "".join(
          getattr(p, "text", "") or ""
          for p in parts
          if not getattr(p, "thought", False)
      )
    if not raw_text:
      raw_text = getattr(response, "text", None) or ""
    return _extract_json_object(raw_text)

  def _construct_mutated_primitive(
      self,
      parent: Question,
      proposal: Mapping[str, Any],
  ) -> tuple[Question, str]:
    rationale = str(proposal.get("rationale", "Refined question and criteria wording.")).strip()

    if isinstance(parent, Noul):
      new_q = str(proposal.get("question", parent.question)).strip() or parent.question
      new_true = (
          str(proposal.get("true_description", parent.true_description)).strip()
          or parent.true_description
      )
      new_false = (
          str(proposal.get("false_description", parent.false_description)).strip()
          or parent.false_description
      )
      return (
          dataclasses.replace(
              parent,
              question=new_q,
              true_description=new_true,
              false_description=new_false,
          ),
          rationale,
      )

    if isinstance(parent, Choice):
      new_q = str(proposal.get("question", parent.question)).strip() or parent.question
      raw_opts = proposal.get("options")
      new_opts: dict[str, Any] = dict(parent.options)
      if isinstance(raw_opts, Mapping):
        for k in parent.options:
          if k in raw_opts and str(raw_opts[k]).strip():
            new_opts[k] = str(raw_opts[k]).strip()
      return (
          dataclasses.replace(parent, question=new_q, options=new_opts),
          rationale,
      )

    if isinstance(parent, Score):
      new_q = str(proposal.get("question", parent.question)).strip() or parent.question
      raw_crit = proposal.get("criteria")
      new_crit = list(parent.criteria)
      if isinstance(raw_crit, Sequence) and not isinstance(raw_crit, (str, bytes)):
        if len(raw_crit) == len(parent.criteria):
          new_crit = [
              str(c).strip() or str(orig)
              for c, orig in zip(raw_crit, parent.criteria, strict=True)
          ]
      return (
          dataclasses.replace(parent, question=new_q, criteria=new_crit),
          rationale,
      )

    raise TypeError(f"Unsupported primitive: {type(parent).__name__}")

  def _rank_key(self, record: CandidateRecord) -> tuple[float, float, int]:
    """Returns a tuple where larger is better according to `self._objective`."""
    if self._objective == "brier":
      return (record.mean_score, record.accuracy, -record.candidate_id)
    if self._objective == "accuracy":
      return (record.accuracy, record.mean_score, -record.candidate_id)
    # "hybrid": prioritize higher accuracy AND lower Brier score via GEPA score = 1 - Brier
    return (record.accuracy, record.mean_score, -record.candidate_id)

  def _select_parent(
      self,
      history: Sequence[CandidateRecord],
      exploration_idx: int,
  ) -> CandidateRecord:
    """Selects a parent from the top Pareto archive candidates for branching."""
    ranked = sorted(history, key=self._rank_key, reverse=True)
    # Branch across the top 2 elite candidates when available (like GEPA Table 5)
    pool_size = min(2, len(ranked))
    parent_idx = (exploration_idx - 1) % pool_size
    return ranked[parent_idx]

  def optimize(
      self,
      primitive: Question,
      dataset: Sequence[Any],
      *,
      num_explorations: int = 10,
      category: str | None = None,
      verbose: bool = False,
  ) -> PromptOptimizationResult:
    """Iteratively explores `num_explorations` wording revisions for `primitive`.

    Args:
      primitive: Starting `Noul`, `Choice`, or `Score` instance.
      dataset: Sequence of benchmark test objects (`NoulTest`, `ChoiceTest`,
        `ScoreTest`), `(state, expected)` tuples, or dicts.
      num_explorations: Number of candidate wording explorations to run (`>= 1`).
      category: Optional category filter when passing a multi-category dataset.
      verbose: If True, prints progress after each exploration step.

    Returns:
      `PromptOptimizationResult` containing `best_primitive`, `history`, and
      `summary_markdown()`.
    """
    if num_explorations < 1:
      raise ValueError(f"num_explorations must be >= 1, got {num_explorations}.")

    t_total_start = time.perf_counter()
    p_type = detect_primitive_type(primitive)
    examples = normalize_optimization_dataset(dataset, category=category)

    # Detect whether dataset items carry heterogeneous per-item questions
    distinct_item_questions = {
        ex.item_question.question.strip()
        for ex in examples
        if ex.item_question is not None
    }
    has_heterogeneous_questions = len(distinct_item_questions) > 1

    # Step 0: Evaluate original primitive (Candidate 0)
    baseline_record = self.evaluate_candidate(
        primitive,
        examples,
        candidate_id=0,
        parent_id=None,
        strategy="baseline",
        rationale="Original baseline primitive wording",
        has_heterogeneous_questions=has_heterogeneous_questions,
    )
    history: list[CandidateRecord] = [baseline_record]

    if verbose:
      print(
          f"[Candidate 0 | Baseline] Acc={baseline_record.accuracy * 100:.1f}% "
          f"({baseline_record.num_correct}/{baseline_record.num_examples}), "
          f"Brier={baseline_record.mean_brier:.4f}"
      )

    # Steps 1..num_explorations: Reflective Evolution Loop
    for step_idx in range(1, num_explorations + 1):
      parent_record = self._select_parent(history, step_idx)
      strategy_name, strategy_instruction = EXPLORATION_STRATEGIES[
          (step_idx - 1) % len(EXPLORATION_STRATEGIES)
      ]
      reflection_prompt = self._build_reflection_prompt(
          primitive_type=p_type,
          parent_record=parent_record,
          strategy_name=strategy_name,
          strategy_instruction=strategy_instruction,
          exploration_idx=step_idx,
      )

      try:
        proposal = self._call_proposer(
            reflection_prompt, p_type, parent_record.primitive
        )
        candidate_primitive, rationale = self._construct_mutated_primitive(
            parent_record.primitive, proposal
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        candidate_primitive = parent_record.primitive
        rationale = f"Fallback to parent wording ({type(exc).__name__})"

      cand_record = self.evaluate_candidate(
          candidate_primitive,
          examples,
          candidate_id=step_idx,
          parent_id=parent_record.candidate_id,
          strategy=strategy_name,
          rationale=rationale,
          baseline_feedback=baseline_record.feedback,
          has_heterogeneous_questions=has_heterogeneous_questions,
      )
      history.append(cand_record)

      if verbose:
        print(
            f"[Candidate {step_idx} | Parent {parent_record.candidate_id} | "
            f"{strategy_name}] Acc={cand_record.accuracy * 100:.1f}% "
            f"({cand_record.num_correct}/{cand_record.num_examples}), "
            f"Brier={cand_record.mean_brier:.4f} | {rationale}"
        )

    best_record = max(history, key=self._rank_key)
    total_time_s = time.perf_counter() - t_total_start

    return PromptOptimizationResult(
        primitive_type=p_type,
        original_primitive=primitive,
        best_primitive=best_record.primitive,
        best_candidate_id=best_record.candidate_id,
        original_record=baseline_record,
        best_record=best_record,
        history=history,
        total_explorations=num_explorations,
        total_time_s=total_time_s,
    )


def optimize_primitive(
    client: GeminiSystemOneClient,
    primitive: Question,
    dataset: Sequence[Any],
    *,
    num_explorations: int = 10,
    proposer_model: str = "gemini-2.5-flash",
    proposer_fn: Callable[[str, PrimitiveType, Question], dict[str, Any]] | None = None,
    category: str | None = None,
    max_feedback_examples: int = 8,
    concurrency: int = 4,
    objective: OptimizationObjective = "hybrid",
    verbose: bool = False,
) -> PromptOptimizationResult:
  """Convenience entrypoint to optimize a `Noul`, `Choice`, or `Score` primitive."""
  optimizer = PromptOptimizer(
      client=client,
      proposer_model=proposer_model,
      proposer_fn=proposer_fn,
      max_feedback_examples=max_feedback_examples,
      concurrency=concurrency,
      objective=objective,
  )
  return optimizer.optimize(
      primitive=primitive,
      dataset=dataset,
      num_explorations=num_explorations,
      category=category,
      verbose=verbose,
  )
