"""Response data contracts for `jev_gemini` (`NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`)."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal


@dataclasses.dataclass(frozen=True)
class StepDiagnostics:
  """Low-level logprob extraction telemetry for observability and debugging.

  Attributes:
    selected_step_index: Index `t*` in `LogprobsResult.top_candidates` from
      which the enum distribution was extracted.
    total_decoding_steps: Total number of decoding steps in `LogprobsResult`.
    thought_tokens_stripped: Number of leading thought/control tokens (`<ctrl94>`
      .. `<ctrl95>`) skipped prior to `selected_step_index`.
    matched_enum_mass: Sum of linear probabilities `sum_{c in V} exp(logprob)`
      matched to valid bijected enum letters at `selected_step_index`.
    unobserved_options: User option keys whose bijected letter did not appear in
      `top_candidates[selected_step_index]` (assigned `0.0` probability).
    calibration_temperature: Post-hoc temperature `T` applied to the distribution.
    echo_leakage_warning: Warning message if a state key had high lexical
      overlap with a `Noul` proposition.
    jensen_shannon_divergence: Jensen-Shannon divergence in `[0, 1]` between
      forward and reversed option passes when `permute_debias=True`.
    is_degraded_fallback: True if `logprobs_result` was missing and the client
      had to fall back to `response.text`.
  """

  selected_step_index: int
  total_decoding_steps: int
  thought_tokens_stripped: int
  matched_enum_mass: float
  unobserved_options: list[str] = dataclasses.field(default_factory=list)
  calibration_temperature: float = 1.0
  echo_leakage_warning: str | None = None
  jensen_shannon_divergence: float | None = None
  is_degraded_fallback: bool = False

  def to_dict(self) -> dict[str, Any]:
    """Serializes diagnostics to a plain dictionary."""
    return dataclasses.asdict(self)

  def model_dump(self) -> dict[str, Any]:
    """Pydantic v2-compatible serialization alias."""
    return self.to_dict()


@dataclasses.dataclass(frozen=True)
class NoulAnswer:
  """Calibrated response for a `Noul` binary proposition.

  Attributes:
    probability: Calibrated `P(True)` in `[0.0, 1.0]`.
    value: Resolved boolean (`True` or `False`), or `None` if `probability`
      falls inside `Noul.abstain_band`.
    confidence: Peak binary probability `max(p, 1 - p)` in `[0.5, 1.0]`.
    normalized_confidence: Chance-adjusted binary margin `|2p - 1|` in `[0.0,
      1.0]`.
    entropy: Normalized binary Shannon entropy `H_2(p)` in `[0.0, 1.0]`.
    logit_gap: Raw log-probability difference `ln P_raw(True) - ln P_raw(False)`.
    diagnostics: Extraction and calibration diagnostics.
    type: Discriminator literal `"noul"`.
  """

  probability: float
  value: bool | None
  confidence: float
  normalized_confidence: float
  entropy: float
  logit_gap: float
  diagnostics: StepDiagnostics
  type: Literal["noul"] = "noul"

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  def model_dump(self) -> dict[str, Any]:
    return self.to_dict()


@dataclasses.dataclass(frozen=True)
class ChoiceAnswer:
  """Calibrated response for a `Choice` categorical question.

  Attributes:
    choice: Winning option key (`argmax` of calibrated probabilities).
    confidence: Peak calibrated probability `max_i p_i` in `[1/K, 1.0]`.
    normalized_confidence: Chance-adjusted peak margin `(K * p_max - 1) / (K -
      1)` in `[0.0, 1.0]`.
    margin: Probability gap `p_(1) - p_(2)` between top-1 and top-2 options.
    certainty: Normalized inverse Shannon entropy `1 - H(p) / ln(K)` in `[0.0,
      1.0]`.
    probabilities: Calibrated probability distribution keyed by user option
      names, sorted from highest to lowest probability.
    raw_probabilities: Pre-temperature-scaling normalized probabilities keyed by
      user option names.
    diagnostics: Extraction and calibration diagnostics.
    type: Discriminator literal `"choice"`.
  """

  choice: str
  confidence: float
  normalized_confidence: float
  margin: float
  certainty: float
  probabilities: dict[str, float]
  raw_probabilities: dict[str, float]
  diagnostics: StepDiagnostics
  type: Literal["choice"] = "choice"

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  def model_dump(self) -> dict[str, Any]:
    return self.to_dict()


@dataclasses.dataclass(frozen=True)
class ScoreAnswer:
  """Calibrated response for a `Score` ordinal rubric question.

  Attributes:
    score: Continuous probability-weighted expected value `sum_i v_i * p_i` in
      `[v_0, v_{L-1}]`.
    normalized_score: Expected value rescaled to `[0.0, 1.0]`.
    confidence: Popoviciu ordinal dispersion confidence `1 - 2 * std_dev /
      (v_{L-1} - v_0)` in `[0.0, 1.0]`.
    std_dev: Ordinal standard deviation `sqrt(sum_i p_i * (v_i - score)^2)`.
    entropy: Normalized Shannon entropy `H(p) / ln(L)` in `[0.0, 1.0]`.
    is_bimodal: True if probability mass splits across non-adjacent rubric
      levels (`|i - j| >= 2` with both `p_i >= 0.20` and `p_j >= 0.20` separated
      by a lower-probability valley).
    mode_level: Index `0 .. L-1` of the single highest-probability rubric level.
    distribution: Calibrated probabilities `[p_0, ..., p_{L-1}]` in rubric
      order.
    level_values: Numeric values `[v_0, ..., v_{L-1}]` corresponding to each
      rubric level.
    diagnostics: Extraction and calibration diagnostics.
    type: Discriminator literal `"score"`.
  """

  score: float
  normalized_score: float
  confidence: float
  std_dev: float
  entropy: float
  is_bimodal: bool
  mode_level: int
  distribution: list[float]
  level_values: list[float]
  diagnostics: StepDiagnostics
  type: Literal["score"] = "score"

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  def model_dump(self) -> dict[str, Any]:
    return self.to_dict()


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclasses.dataclass(frozen=True)
class SystemOneResponse:
  """Container returned by `GeminiSystemOneClient.ask` and `ask_async`.

  Attributes:
    answers: Mapping from question key to its calibrated `Answer`
      (`NoulAnswer`, `ChoiceAnswer`, or `ScoreAnswer`).
    latency_ms: End-to-end wall-clock evaluation latency in milliseconds.
    model: Gemini model identifier used for evaluation.
  """

  answers: dict[str, Answer]
  latency_ms: float
  model: str

  def __getitem__(self, key: str) -> Answer:
    return self.answers[key]

  def __contains__(self, key: object) -> bool:
    return key in self.answers

  def keys(self):
    return self.answers.keys()

  def items(self):
    return self.answers.items()

  def values(self):
    return self.answers.values()

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  def model_dump(self) -> dict[str, Any]:
    return self.to_dict()
