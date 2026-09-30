"""Jev System-One question primitives (`Noul`, `Choice`, `Score`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import enum
from typing import Any, Literal

JSONContent = (
    str | int | float | bool | None | Mapping[str, Any] | Sequence[Any]
)


@dataclasses.dataclass(frozen=True)
class Noul:
  """Binary proposition evaluated against state (`True` / `False` / Abstain).

  Attributes:
    question: Binary yes/no proposition to evaluate against the current state.
    true_description: Explicit rubric description for the `True` ('A') outcome.
    false_description: Explicit rubric description for the `False` ('B')
      outcome.
    abstain_band: Optional `(low, high)` probability interval in `[0, 1]` where
      `NoulAnswer.value` resolves to `None` (abstain / escalate to System Two).
    temperature: Optional per-question calibration temperature override.
    platt_bias: Optional Platt scaling intercept bias `b` in
      `sigma((logit_A - logit_B) / T + b)`.
    exclude_state_keys: State dictionary keys to strip before evaluation to
      prevent boolean echo-leakage.
    type: Discriminator literal `"noul"`.
  """

  question: str
  true_description: str = (
      "Yes, the proposition is true and supported by the state."
  )
  false_description: str = (
      "No, the proposition is false or not supported by the state."
  )
  abstain_band: tuple[float, float] | None = None
  temperature: float | None = None
  platt_bias: float = 0.0
  exclude_state_keys: Sequence[str] = ()
  type: Literal["noul"] = "noul"

  def __post_init__(self) -> None:
    if not self.question or not self.question.strip():
      raise ValueError("Noul.question must be a non-empty string.")
    if self.temperature is not None and self.temperature <= 0.0:
      raise ValueError(f"Noul.temperature must be > 0, got {self.temperature}.")
    if self.abstain_band is not None:
      low, high = self.abstain_band
      if not (0.0 <= low < high <= 1.0):
        raise ValueError(
            "Noul.abstain_band must satisfy 0.0 <= low < high <= 1.0, "
            f"got ({low}, {high})."
        )


@dataclasses.dataclass(frozen=True)
class Choice:
  """Categorical selection over K mutually exclusive options (2 <= K <= 255).

  For `2 <= K <= 20`, options are bijected to single uppercase ASCII letters
  `A..T` in a single `text/x.enum` call. For `20 < K <= 255`, options are
  evaluated via a 2-stage hierarchical coarse-to-fine tree of `text/x.enum`
  calls (each <= 20 options).

  Attributes:
    question: Categorical question to evaluate against the current state.
    options: Mapping of `option_key -> description_or_metadata`.
    allow_none_escape_hatch: If True, injects an explicit `NONE_OF_THE_ABOVE`
      option to capture out-of-domain states.
    none_option_key: Key name used when `allow_none_escape_hatch=True`.
    permute_debias: If True, runs both forward and reversed option orderings and
      averages the resulting distributions while reporting Jensen-Shannon
      divergence.
    temperature: Optional per-question calibration temperature override.
    type: Discriminator literal `"choice"`.
  """

  question: str
  options: Mapping[str, JSONContent]
  allow_none_escape_hatch: bool = False
  none_option_key: str = "NONE_OF_THE_ABOVE"
  permute_debias: bool = False
  temperature: float | None = None
  type: Literal["choice"] = "choice"

  def __post_init__(self) -> None:
    if not self.question or not self.question.strip():
      raise ValueError("Choice.question must be a non-empty string.")
    effective_count = len(self.options) + (
        1
        if self.allow_none_escape_hatch
        and self.none_option_key not in self.options
        else 0
    )
    if not (2 <= effective_count <= 255):
      raise ValueError(
          "Choice requires between 2 and 255 options (including escape hatch), "
          f"got {effective_count}."
      )
    if self.temperature is not None and self.temperature <= 0.0:
      raise ValueError(
          f"Choice.temperature must be > 0, got {self.temperature}."
      )

  @classmethod
  def from_enum(
      cls,
      question: str,
      enum_cls: type[enum.Enum],
      descriptions: Mapping[str, JSONContent] | None = None,
      *,
      allow_none_escape_hatch: bool = False,
      permute_debias: bool = False,
      temperature: float | None = None,
  ) -> Choice:
    """Constructs a `Choice` primitive directly from a Python `Enum` class.

    Matches the ergonomics of passing a `response_schema` Enum class in
    `google-genai` while applying single-letter bijection and calibration.

    Args:
      question: The classification question.
      enum_cls: A subclass of `enum.Enum`.
      descriptions: Optional mapping from enum value (or name) to semantic
        description. Defaults to human-readable enum member names.
      allow_none_escape_hatch: Whether to append a `NONE_OF_THE_ABOVE` option.
      permute_debias: Whether to run 2-pass forward/reverse debiasing.
      temperature: Optional calibration temperature override.

    Returns:
      A validated `Choice` instance.
    """
    desc_map = descriptions or {}
    opts: dict[str, JSONContent] = {}
    for member in enum_cls:
      key = str(member.value)
      default_desc = member.name.replace("_", " ").title()
      opts[key] = desc_map.get(key, desc_map.get(member.name, default_desc))
    return cls(
        question=question,
        options=opts,
        allow_none_escape_hatch=allow_none_escape_hatch,
        permute_debias=permute_debias,
        temperature=temperature,
    )


@dataclasses.dataclass(frozen=True)
class Score:
  """Continuous ordinal rating evaluated as expected value over an L-level rubric.

  Bijects `L` ordered rubric criteria (`2 <= L <= 10`) to single uppercase ASCII
  letters `A..J` via `text/x.enum`, computes the probability-weighted expected
  value `sum_i v_i * p_i`, and measures ordinal confidence via Popoviciu's
  variance bound.

  Attributes:
    question: Ordinal evaluation question.
    criteria: Ordered sequence of rubric level descriptions from lowest
      (`index 0`) to highest (`index L - 1`). Must have `2 <= len(criteria) <=
      10`.
    level_values: Optional custom strictly increasing numeric values `[v_0, ...,
      v_{L-1}]`. Defaults to `[0.0, 1.0, ..., float(L - 1)]`.
    temperature: Optional per-question calibration temperature override for the
      probability distribution and confidence calculation.
    point_estimate_temperature: Temperature used when computing the point
      expectation `score = sum(v_i * p_i)`. Defaults to `1.0` to avoid midpoint
      shrinkage under high `T`.
    type: Discriminator literal `"score"`.
  """

  question: str
  criteria: Sequence[JSONContent]
  level_values: Sequence[float] | None = None
  temperature: float | None = None
  point_estimate_temperature: float = 1.0
  type: Literal["score"] = "score"

  def __post_init__(self) -> None:
    if not self.question or not self.question.strip():
      raise ValueError("Score.question must be a non-empty string.")
    num_levels = len(self.criteria)
    if not (2 <= num_levels <= 10):
      raise ValueError(
          f"Score.criteria must have between 2 and 10 levels, got {num_levels}."
      )
    if self.level_values is not None:
      if len(self.level_values) != num_levels:
        raise ValueError(
            f"Score.level_values length ({len(self.level_values)}) must match "
            f"criteria length ({num_levels})."
        )
      for idx in range(1, num_levels):
        if self.level_values[idx] <= self.level_values[idx - 1]:
          raise ValueError(
              "Score.level_values must be strictly increasing, "
              f"got {list(self.level_values)}."
          )
    if self.temperature is not None and self.temperature <= 0.0:
      raise ValueError(
          f"Score.temperature must be > 0, got {self.temperature}."
      )
    if self.point_estimate_temperature <= 0.0:
      raise ValueError(
          "Score.point_estimate_temperature must be > 0, got "
          f"{self.point_estimate_temperature}."
      )

  def resolved_level_values(self) -> list[float]:
    """Returns the numeric values `[v_0, ..., v_{L-1}]` for each rubric level."""
    if self.level_values is not None:
      return [float(v) for v in self.level_values]
    return [float(i) for i in range(len(self.criteria))]


Question = Noul | Choice | Score
