"""Hardened decision-step logprob extractor for Gemini Flash (`text/x.enum`).

Implements:
  1. `<ctrl94>` / `<ctrl95>` (`<end_of_thought>`) thought-wrapper stripping for
     Gemini 3.x / v4 models where `thinking_level="MINIMAL"` emits 4 control
     tokens prior to the constrained enum decision token.
  2. Reverse (`pick="last"`) enum-mass filtering (`M_t >= min_mass_threshold`)
     so leaked English tokens inside thoughts (e.g., article `"A"`) never
     corrupt the probability distribution.
  3. Surface subword normalization and linear probability summation (`"A"` +
     `" A"`).
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

CTRL94_START_THOUGHT = "<ctrl94>"
CTRL95_END_THOUGHT = "<ctrl95>"
STRIP_CHARS = " \"'\n\t,:{}[]"


class ThoughtTruncationError(RuntimeError):
  """Raised when generation truncates inside `<ctrl94>` before `<ctrl95>`."""


class MissingLogprobsError(ValueError):
  """Raised when `LogprobsResult` is absent or contains zero valid enum mass."""


@dataclasses.dataclass(frozen=True)
class ExtractedStepResult:
  """Raw extraction output at the selected decision step `t*`."""

  symbol_linear_probs: dict[str, float]
  selected_step_index: int
  total_decoding_steps: int
  thought_tokens_stripped: int
  matched_enum_mass: float
  unobserved_symbols: list[str]
  is_degraded_fallback: bool = False


def normalize_token(token: str) -> str:
  """Normalizes a SentencePiece token to its bare uppercase enum symbol."""
  return token.strip(STRIP_CHARS)


def extract_decision_step_distribution(
    response: Any,
    valid_symbols: Sequence[str],
    *,
    min_mass_threshold: float = 0.80,
    allow_text_fallback: bool = False,
) -> ExtractedStepResult:
  """Extracts linear probabilities over `valid_symbols` at decision step `t*`.

  Args:
    response: A `google.genai.types.GenerateContentResponse`.
    valid_symbols: Ordered sequence of expected single-letter symbols (e.g.,
      `["A", "B", "C"]`).
    min_mass_threshold: Minimum matched enum probability mass `M_t` required for
      the reverse step scan (`default = 0.80`).
    allow_text_fallback: If True and `logprobs_result` is missing, falls back to
      parsing `response.text` with `is_degraded_fallback=True`.

  Returns:
    An `ExtractedStepResult` containing linear probabilities per symbol and
    step diagnostics.

  Raises:
    ThoughtTruncationError: If `<ctrl94>` appears without `<ctrl95>`.
    MissingLogprobsError: If `logprobs_result` is missing (when
      `allow_text_fallback=False`) or zero probability mass matches
      `valid_symbols`.
  """
  valid_set = set(valid_symbols)
  candidates = getattr(response, "candidates", None) or []
  lp_result = (
      getattr(candidates[0], "logprobs_result", None) if candidates else None
  )

  if lp_result is None or not (
      getattr(lp_result, "chosen_candidates", None)
      and getattr(lp_result, "top_candidates", None)
  ):
    if allow_text_fallback:
      raw_text = normalize_token(getattr(response, "text", "") or "")
      if raw_text in valid_set:
        probs = {
            sym: (1.0 if sym == raw_text else 0.0) for sym in valid_symbols
        }
        unobs = [sym for sym in valid_symbols if sym != raw_text]
        return ExtractedStepResult(
            symbol_linear_probs=probs,
            selected_step_index=0,
            total_decoding_steps=1,
            thought_tokens_stripped=0,
            matched_enum_mass=1.0,
            unobserved_symbols=unobs,
            is_degraded_fallback=True,
        )
    raise MissingLogprobsError(
        "Response does not contain populated `logprobs_result`. Ensure you are "
        "calling Vertex AI (`vertexai=True`) with `response_logprobs=True`."
    )

  chosen = lp_result.chosen_candidates or []
  top_per_step = lp_result.top_candidates or []
  total_steps = len(top_per_step)

  # 1. Detect <ctrl94> (<start_of_thought>) and <ctrl95> (<end_of_thought>)
  ctrl94_idx = -1
  ctrl95_idx = -1
  for idx, cand in enumerate(chosen):
    tok = getattr(cand, "token", None) or ""
    if tok == CTRL94_START_THOUGHT and ctrl94_idx == -1:
      ctrl94_idx = idx
    elif tok == CTRL95_END_THOUGHT:
      ctrl95_idx = idx

  if ctrl94_idx != -1 and ctrl95_idx == -1:
    raise ThoughtTruncationError(
        f"Model emitted {CTRL94_START_THOUGHT} at step {ctrl94_idx} without "
        f"closing {CTRL95_END_THOUGHT}. Increase `max_output_tokens` "
        "(recommended >= 5)."
    )

  search_start = (ctrl95_idx + 1) if ctrl95_idx != -1 else 0
  if search_start >= total_steps:
    raise ThoughtTruncationError(
        f"Thought block ended at step {ctrl95_idx}, leaving zero decoding "
        f"steps for the enum token (total_steps={total_steps}). Increase "
        "`max_output_tokens` (recommended >= 5)."
    )

  # 2. Aggregate linear probabilities per valid symbol across post-thought steps
  step_candidates: list[tuple[int, float, dict[str, float], set[str]]] = []
  for step_idx in range(search_start, total_steps):
    agg_probs: dict[str, float] = {sym: 0.0 for sym in valid_symbols}
    observed_syms: set[str] = set()
    step_top = getattr(top_per_step[step_idx], "candidates", None) or []

    # Fallback if top_candidates[step_idx] is empty but chosen[step_idx] exists
    if not step_top and step_idx < len(chosen):
      step_top = [chosen[step_idx]]

    for cand in step_top:
      norm_tok = normalize_token(getattr(cand, "token", None) or "")
      log_prob = getattr(cand, "log_probability", None)
      if norm_tok in valid_set and log_prob is not None:
        # Sum linear probabilities across surface subword variants ("A" + " A")
        agg_probs[norm_tok] += math.exp(float(log_prob))
        observed_syms.add(norm_tok)

    matched_mass = sum(agg_probs.values())
    step_candidates.append((step_idx, matched_mass, agg_probs, observed_syms))

  # 3. Reverse scan (`pick="last"`): select the last step with mass >= threshold
  selected_tuple: tuple[int, float, dict[str, float], set[str]] | None = None
  for item in reversed(step_candidates):
    if item[1] >= min_mass_threshold:
      selected_tuple = item
      break

  # Fallback to the post-thought step with highest positive matched mass
  if selected_tuple is None:
    best_item = max(step_candidates, key=lambda x: x[1])
    if best_item[1] <= 0.0:
      raise MissingLogprobsError(
          f"Zero probability mass matched valid enum symbols {list(valid_symbols)} "
          f"across steps [{search_start}..{total_steps - 1}]."
      )
    selected_tuple = best_item

  sel_idx, sel_mass, sel_probs, sel_observed = selected_tuple
  unobserved = [sym for sym in valid_symbols if sym not in sel_observed]
  return ExtractedStepResult(
      symbol_linear_probs=sel_probs,
      selected_step_index=sel_idx,
      total_decoding_steps=total_steps,
      thought_tokens_stripped=search_start,
      matched_enum_mass=sel_mass,
      unobserved_symbols=unobserved,
      is_degraded_fallback=False,
  )
