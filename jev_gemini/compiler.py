"""State canonicalizer, echo-leakage guardrail, and `A..T` prompt compiler."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import re
import string
from typing import Any

from jev_gemini.primitives import Choice
from jev_gemini.primitives import Noul
from jev_gemini.primitives import Question
from jev_gemini.primitives import Score

LETTERS = tuple(string.ascii_uppercase[:20])  # A..T (Vertex AI max logprobs=20)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a calibrated System-One state evaluator.\n"
    "Evaluate the provided STATE and answer the QUESTION by selecting the "
    "single uppercase letter corresponding to the best option.\n"
    "Do not explain your reasoning. Output only the single option letter."
)

_STOPWORDS = frozenset({
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "state",
    "that",
    "the",
    "this",
    "to",
    "user",
    "was",
    "we",
    "with",
})


@dataclasses.dataclass(frozen=True)
class CompiledPrompt:
  """Compiled prompt and single-letter bijection metadata for a primitive."""

  prompt: str
  symbols: tuple[str, ...]
  symbol_to_key: dict[str, str]
  key_to_symbol: dict[str, str]
  echo_leakage_warning: str | None = None


def canonicalize_state(
    state: Any,
    *,
    exclude_keys: Sequence[str] = (),
) -> str:
  """Serializes `state` into a deterministic prefix string for caching."""
  if isinstance(state, str):
    return state
  exclude_set = set(exclude_keys)
  if hasattr(state, "model_dump") and callable(state.model_dump):
    state = state.model_dump()
  elif dataclasses.is_dataclass(state) and not isinstance(state, type):
    state = dataclasses.asdict(state)

  if isinstance(state, Mapping):
    filtered = {k: v for k, v in state.items() if str(k) not in exclude_set}
    return json.dumps(filtered, indent=2, sort_keys=True, default=str)

  return json.dumps(state, indent=2, sort_keys=True, default=str)


def _tokenize_identifier(text: str) -> set[str]:
  tokens = re.findall(r"[a-z0-9]+", text.lower())
  return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def detect_echo_leakage(state: Any, question_text: str) -> str | None:
  """Warns if a boolean key in `state` has high lexical overlap with `Noul`."""
  if not isinstance(state, Mapping):
    return None
  q_tokens = _tokenize_identifier(question_text)
  if not q_tokens:
    return None
  for key, val in state.items():
    if isinstance(val, bool):
      key_tokens = _tokenize_identifier(str(key))
      if key_tokens and key_tokens.issubset(q_tokens):
        return (
            f"State contains boolean key {key!r}={val} matching tokens in "
            f"Noul question {question_text!r}. Consider passing "
            f"`exclude_state_keys=[{key!r}]` to prevent echo-leakage."
        )
  return None


def _format_description(desc: Any) -> str:
  if isinstance(desc, str):
    return desc
  return json.dumps(desc, sort_keys=True, default=str)


def compile_primitive_prompt(
    state: Any,
    question: Question,
    *,
    reverse_options: bool = False,
    precanonicalized_state: str | None = None,
) -> CompiledPrompt:
  """Compiles a `Noul`, `Choice`, or `Score` into a single-letter enum prompt.

  Places `=== STATE ===` at the very start of the prompt so concurrent calls
  for multiple questions against the same state share an identical token prefix.
  """
  if isinstance(question, Noul):
    state_str = (
        canonicalize_state(state, exclude_keys=question.exclude_state_keys)
        if question.exclude_state_keys or precanonicalized_state is None
        else precanonicalized_state
    )
    leak_warning = None
    if isinstance(state, Mapping):
      unfiltered = {
          k: v
          for k, v in state.items()
          if str(k) not in set(question.exclude_state_keys)
      }
      leak_warning = detect_echo_leakage(unfiltered, question.question)

    items = [
        ("True", question.true_description),
        ("False", question.false_description),
    ]
    if reverse_options:
      items.reverse()
    symbols = LETTERS[:2]
    sym_to_key = {sym: k for sym, (k, _) in zip(symbols, items, strict=True)}
    key_to_sym = {k: sym for sym, k in sym_to_key.items()}
    options_text = "\n".join(
        f"[{sym}]: {desc}"
        for sym, (_, desc) in zip(symbols, items, strict=True)
    )
    prompt = (
        f"=== STATE ===\n{state_str}\n\n"
        f"=== QUESTION (NOUL) ===\n{question.question}\n\n"
        f"=== OPTIONS ===\n{options_text}\n\n"
        f"Select the single best letter ({', '.join(symbols)}):"
    )
    return CompiledPrompt(
        prompt=prompt,
        symbols=symbols,
        symbol_to_key=sym_to_key,
        key_to_symbol=key_to_sym,
        echo_leakage_warning=leak_warning,
    )

  state_str = (
      precanonicalized_state
      if precanonicalized_state is not None
      else canonicalize_state(state)
  )

  if isinstance(question, Choice):
    items_list = list(question.options.items())
    if (
        question.allow_none_escape_hatch
        and question.none_option_key not in question.options
    ):
      items_list.append((
          question.none_option_key,
          "None of the listed options apply to the current state.",
      ))
    if len(items_list) > 20:
      raise ValueError(
          f"compile_primitive_prompt requires <= 20 options per single call, "
          f"got {len(items_list)} (use hierarchical routing in client)."
      )
    if reverse_options:
      items_list.reverse()

    symbols = LETTERS[: len(items_list)]
    sym_to_key = {
        sym: str(k) for sym, (k, _) in zip(symbols, items_list, strict=True)
    }
    key_to_sym = {k: sym for sym, k in sym_to_key.items()}
    options_text = "\n".join(
        f"[{sym}]: ({k}) {_format_description(desc)}"
        for sym, (k, desc) in zip(symbols, items_list, strict=True)
    )
    prompt = (
        f"=== STATE ===\n{state_str}\n\n"
        f"=== QUESTION (CHOICE) ===\n{question.question}\n\n"
        f"=== OPTIONS ===\n{options_text}\n\n"
        f"Select the single best letter ({', '.join(symbols)}):"
    )
    return CompiledPrompt(
        prompt=prompt,
        symbols=symbols,
        symbol_to_key=sym_to_key,
        key_to_symbol=key_to_sym,
    )

  if isinstance(question, Score):
    values = question.resolved_level_values()
    indexed_criteria = [
        (str(idx), values[idx], question.criteria[idx])
        for idx in range(len(question.criteria))
    ]
    if reverse_options:
      indexed_criteria.reverse()

    symbols = LETTERS[: len(indexed_criteria)]
    sym_to_key = {
        sym: k
        for sym, (k, _, _) in zip(symbols, indexed_criteria, strict=True)
    }
    key_to_sym = {k: sym for sym, k in sym_to_key.items()}
    rubric_text = "\n".join(
        f"[{sym}] (Level {val:g}): {_format_description(desc)}"
        for sym, (_, val, desc) in zip(symbols, indexed_criteria, strict=True)
    )
    order_note = (
        "Ordered Highest to Lowest"
        if reverse_options
        else "Ordered Lowest to Highest"
    )
    prompt = (
        f"=== STATE ===\n{state_str}\n\n"
        f"=== QUESTION (SCORE) ===\n{question.question}\n\n"
        f"=== RUBRIC LEVELS ({order_note}) ===\n{rubric_text}\n\n"
        f"Select the single best letter ({', '.join(symbols)}):"
    )
    return CompiledPrompt(
        prompt=prompt,
        symbols=symbols,
        symbol_to_key=sym_to_key,
        key_to_symbol=key_to_sym,
    )

  raise TypeError(f"Unsupported question primitive: {type(question)}")
