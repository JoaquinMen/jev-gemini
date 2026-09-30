"""Compatibility loader for `google.genai` and `google.genai.types`.

Uses the official `google-genai` SDK when installed in the Python environment,
and provides lightweight dataclass equivalents for `google.genai.types` so unit
tests and offline simulations can run even before `google-genai` is installed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

try:
  from google import genai as _genai  # type: ignore[import-not-found]
  from google.genai import types as _types  # type: ignore[import-not-found]

  HAS_GOOGLE_GENAI = True
  genai = _genai
  types = _types
except ImportError:
  HAS_GOOGLE_GENAI = False
  genai = None  # type: ignore[assignment]

  class _TypesFallback:
    """Fallback types matching `google.genai.types` structure."""

    @dataclasses.dataclass
    class ThinkingConfig:
      thinking_budget: int | None = None
      thinking_level: str | None = None

    @dataclasses.dataclass
    class GenerateContentConfig:
      system_instruction: str | None = None
      response_mime_type: str | None = None
      response_schema: Any = None
      response_logprobs: bool | None = None
      logprobs: int | None = None
      temperature: float | None = None
      max_output_tokens: int | None = None
      thinking_config: Any = None
      cached_content: str | None = None
      tools: list[Any] | None = None

    @dataclasses.dataclass
    class LogprobsResultCandidate:
      token: str | None = None
      log_probability: float | None = None
      token_id: int | None = None

    @dataclasses.dataclass
    class LogprobsResultTopCandidates:
      candidates: list[Any] | None = None

    @dataclasses.dataclass
    class LogprobsResult:
      chosen_candidates: list[Any] | None = None
      top_candidates: list[Any] | None = None

    @dataclasses.dataclass
    class Part:
      text: str = ""

      @classmethod
      def from_text(cls, *, text: str) -> _TypesFallback.Part:
        return cls(text=text)

    @dataclasses.dataclass
    class Content:
      role: str = "model"
      parts: list[Any] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class Candidate:
      content: Any = None
      logprobs_result: Any = None
      finish_reason: str | None = None

    @dataclasses.dataclass
    class GenerateContentResponse:
      candidates: list[Any] = dataclasses.field(default_factory=list)

      @property
      def text(self) -> str:
        if (
            self.candidates
            and self.candidates[0].content
            and self.candidates[0].content.parts
        ):
          return str(self.candidates[0].content.parts[0].text)
        return ""

  types = _TypesFallback()  # type: ignore[assignment]
