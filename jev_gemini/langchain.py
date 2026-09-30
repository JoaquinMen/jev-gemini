"""Optional LangChain / LangGraph `Runnable` adapter for `jev_gemini`.

Provides `GeminiJevClassifier`, a drop-in replacement for TypeSafe AI's
`TypeSafeClassifier` that evaluates `Noul`, `Choice`, and `Score` primitives
and injects the calibrated answers into a LangChain / LangGraph state dict.
Does not require `langchain-core` to be installed unless LangChain features are
used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jev_gemini.client import GeminiSystemOneClient
from jev_gemini.primitives import Question

try:
  from langchain_core.runnables import Runnable  # type: ignore[import-not-found]
  from langchain_core.runnables import RunnableConfig  # type: ignore[import-not-found]
except ImportError:
  RunnableConfig = Any  # type: ignore[assignment,misc]

  class Runnable:  # type: ignore[no-redef]
    """Minimal fallback `Runnable` interface when `langchain-core` is absent."""

    def __or__(self, other: Any) -> _ChainedRunnable:
      return _ChainedRunnable(self, other)


class _ChainedRunnable(Runnable):
  """Simple pipe composition (`runnable | fn`) when `langchain-core` is absent."""

  def __init__(self, first: Any, second: Any) -> None:
    self._first = first
    self._second = second

  def invoke(self, input_data: Any, config: Any = None) -> Any:
    intermediate = self._first.invoke(input_data, config=config)
    if hasattr(self._second, "invoke"):
      return self._second.invoke(intermediate, config=config)
    if callable(self._second):
      return self._second(intermediate)
    raise TypeError(f"Downstream stage is not callable: {self._second!r}")


class GeminiJevClassifier(Runnable):
  """LangChain `Runnable` drop-in replacement for `TypeSafeClassifier`.

  Evaluates a dictionary of `Noul`, `Choice`, and `Score` primitives against an
  input state dictionary and writes the resulting answers under `output_key`.
  """

  def __init__(
      self,
      client: GeminiSystemOneClient,
      questions: Mapping[str, Question],
      *,
      state_key: str | None = None,
      output_key: str = "decisions",
      serialize_answers: bool = False,
  ) -> None:
    """Initializes `GeminiJevClassifier`.

    Args:
      client: Configured `GeminiSystemOneClient` instance.
      questions: Mapping of `question_name -> Noul | Choice | Score`.
      state_key: Optional key to extract from the input dictionary as `state`.
        If `None`, the entire input dictionary (excluding `output_key`) is
        evaluated as the state.
      output_key: Key name in the returned dictionary where the calibrated
        answers are stored (default `"decisions"`).
      serialize_answers: If True, stores plain dicts (`.to_dict()`) instead of
        `NoulAnswer` / `ChoiceAnswer` / `ScoreAnswer` dataclass objects.
    """
    super().__init__()
    self.client = client
    self.questions = dict(questions)
    self.state_key = state_key
    self.output_key = output_key
    self.serialize_answers = serialize_answers

  def _extract_state(self, input_data: Any) -> Any:
    if isinstance(input_data, Mapping):
      if self.state_key is not None and self.state_key in input_data:
        return input_data[self.state_key]
      return {k: v for k, v in input_data.items() if k != self.output_key}
    return input_data

  def _format_output(
      self, input_data: Any, response_answers: Mapping[str, Any]
  ) -> dict[str, Any]:
    formatted = (
        {k: v.to_dict() for k, v in response_answers.items()}
        if self.serialize_answers
        else dict(response_answers)
    )
    if isinstance(input_data, Mapping):
      out = dict(input_data)
      out[self.output_key] = formatted
      return out
    return {"state": input_data, self.output_key: formatted}

  def invoke(
      self,
      input: Any,  # pylint: disable=redefined-builtin
      config: RunnableConfig | None = None,
      **kwargs: Any,
  ) -> dict[str, Any]:
    """Synchronously evaluates questions and returns updated state dict."""
    del config, kwargs
    target_state = self._extract_state(input)
    sys_one_resp = self.client.ask(state=target_state, questions=self.questions)
    return self._format_output(input, sys_one_resp.answers)

  async def ainvoke(
      self,
      input: Any,  # pylint: disable=redefined-builtin
      config: RunnableConfig | None = None,
      **kwargs: Any,
  ) -> dict[str, Any]:
    """Asynchronously evaluates questions and returns updated state dict."""
    del config, kwargs
    target_state = self._extract_state(input)
    sys_one_resp = await self.client.ask_async(
        state=target_state, questions=self.questions
    )
    return self._format_output(input, sys_one_resp.answers)
