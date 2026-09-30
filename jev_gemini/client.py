"""Gemini Flash System-One client on Vertex AI (`google-genai` SDK)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent import futures
import enum
import math
import time
from typing import Any

from jev_gemini._types_compat import genai
from jev_gemini._types_compat import HAS_GOOGLE_GENAI
from jev_gemini._types_compat import types
from jev_gemini.calibration import apply_platt_scaling
from jev_gemini.calibration import apply_temperature_scaling
from jev_gemini.calibration import compute_score_metrics
from jev_gemini.calibration import jensen_shannon_divergence
from jev_gemini.calibration import normalize_distribution
from jev_gemini.calibration import normalized_shannon_entropy
from jev_gemini.compiler import canonicalize_state
from jev_gemini.compiler import compile_primitive_prompt
from jev_gemini.compiler import CompiledPrompt
from jev_gemini.compiler import DEFAULT_SYSTEM_INSTRUCTION
from jev_gemini.extractor import extract_decision_step_distribution
from jev_gemini.extractor import ExtractedStepResult
from jev_gemini.primitives import Choice
from jev_gemini.primitives import JSONContent
from jev_gemini.primitives import Noul
from jev_gemini.primitives import Question
from jev_gemini.primitives import Score
from jev_gemini.responses import Answer
from jev_gemini.responses import ChoiceAnswer
from jev_gemini.responses import NoulAnswer
from jev_gemini.responses import ScoreAnswer
from jev_gemini.responses import StepDiagnostics
from jev_gemini.responses import SystemOneResponse


class UnsupportedBackendError(ValueError):
  """Raised if a caller configures `vertexai=False` (AI Studio lacks logprobs)."""


class GeminiSystemOneClient:
  """Evaluates calibrated Jev primitives (`Noul`, `Choice`, `Score`) on Vertex AI.

  Uses constrained enum decoding (`response_mime_type="text/x.enum"`),
  `response_logprobs=True`, and `max_output_tokens=5` with zero/minimal
  thinking on Gemini Flash via the `google-genai` SDK (`vertexai=True`).
  """

  def __init__(
      self,
      *,
      model: str = "gemini-2.5-flash",
      project: str | None = None,
      location: str = "us-central1",
      vertexai: bool = True,
      default_temperature: float = 2.0,
      min_mass_threshold: float = 0.80,
      max_output_tokens: int = 5,
      system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
      allow_text_fallback: bool = False,
      hierarchical_top_buckets: int = 2,
      genai_client: Any | None = None,
  ) -> None:
    """Initializes `GeminiSystemOneClient` targeting Gemini on Vertex AI.

    Args:
      model: Vertex AI Gemini model ID (default `"gemini-2.5-flash"`).
      project: Google Cloud project ID for Vertex AI.
      location: Google Cloud region (default `"us-central1"`).
      vertexai: Must be `True` (public AI Studio API does not support
        `response_logprobs=True`).
      default_temperature: Default post-hoc calibration temperature `T > 0`.
      min_mass_threshold: Minimum matched enum mass at the selected decoding
        step (`default = 0.80`).
      max_output_tokens: Output token budget (`>= 5` required to accommodate
        Gemini 3's 4-token `<ctrl94>thought\n<ctrl95>` header).
      system_instruction: System instruction for System-One evaluation.
      allow_text_fallback: Whether to degrade gracefully if `logprobs_result`
        is absent.
      hierarchical_top_buckets: Number of top coarse buckets to expand when
        `Choice` has `> 20` options.
      genai_client: Optional pre-initialized `google.genai.Client` (or fake
        client for unit tests / offline demos).
    """
    if not vertexai:
      raise UnsupportedBackendError(
          "`vertexai=False` (AI Studio Developer API) does not support "
          "`response_logprobs=True`. Please use Vertex AI (`vertexai=True`)."
      )
    if default_temperature <= 0.0:
      raise ValueError(
          f"default_temperature must be > 0, got {default_temperature}."
      )
    if max_output_tokens < 5:
      raise ValueError(
          f"max_output_tokens must be >= 5 to prevent truncation inside "
          f"minimal thought headers, got {max_output_tokens}."
      )

    self._model = model
    self._project = project
    self._location = location
    self._default_temperature = default_temperature
    self._min_mass_threshold = min_mass_threshold
    self._max_output_tokens = max_output_tokens
    self._system_instruction = system_instruction
    self._allow_text_fallback = allow_text_fallback
    self._hierarchical_top_buckets = max(1, hierarchical_top_buckets)

    if genai_client is not None:
      self._client = genai_client
    else:
      if not HAS_GOOGLE_GENAI or genai is None:
        raise ImportError(
            "The `google-genai` package is required to connect to Vertex AI. "
            "Install it with `pip install google-genai`."
        )
      self._client = genai.Client(
          vertexai=True,
          project=project,
          location=location,
      )

  @property
  def model(self) -> str:
    return self._model

  @property
  def genai_client(self) -> Any:
    return self._client

  def _build_thinking_config(self) -> Any:
    """Selects `thinking_budget=0` (2.5) or `thinking_level='MINIMAL'` (3.x)."""
    model_lower = self._model.lower()
    if "gemini-3" in model_lower or "v4" in model_lower:
      return types.ThinkingConfig(thinking_level="MINIMAL")
    return types.ThinkingConfig(thinking_budget=0)

  def _build_generate_config(self, symbols: Sequence[str]) -> Any:
    return types.GenerateContentConfig(
        system_instruction=self._system_instruction,
        response_mime_type="text/x.enum",
        response_schema={"type": "STRING", "enum": list(symbols)},
        response_logprobs=True,
        logprobs=len(symbols),
        temperature=0.0,
        max_output_tokens=self._max_output_tokens,
        thinking_config=self._build_thinking_config(),
    )

  def _invoke_enum_sync(self, compiled: CompiledPrompt) -> ExtractedStepResult:
    config = self._build_generate_config(compiled.symbols)
    response = self._client.models.generate_content(
        model=self._model,
        contents=compiled.prompt,
        config=config,
    )
    return extract_decision_step_distribution(
        response,
        compiled.symbols,
        min_mass_threshold=self._min_mass_threshold,
        allow_text_fallback=self._allow_text_fallback,
    )

  async def _invoke_enum_async(
      self, compiled: CompiledPrompt
  ) -> ExtractedStepResult:
    config = self._build_generate_config(compiled.symbols)
    aio = getattr(self._client, "aio", None)
    if aio is not None and hasattr(aio, "models"):
      response = await aio.models.generate_content(
          model=self._model,
          contents=compiled.prompt,
          config=config,
      )
    else:
      loop = asyncio.get_running_loop()
      response = await loop.run_in_executor(
          None,
          lambda: self._client.models.generate_content(
              model=self._model,
              contents=compiled.prompt,
              config=config,
          ),
      )
    return extract_decision_step_distribution(
        response,
        compiled.symbols,
        min_mass_threshold=self._min_mass_threshold,
        allow_text_fallback=self._allow_text_fallback,
    )

  # --------------------------------------------------------------------------
  # Post-Extraction Primitive Assemblers
  # --------------------------------------------------------------------------

  def _assemble_noul(
      self,
      question: Noul,
      compiled: CompiledPrompt,
      extracted: ExtractedStepResult,
  ) -> NoulAnswer:
    temp = question.temperature or self._default_temperature
    sym_true = compiled.key_to_symbol["True"]
    sym_false = compiled.key_to_symbol["False"]
    p_true_raw = extracted.symbol_linear_probs.get(sym_true, 0.0)
    p_false_raw = extracted.symbol_linear_probs.get(sym_false, 0.0)

    p_true, logit_gap = apply_platt_scaling(
        p_true_raw,
        p_false_raw,
        temperature=temp,
        platt_bias=question.platt_bias,
    )
    if question.abstain_band is not None:
      low, high = question.abstain_band
      value: bool | None = None if (low <= p_true < high) else (p_true >= high)
    else:
      value = p_true >= 0.5

    confidence = max(p_true, 1.0 - p_true)
    norm_conf = abs(2.0 * p_true - 1.0)
    entropy = normalized_shannon_entropy([p_true, 1.0 - p_true])
    unobserved_keys = [
        compiled.symbol_to_key[s] for s in extracted.unobserved_symbols
    ]
    diagnostics = StepDiagnostics(
        selected_step_index=extracted.selected_step_index,
        total_decoding_steps=extracted.total_decoding_steps,
        thought_tokens_stripped=extracted.thought_tokens_stripped,
        matched_enum_mass=extracted.matched_enum_mass,
        unobserved_options=unobserved_keys,
        calibration_temperature=temp,
        echo_leakage_warning=compiled.echo_leakage_warning,
        is_degraded_fallback=extracted.is_degraded_fallback,
    )
    return NoulAnswer(
        probability=p_true,
        value=value,
        confidence=confidence,
        normalized_confidence=norm_conf,
        entropy=entropy,
        logit_gap=logit_gap,
        diagnostics=diagnostics,
    )

  def _assemble_choice(
      self,
      question: Choice,
      compiled: CompiledPrompt,
      extracted: ExtractedStepResult,
      *,
      rev_compiled: CompiledPrompt | None = None,
      rev_extracted: ExtractedStepResult | None = None,
  ) -> ChoiceAnswer:
    temp = question.temperature or self._default_temperature
    raw_by_key = {
        compiled.symbol_to_key[sym]: prob
        for sym, prob in extracted.symbol_linear_probs.items()
    }
    raw_norm = normalize_distribution(raw_by_key)
    cal_fwd = apply_temperature_scaling(raw_by_key, temperature=temp)

    js_div: float | None = None
    if rev_compiled is not None and rev_extracted is not None:
      raw_rev_by_key = {
          rev_compiled.symbol_to_key[sym]: prob
          for sym, prob in rev_extracted.symbol_linear_probs.items()
      }
      cal_rev = apply_temperature_scaling(raw_rev_by_key, temperature=temp)
      js_div = jensen_shannon_divergence(cal_fwd, cal_rev)
      cal_combined = {
          k: 0.5 * (cal_fwd[k] + cal_rev.get(k, 0.0)) for k in cal_fwd
      }
      cal_probs = normalize_distribution(cal_combined)
    else:
      cal_probs = cal_fwd

    sorted_items = sorted(
        cal_probs.items(), key=lambda kv: kv[1], reverse=True
    )
    winning_key, peak_prob = sorted_items[0]
    second_prob = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    k_count = len(cal_probs)
    norm_conf = (
        max(0.0, min(1.0, (k_count * peak_prob - 1.0) / (k_count - 1.0)))
        if k_count > 1
        else 1.0
    )
    entropy = normalized_shannon_entropy(list(cal_probs.values()))
    certainty = max(0.0, min(1.0, 1.0 - entropy))

    unobserved_keys = [
        compiled.symbol_to_key[s] for s in extracted.unobserved_symbols
    ]
    diagnostics = StepDiagnostics(
        selected_step_index=extracted.selected_step_index,
        total_decoding_steps=extracted.total_decoding_steps,
        thought_tokens_stripped=extracted.thought_tokens_stripped,
        matched_enum_mass=extracted.matched_enum_mass,
        unobserved_options=unobserved_keys,
        calibration_temperature=temp,
        jensen_shannon_divergence=js_div,
        is_degraded_fallback=extracted.is_degraded_fallback,
    )
    return ChoiceAnswer(
        choice=winning_key,
        confidence=peak_prob,
        normalized_confidence=norm_conf,
        margin=peak_prob - second_prob,
        certainty=certainty,
        probabilities=dict(sorted_items),
        raw_probabilities=raw_norm,
        diagnostics=diagnostics,
    )

  def _assemble_score(
      self,
      question: Score,
      compiled: CompiledPrompt,
      extracted: ExtractedStepResult,
  ) -> ScoreAnswer:
    temp = question.temperature or self._default_temperature
    num_levels = len(question.criteria)
    values = question.resolved_level_values()

    raw_by_idx = {
        int(compiled.symbol_to_key[sym]): prob
        for sym, prob in extracted.symbol_linear_probs.items()
    }
    raw_str_map = {str(i): raw_by_idx.get(i, 0.0) for i in range(num_levels)}
    cal_str_map = apply_temperature_scaling(raw_str_map, temperature=temp)
    point_str_map = apply_temperature_scaling(
        raw_str_map, temperature=question.point_estimate_temperature
    )

    cal_probs = [cal_str_map[str(i)] for i in range(num_levels)]
    point_probs = [point_str_map[str(i)] for i in range(num_levels)]

    (
        score_val,
        norm_score,
        confidence,
        std_dev,
        entropy,
        is_bimodal,
        mode_level,
    ) = compute_score_metrics(
        calibrated_probs=cal_probs,
        level_values=values,
        point_probs=point_probs,
    )

    unobserved_keys = [
        f"level_{compiled.symbol_to_key[s]}"
        for s in extracted.unobserved_symbols
    ]
    diagnostics = StepDiagnostics(
        selected_step_index=extracted.selected_step_index,
        total_decoding_steps=extracted.total_decoding_steps,
        thought_tokens_stripped=extracted.thought_tokens_stripped,
        matched_enum_mass=extracted.matched_enum_mass,
        unobserved_options=unobserved_keys,
        calibration_temperature=temp,
        is_degraded_fallback=extracted.is_degraded_fallback,
    )
    return ScoreAnswer(
        score=score_val,
        normalized_score=norm_score,
        confidence=confidence,
        std_dev=std_dev,
        entropy=entropy,
        is_bimodal=is_bimodal,
        mode_level=mode_level,
        distribution=cal_probs,
        level_values=values,
        diagnostics=diagnostics,
    )

  # --------------------------------------------------------------------------
  # Hierarchical Coarse-to-Fine Tree Routing for Choice (20 < K <= 255)
  # --------------------------------------------------------------------------

  def _partition_choice_buckets(
      self, question: Choice
  ) -> list[tuple[str, dict[str, JSONContent]]]:
    items = list(question.options.items())
    if (
        question.allow_none_escape_hatch
        and question.none_option_key not in question.options
    ):
      items.append((
          question.none_option_key,
          "None of the listed options apply to the current state.",
      ))
    num_items = len(items)
    num_buckets = min(20, math.ceil(num_items / 20))
    bucket_size = math.ceil(num_items / num_buckets)
    buckets: list[tuple[str, dict[str, JSONContent]]] = []
    for b_idx in range(num_buckets):
      chunk = items[b_idx * bucket_size : (b_idx + 1) * bucket_size]
      if not chunk:
        continue
      b_key = f"bucket_{b_idx}"
      buckets.append((b_key, dict(chunk)))
    return buckets

  def _evaluate_hierarchical_choice_sync(
      self, state: Any, question: Choice, state_str: str
  ) -> ChoiceAnswer:
    buckets = self._partition_choice_buckets(question)
    coarse_options = {
        b_key: f"Options group containing: {', '.join(list(sub_opts.keys())[:8])}"
        for b_key, sub_opts in buckets
    }
    coarse_q = Choice(
        question=question.question,
        options=coarse_options,
        temperature=question.temperature,
    )
    coarse_ans = self._evaluate_single_sync(state, coarse_q, state_str)
    assert isinstance(coarse_ans, ChoiceAnswer)

    bucket_map = dict(buckets)
    top_bucket_keys = list(coarse_ans.probabilities.keys())[
        : self._hierarchical_top_buckets
    ]

    combined_probs: dict[str, float] = {k: 0.0 for k in question.options}
    if question.allow_none_escape_hatch:
      combined_probs[question.none_option_key] = 0.0

    for b_key in top_bucket_keys:
      b_prob = coarse_ans.probabilities[b_key]
      sub_opts = bucket_map[b_key]
      if len(sub_opts) == 1:
        only_key = next(iter(sub_opts.keys()))
        combined_probs[only_key] = b_prob
      else:
        fine_q = Choice(
            question=question.question,
            options=sub_opts,
            temperature=question.temperature,
        )
        fine_ans = self._evaluate_single_sync(state, fine_q, state_str)
        assert isinstance(fine_ans, ChoiceAnswer)
        for k, p_cond in fine_ans.probabilities.items():
          combined_probs[k] = b_prob * p_cond

    norm_probs = normalize_distribution(combined_probs)
    sorted_items = sorted(
        norm_probs.items(), key=lambda kv: kv[1], reverse=True
    )
    win_key, peak_prob = sorted_items[0]
    second_prob = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    k_count = len(norm_probs)
    norm_conf = (k_count * peak_prob - 1.0) / (k_count - 1.0)
    entropy = normalized_shannon_entropy(list(norm_probs.values()))
    return ChoiceAnswer(
        choice=win_key,
        confidence=peak_prob,
        normalized_confidence=max(0.0, min(1.0, norm_conf)),
        margin=peak_prob - second_prob,
        certainty=max(0.0, min(1.0, 1.0 - entropy)),
        probabilities=dict(sorted_items),
        raw_probabilities=dict(sorted_items),
        diagnostics=coarse_ans.diagnostics,
    )

  # --------------------------------------------------------------------------
  # Single & Multi-Question Evaluation Entry Points
  # --------------------------------------------------------------------------

  def _evaluate_single_sync(
      self,
      state: Any,
      question: Question,
      precanonicalized_state: str | None = None,
  ) -> Answer:
    if isinstance(question, Choice):
      eff_count = len(question.options) + (
          1
          if question.allow_none_escape_hatch
          and question.none_option_key not in question.options
          else 0
      )
      if eff_count > 20:
        state_str = precanonicalized_state or canonicalize_state(state)
        return self._evaluate_hierarchical_choice_sync(
            state, question, state_str
        )

    compiled = compile_primitive_prompt(
        state, question, precanonicalized_state=precanonicalized_state
    )
    extracted = self._invoke_enum_sync(compiled)

    if isinstance(question, Noul):
      return self._assemble_noul(question, compiled, extracted)

    if isinstance(question, Choice):
      if question.permute_debias:
        rev_compiled = compile_primitive_prompt(
            state,
            question,
            reverse_options=True,
            precanonicalized_state=precanonicalized_state,
        )
        rev_extracted = self._invoke_enum_sync(rev_compiled)
        return self._assemble_choice(
            question,
            compiled,
            extracted,
            rev_compiled=rev_compiled,
            rev_extracted=rev_extracted,
        )
      return self._assemble_choice(question, compiled, extracted)

    if isinstance(question, Score):
      return self._assemble_score(question, compiled, extracted)

    raise TypeError(f"Unsupported question type: {type(question)}")

  async def _evaluate_single_async(
      self,
      state: Any,
      question: Question,
      precanonicalized_state: str | None = None,
  ) -> Answer:
    if isinstance(question, Choice):
      eff_count = len(question.options) + (
          1
          if question.allow_none_escape_hatch
          and question.none_option_key not in question.options
          else 0
      )
      if eff_count > 20:
        loop = asyncio.get_running_loop()
        state_str = precanonicalized_state or canonicalize_state(state)
        return await loop.run_in_executor(
            None,
            self._evaluate_hierarchical_choice_sync,
            state,
            question,
            state_str,
        )

    compiled = compile_primitive_prompt(
        state, question, precanonicalized_state=precanonicalized_state
    )
    if isinstance(question, Choice) and question.permute_debias:
      rev_compiled = compile_primitive_prompt(
          state,
          question,
          reverse_options=True,
          precanonicalized_state=precanonicalized_state,
      )
      extracted, rev_extracted = await asyncio.gather(
          self._invoke_enum_async(compiled),
          self._invoke_enum_async(rev_compiled),
      )
      return self._assemble_choice(
          question,
          compiled,
          extracted,
          rev_compiled=rev_compiled,
          rev_extracted=rev_extracted,
      )

    extracted = await self._invoke_enum_async(compiled)
    if isinstance(question, Noul):
      return self._assemble_noul(question, compiled, extracted)
    if isinstance(question, Choice):
      return self._assemble_choice(question, compiled, extracted)
    if isinstance(question, Score):
      return self._assemble_score(question, compiled, extracted)
    raise TypeError(f"Unsupported question type: {type(question)}")

  def evaluate_one(self, state: Any, question: Question) -> Answer:
    """Evaluates a single `Noul`, `Choice`, or `Score` question synchronously."""
    return self._evaluate_single_sync(state, question)

  async def evaluate_one_async(self, state: Any, question: Question) -> Answer:
    """Evaluates a single `Noul`, `Choice`, or `Score` question asynchronously."""
    return await self._evaluate_single_async(state, question)

  def ask(
      self,
      state: Any,
      questions: Mapping[str, Question],
      *,
      max_workers: int | None = None,
  ) -> SystemOneResponse:
    """Evaluates multiple questions in parallel against a shared `state` prefix."""
    start_t = time.perf_counter()
    shared_state_str = canonicalize_state(state)
    worker_count = max_workers or max(1, len(questions))

    with futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
      fut_map = {
          key: pool.submit(
              self._evaluate_single_sync, state, q, shared_state_str
          )
          for key, q in questions.items()
      }
      answers = {key: fut.result() for key, fut in fut_map.items()}

    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return SystemOneResponse(
        answers=answers,
        latency_ms=elapsed_ms,
        model=self._model,
    )

  async def ask_async(
      self,
      state: Any,
      questions: Mapping[str, Question],
  ) -> SystemOneResponse:
    """Evaluates multiple questions concurrently via `asyncio.gather`."""
    start_t = time.perf_counter()
    shared_state_str = canonicalize_state(state)
    keys = list(questions.keys())
    coros = [
        self._evaluate_single_async(state, questions[k], shared_state_str)
        for k in keys
    ]
    results = await asyncio.gather(*coros)
    answers = dict(zip(keys, results, strict=True))
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return SystemOneResponse(
        answers=answers,
        latency_ms=elapsed_ms,
        model=self._model,
    )

  def classify_enum(
      self,
      state: Any,
      question: str,
      enum_cls: type[enum.Enum],
      descriptions: Mapping[str, JSONContent] | None = None,
      *,
      allow_none_escape_hatch: bool = False,
      permute_debias: bool = False,
      temperature: float | None = None,
  ) -> ChoiceAnswer:
    """Classifies `state` directly into a Python `Enum` with calibrated probs."""
    choice_q = Choice.from_enum(
        question=question,
        enum_cls=enum_cls,
        descriptions=descriptions,
        allow_none_escape_hatch=allow_none_escape_hatch,
        permute_debias=permute_debias,
        temperature=temperature,
    )
    ans = self.evaluate_one(state, choice_q)
    assert isinstance(ans, ChoiceAnswer)
    return ans

  async def classify_enum_async(
      self,
      state: Any,
      question: str,
      enum_cls: type[enum.Enum],
      descriptions: Mapping[str, JSONContent] | None = None,
      *,
      allow_none_escape_hatch: bool = False,
      permute_debias: bool = False,
      temperature: float | None = None,
  ) -> ChoiceAnswer:
    """Asynchronously classifies `state` directly into a Python `Enum`."""
    choice_q = Choice.from_enum(
        question=question,
        enum_cls=enum_cls,
        descriptions=descriptions,
        allow_none_escape_hatch=allow_none_escape_hatch,
        permute_debias=permute_debias,
        temperature=temperature,
    )
    ans = await self.evaluate_one_async(state, choice_q)
    assert isinstance(ans, ChoiceAnswer)
    return ans
