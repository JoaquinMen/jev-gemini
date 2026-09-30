"""Post-hoc calibration, ordinal `Score` mathematics, and temperature fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import math

_EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class CalibrationResult:
  """Summary metrics returned by `fit_temperature` and `evaluate_calibration`."""

  optimal_temperature: float
  raw_nll: float
  calibrated_nll: float
  raw_ece: float
  calibrated_ece: float
  raw_brier: float
  calibrated_brier: float


def normalize_distribution(probs: Mapping[str, float]) -> dict[str, float]:
  """Normalizes a dictionary of non-negative linear probabilities to sum to 1."""
  total = sum(max(0.0, v) for v in probs.values())
  if total <= 0.0:
    n = len(probs)
    return {k: (1.0 / n if n > 0 else 0.0) for k in probs}
  return {k: max(0.0, v) / total for k, v in probs.items()}


def apply_temperature_scaling(
    raw_probs: Mapping[str, float],
    temperature: float = 1.0,
) -> dict[str, float]:
  """Applies post-hoc temperature scaling `p_k^(1/T) / sum_j p_j^(1/T)`.

  Args:
    raw_probs: Mapping from symbol/option key to linear probability `>= 0`.
    temperature: Calibration temperature `T > 0`. `T > 1` softens over-sharp
      distributions; `T < 1` sharpens them.

  Returns:
    Normalized calibrated probability dictionary.
  """
  if temperature <= 0.0:
    raise ValueError(f"temperature must be > 0, got {temperature}.")
  if not raw_probs:
    return {}
  inv_t = 1.0 / temperature
  scaled: dict[str, float] = {}
  max_log = -math.inf
  log_vals: dict[str, float] = {}
  for k, p in raw_probs.items():
    if p > 0.0:
      lv = math.log(p) * inv_t
      log_vals[k] = lv
      if lv > max_log:
        max_log = lv

  if max_log == -math.inf:
    uniform = 1.0 / len(raw_probs)
    return {k: uniform for k in raw_probs}

  for k in raw_probs:
    if k in log_vals:
      scaled[k] = math.exp(log_vals[k] - max_log)
    else:
      scaled[k] = 0.0

  return normalize_distribution(scaled)


def apply_platt_scaling(
    p_true_raw: float,
    p_false_raw: float,
    temperature: float = 1.0,
    platt_bias: float = 0.0,
) -> tuple[float, float]:
  """Computes calibrated binary `P(True)` and raw logit gap `ln(P_T) - ln(P_F)`.

  Args:
    p_true_raw: Linear probability of the `True` symbol (`"A"`).
    p_false_raw: Linear probability of the `False` symbol (`"B"`).
    temperature: Calibration temperature `T > 0`.
    platt_bias: Intercept bias `b` in `sigma(logit_gap / T + b)`.

  Returns:
    `(calibrated_p_true, raw_logit_gap)`.
  """
  if temperature <= 0.0:
    raise ValueError(f"temperature must be > 0, got {temperature}.")
  pt = max(p_true_raw, _EPS)
  pf = max(p_false_raw, _EPS)
  logit_gap = math.log(pt) - math.log(pf)
  z = (logit_gap / temperature) + platt_bias
  if z >= 0.0:
    p_cal = 1.0 / (1.0 + math.exp(-z))
  else:
    exp_z = math.exp(z)
    p_cal = exp_z / (1.0 + exp_z)
  return p_cal, logit_gap


def normalized_shannon_entropy(probs: Sequence[float]) -> float:
  """Computes normalized Shannon entropy `H(p) / ln(K)` in `[0.0, 1.0]`."""
  k_count = len(probs)
  if k_count <= 1:
    return 0.0
  entropy = 0.0
  for p in probs:
    if p > _EPS:
      entropy -= p * math.log(min(p, 1.0))
  max_entropy = math.log(k_count)
  return max(0.0, min(1.0, entropy / max_entropy))


def jensen_shannon_divergence(
    dist_p: Mapping[str, float],
    dist_q: Mapping[str, float],
) -> float:
  """Computes normalized Jensen-Shannon divergence `D_JS(P || Q)` in `[0, 1]`."""
  keys = list(dist_p.keys())
  if not keys:
    return 0.0
  kl_pm = 0.0
  kl_qm = 0.0
  for k in keys:
    p = max(0.0, dist_p.get(k, 0.0))
    q = max(0.0, dist_q.get(k, 0.0))
    m = 0.5 * (p + q)
    if p > _EPS and m > _EPS:
      kl_pm += p * math.log(p / m)
    if q > _EPS and m > _EPS:
      kl_qm += q * math.log(q / m)
  js_nats = 0.5 * (kl_pm + kl_qm)
  return max(0.0, min(1.0, js_nats / math.log(2.0)))


def compute_score_metrics(
    calibrated_probs: Sequence[float],
    level_values: Sequence[float],
    point_probs: Sequence[float] | None = None,
    bimodal_min_peak: float = 0.20,
) -> tuple[float, float, float, float, float, bool, int]:
  """Computes expected score, Popoviciu ordinal confidence, and bimodality.

  Args:
    calibrated_probs: Calibrated probability distribution `[p_0, ..., p_{L-1}]`.
    level_values: Strictly increasing numeric rubric values `[v_0, ...,
      v_{L-1}]`.
    point_probs: Optional distribution used for the point expectation `score =
      sum(v_i * p_i)` (defaults to `calibrated_probs`).
    bimodal_min_peak: Minimum probability threshold for non-adjacent peaks to
      trigger `is_bimodal=True` (default `0.20`).

  Returns:
    Tuple of `(score, normalized_score, confidence, std_dev, entropy,
    is_bimodal, mode_level)`.
  """
  if len(calibrated_probs) != len(level_values):
    raise ValueError("calibrated_probs and level_values must have equal length.")
  probs_for_mean = (
      point_probs if point_probs is not None else calibrated_probs
  )
  mean_score = sum(
      v * p for v, p in zip(level_values, probs_for_mean, strict=True)
  )
  v_min = level_values[0]
  v_max = level_values[-1]
  span = v_max - v_min
  normalized_score = (
      max(0.0, min(1.0, (mean_score - v_min) / span)) if span > 0.0 else 0.0
  )

  # Ordinal dispersion around the calibrated distribution's mean
  cal_mean = sum(
      v * p for v, p in zip(level_values, calibrated_probs, strict=True)
  )
  variance = sum(
      p * ((v - cal_mean) ** 2)
      for v, p in zip(level_values, calibrated_probs, strict=True)
  )
  std_dev = math.sqrt(max(0.0, variance))

  # Popoviciu's inequality: max std_dev on [v_min, v_max] is (v_max - v_min) / 2
  max_std_dev = span / 2.0
  confidence = (
      max(0.0, min(1.0, 1.0 - (std_dev / max_std_dev)))
      if max_std_dev > 0.0
      else 1.0
  )

  entropy = normalized_shannon_entropy(calibrated_probs)
  mode_level = max(
      range(len(calibrated_probs)), key=lambda idx: calibrated_probs[idx]
  )

  # Bimodality detection: two non-adjacent levels (|i - j| >= 2) both >=
  # bimodal_min_peak with a lower-probability valley between them.
  is_bimodal = False
  num_levels = len(calibrated_probs)
  for i in range(num_levels):
    if calibrated_probs[i] < bimodal_min_peak:
      continue
    for j in range(i + 2, num_levels):
      if calibrated_probs[j] < bimodal_min_peak:
        continue
      valley = min(calibrated_probs[i + 1 : j])
      if valley < min(calibrated_probs[i], calibrated_probs[j]):
        is_bimodal = True
        break
    if is_bimodal:
      break

  return (
      mean_score,
      normalized_score,
      confidence,
      std_dev,
      entropy,
      is_bimodal,
      mode_level,
  )


def evaluate_calibration(
    distributions: Sequence[Mapping[str, float]],
    true_labels: Sequence[str],
    *,
    temperature: float = 1.0,
    num_bins: int = 10,
) -> tuple[float, float, float]:
  """Computes `(nll, ece, brier)` on labeled validation distributions."""
  if len(distributions) != len(true_labels) or not distributions:
    raise ValueError(
        "distributions and true_labels must be non-empty and equal length."
    )
  n_samples = len(distributions)
  total_nll = 0.0
  total_brier = 0.0
  bin_conf_sum = [0.0] * num_bins
  bin_acc_sum = [0.0] * num_bins
  bin_counts = [0] * num_bins

  for raw_dist, y_true in zip(distributions, true_labels, strict=True):
    cal_dist = apply_temperature_scaling(raw_dist, temperature=temperature)
    p_true = max(cal_dist.get(y_true, 0.0), _EPS)
    total_nll -= math.log(p_true)

    for k, p_k in cal_dist.items():
      target = 1.0 if k == y_true else 0.0
      total_brier += (p_k - target) ** 2

    pred_label, pred_conf = max(cal_dist.items(), key=lambda kv: kv[1])
    acc = 1.0 if pred_label == y_true else 0.0
    bin_idx = min(num_bins - 1, int(pred_conf * num_bins))
    bin_conf_sum[bin_idx] += pred_conf
    bin_acc_sum[bin_idx] += acc
    bin_counts[bin_idx] += 1

  ece = 0.0
  for idx in range(num_bins):
    if bin_counts[idx] > 0:
      avg_conf = bin_conf_sum[idx] / bin_counts[idx]
      avg_acc = bin_acc_sum[idx] / bin_counts[idx]
      ece += (bin_counts[idx] / n_samples) * abs(avg_acc - avg_conf)

  return total_nll / n_samples, ece, total_brier / n_samples


def fit_temperature(
    raw_distributions: Sequence[Mapping[str, float]],
    true_labels: Sequence[str],
    *,
    num_bins: int = 10,
    min_temp: float = 0.1,
    max_temp: float = 10.0,
    tol: float = 1e-4,
) -> CalibrationResult:
  """Fits optimal post-hoc temperature `T` by minimizing NLL via golden section."""
  raw_nll, raw_ece, raw_brier = evaluate_calibration(
      raw_distributions, true_labels, temperature=1.0, num_bins=num_bins
  )

  inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
  inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0
  a, b = min_temp, max_temp
  h = b - a
  c = a + inv_phi2 * h
  d = a + inv_phi * h
  fc = evaluate_calibration(
      raw_distributions, true_labels, temperature=c, num_bins=num_bins
  )[0]
  fd = evaluate_calibration(
      raw_distributions, true_labels, temperature=d, num_bins=num_bins
  )[0]

  for _ in range(64):
    if (b - a) <= tol:
      break
    if fc < fd:
      b = d
      d = c
      fd = fc
      h = inv_phi * h
      c = a + inv_phi2 * h
      fc = evaluate_calibration(
          raw_distributions, true_labels, temperature=c, num_bins=num_bins
      )[0]
    else:
      a = c
      c = d
      fc = fd
      h = inv_phi * h
      d = a + inv_phi * h
      fd = evaluate_calibration(
          raw_distributions, true_labels, temperature=d, num_bins=num_bins
      )[0]

  opt_t = 0.5 * (a + b)
  cal_nll, cal_ece, cal_brier = evaluate_calibration(
      raw_distributions, true_labels, temperature=opt_t, num_bins=num_bins
  )
  return CalibrationResult(
      optimal_temperature=opt_t,
      raw_nll=raw_nll,
      calibrated_nll=cal_nll,
      raw_ece=raw_ece,
      calibrated_ece=cal_ece,
      raw_brier=raw_brier,
      calibrated_brier=cal_brier,
  )
