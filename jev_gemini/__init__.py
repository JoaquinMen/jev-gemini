"""`jev_gemini`: Calibrated System-One primitives (`Noul`, `Choice`, `Score`) on Gemini Flash."""

from jev_gemini.calibration import apply_platt_scaling
from jev_gemini.calibration import apply_temperature_scaling
from jev_gemini.calibration import CalibrationResult
from jev_gemini.calibration import compute_score_metrics
from jev_gemini.calibration import evaluate_calibration
from jev_gemini.calibration import fit_temperature
from jev_gemini.calibration import jensen_shannon_divergence
from jev_gemini.calibration import normalized_shannon_entropy
from jev_gemini.client import GeminiSystemOneClient
from jev_gemini.client import UnsupportedBackendError
from jev_gemini.extractor import extract_decision_step_distribution
from jev_gemini.extractor import extract_monte_carlo_distribution
from jev_gemini.extractor import ExtractedStepResult
from jev_gemini.extractor import MissingLogprobsError
from jev_gemini.extractor import ThoughtTruncationError
from jev_gemini.langchain import GeminiJevClassifier
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

__all__ = [
    "Answer",
    "CalibrationResult",
    "Choice",
    "ChoiceAnswer",
    "ExtractedStepResult",
    "GeminiJevClassifier",
    "GeminiSystemOneClient",
    "JSONContent",
    "MissingLogprobsError",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
    "StepDiagnostics",
    "SystemOneResponse",
    "ThoughtTruncationError",
    "UnsupportedBackendError",
    "apply_platt_scaling",
    "apply_temperature_scaling",
    "compute_score_metrics",
    "evaluate_calibration",
    "extract_decision_step_distribution",
    "extract_monte_carlo_distribution",
    "fit_temperature",
    "jensen_shannon_divergence",
    "normalized_shannon_entropy",
]
