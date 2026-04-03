from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VariantInput:
    name: str
    visitors: int
    conversions: int
    revenue_sum: float | None = None
    revenue_sum_squares: float | None = None
    is_control: bool = False


@dataclass
class GuardrailInput:
    name: str
    control: float
    treatment: float
    direction: str = "decrease"
    max_relative_change: float = 0.0


@dataclass
class DecisionThresholds:
    bayes_prob_beats_control: float = 0.95
    max_expected_loss: float = 0.001


@dataclass
class DecisionPolicy:
    enabled: bool = False
    thresholds: DecisionThresholds | None = None
    source: str = "none"


@dataclass
class AnalysisInput:
    experiment_name: str
    method: str
    variants: list[VariantInput]
    primary_metric: str = "conversion_rate"
    alpha: float = 0.05
    look_index: int = 1
    max_looks: int = 1
    information_fraction: float | None = None
    guardrails: list[GuardrailInput] = field(default_factory=list)
    decision_policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    random_seed: int = 7
    samples: int = 50000
    input_interpretation: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisSettings:
    primary_metric: str
    alpha: float
    look_index: int
    max_looks: int
    information_fraction: float | None
    samples: int
    random_seed: int
    priors: dict[str, Any]
    decision_policy: dict[str, Any]
    input_interpretation: dict[str, Any]


@dataclass
class GuardrailResult:
    name: str
    direction: str
    control: float
    treatment: float
    relative_change: float
    max_relative_change: float
    passed: bool
    reason: str


@dataclass
class ComparisonResult:
    treatment: str
    control: str
    metric: str
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    p_value: float | None = None
    alpha_spent: float | None = None
    significant: bool | None = None
    probability_beats_control: float | None = None
    expected_loss: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass
class Recommendation:
    action: str
    rationale: str
    decision_confidence: float
    next_best_action: str
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class SrmResult:
    passed: bool
    p_value: float
    observed: list[int]
    expected: list[float]
    reason: str


@dataclass
class AnalysisResult:
    experiment_name: str
    method: str
    analysis_settings: AnalysisSettings
    control_variant: str
    comparisons: list[ComparisonResult]
    guardrails_passed: bool
    guardrails: list[GuardrailResult]
    srm: SrmResult
    recommendation: Recommendation | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
