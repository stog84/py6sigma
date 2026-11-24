"""Statistical tools for Lean Six Sigma."""

from matplotlib.pyplot import show

from .analysis import Analysis, AndersonResult, CapabilityPerformance, CapabilityResult, DescriptiveResult, OneSampleTResult, RunChartResult, RunsResult, SigmaResult, TestResult, sigma_level
from .pareto import pareto_chart
from .method_selection import MethodCandidate, MethodSelection, select_statistical_method, xy_table_choice
from .scatter import scatter_plot

__all__ = [
    "Analysis",
    "AndersonResult",
    "CapabilityPerformance",
    "CapabilityResult",
    "DescriptiveResult",
    "OneSampleTResult",
    "MethodCandidate",
    "MethodSelection",
    "RunChartResult",
    "RunsResult",
    "SigmaResult",
    "TestResult",
    "pareto_chart",
    "scatter_plot",
    "select_statistical_method",
    "xy_table_choice",
    "sigma_level",
    "show",
]
