"""Preliminary guidance for selecting statistical methods."""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal


@dataclass(frozen=True)
class MethodCandidate:
    name: str
    available: bool
    note: str


@dataclass(frozen=True)
class MethodSelection:
    candidates: tuple[MethodCandidate, ...]
    missing: tuple[str, ...]
    warnings: tuple[str, ...]


def xy_table_choice() -> str:
    """Return an overview generated from the method-selection guidance."""
    scenarios = (
        ("Continuous Y: one sample, normal", select_statistical_method(
            y_type="continuous", x_type=None, objective="compare", samples=1, distribution="normal")),
        ("Continuous Y: one sample, non-normal", select_statistical_method(
            y_type="continuous", x_type=None, objective="compare", samples=1, distribution="non-normal")),
        ("Continuous Y: paired samples, normal differences", select_statistical_method(
            y_type="continuous", x_type="discrete", objective="compare", samples=2, paired=True, distribution="normal")),
        ("Continuous Y: paired samples, non-normal differences", select_statistical_method(
            y_type="continuous", x_type="discrete", objective="compare", samples=2, paired=True, distribution="non-normal")),
        ("Continuous Y: two independent groups, normal", select_statistical_method(
            y_type="continuous", x_type="discrete", objective="compare", samples=2, paired=False, distribution="normal")),
        ("Continuous Y: two independent groups, non-normal", select_statistical_method(
            y_type="continuous", x_type="discrete", objective="compare", samples=2, paired=False, distribution="non-normal")),
        ("Continuous Y: three or more independent groups, normal", select_statistical_method(
            y_type="continuous", x_type="discrete", objective="compare", samples=3, paired=False, distribution="normal")),
        ("Continuous Y: three or more independent groups, non-normal", select_statistical_method(
            y_type="continuous", x_type="discrete", objective="compare", samples=3, paired=False, distribution="non-normal")),
        ("Variance: one normal sample", select_statistical_method(
            y_type="continuous", x_type=None, objective="variance", samples=1, distribution="normal")),
        ("Variance: two independent normal groups", select_statistical_method(
            y_type="continuous", x_type="discrete", objective="variance", samples=2, paired=False, distribution="normal")),
        ("Variance: three or more independent normal groups", select_statistical_method(
            y_type="continuous", x_type="discrete", objective="variance", samples=3, paired=False, distribution="normal")),
        ("Continuous X and Y: association", select_statistical_method(
            y_type="continuous", x_type="continuous", objective="correlation")),
        ("Continuous X and Y: prediction", select_statistical_method(
            y_type="continuous", x_type="continuous", objective="regression")),
        ("Two categorical variables: independent observations", select_statistical_method(
            y_type="discrete", x_type="discrete", objective="association", paired=False)),
        ("Binary Y: one sample", select_statistical_method(
            y_type="discrete", x_type=None, objective="proportion", samples=1, binary=True)),
        ("Binary Y: two independent groups", select_statistical_method(
            y_type="discrete", x_type="discrete", objective="proportion", samples=2, paired=False, binary=True)),
    )
    headers = ("Situation", "Candidate", "In py6sigma")
    rows = tuple(
        (situation, candidate.name, "Yes" if candidate.available else "No")
        for situation, selection in scenarios
        for candidate in selection.candidates
    )
    widths = [max(len(row[i]) for row in (headers, *rows)) for i in range(len(headers))]

    def format_row(row: tuple[str, ...]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()

    return "\n".join((
        "Statistical method guide - illustrative situations",
        "",
        format_row(headers),
        "-+-".join("-" * width for width in widths),
        *(format_row(row) for row in rows),
        "",
        "Candidates are not automatic decisions. Check design, estimand, assumptions and sample size.",
        "Use select_statistical_method() for details, missing information and limitations.",
    ))


def select_statistical_method(
    *,
    y_type: Literal["continuous", "discrete"],
    x_type: Literal["continuous", "discrete"] | None,
    objective: Literal["compare", "variance", "correlation", "regression", "association", "proportion"],
    samples: int | None = None,
    paired: bool | None = None,
    distribution: Literal["normal", "non-normal", "unknown"] = "unknown",
    binary: bool | None = None,
) -> MethodSelection:
    """Suggest preliminary candidates; never execute a test or infer its assumptions."""
    if y_type not in ("continuous", "discrete"):
        raise ValueError("y_type must be 'continuous' or 'discrete'")
    if x_type not in ("continuous", "discrete", None):
        raise ValueError("x_type must be 'continuous', 'discrete', or None")
    if objective not in ("compare", "variance", "correlation", "regression", "association", "proportion"):
        raise ValueError("invalid objective")
    if distribution not in ("normal", "non-normal", "unknown"):
        raise ValueError("invalid distribution")
    if samples is not None and (isinstance(samples, bool) or not isinstance(samples, Integral) or samples < 1):
        raise ValueError("samples must be a positive integer or None")
    if paired is not None and not isinstance(paired, bool):
        raise ValueError("paired must be a boolean or None")
    if binary is not None and not isinstance(binary, bool):
        raise ValueError("binary must be a boolean or None")
    if samples == 1 and paired is not None:
        raise ValueError("paired is not applicable to one sample")

    candidates: list[MethodCandidate] = []
    missing: list[str] = []
    warnings = [
        "Preliminary guidance only; verify the study design, estimand, assumptions and sample size.",
        "Discrete counts are not necessarily categorical; classify the variable by its statistical role.",
    ]

    def add(name: str, note: str, available: bool = False) -> None:
        candidates.append(MethodCandidate(name, available, note))

    if objective in ("compare", "variance") and y_type == "continuous" and x_type in ("discrete", None):
        if samples is None:
            missing.append("Specify the number of samples or groups.")
        elif samples > 1 and x_type is None:
            missing.append("Specify the grouping variable type; this guide handles discrete groups.")
        elif samples > 1 and paired is None:
            missing.append("Specify whether observations are paired or independent.")
        elif samples > 2 and paired:
            warnings.append("Repeated measures with more than two groups require a dedicated analysis.")
        elif objective == "variance" and samples > 1 and paired:
            warnings.append("Variance comparisons for paired observations require a dedicated method.")
        elif objective == "variance" and samples == 1:
            if distribution == "normal":
                add("1-Variance test", "Compare one population variance to a specified value; normality is important.")
            elif distribution == "unknown":
                missing.append("Assess the distribution and specify the reference variance.")
            else:
                warnings.append("A one-variance chi-square test is sensitive to non-normality; seek a suitable alternative.")
        elif objective == "variance":
            add("Test for equal variances (Levene)", "For independent groups; robust to moderate non-normality.")
            if distribution == "normal":
                add("2-Variance test (F)" if samples == 2 else "Bartlett test",
                    "For independent normally distributed groups; sensitive to non-normality.")
        elif distribution == "unknown":
            missing.append("Assess the relevant distribution: paired differences, or observations within each group.")
        elif samples == 1:
            if distribution == "normal":
                add("1-Sample t test", "Tests the population mean against a specified value.", available=True)
            else:
                add("1-Sample Sign test", "Tests a median against a specified value; does not test the mean.")
        elif samples == 2 and paired:
            if distribution == "normal":
                add("Paired t test", "Tests the mean of paired differences; assess their distribution.")
            else:
                add("1-Sample Sign test on paired differences", "Tests the median paired difference; not the mean.")
                add("Wilcoxon signed-rank test", "For paired differences; requires suitable symmetry for a location interpretation.")
        elif samples == 2:
            if distribution == "normal":
                add("2-Sample t test (Welch)", "Compares means of independent groups without assuming equal variances.")
            else:
                add("Mann-Whitney test", "For independent groups; a median interpretation requires additional shape assumptions.")
        elif distribution == "normal":
            add("One-Way ANOVA", "Compares group means; check independence and variance assumptions.")
        else:
            add("Mood's median test", "Compares group medians; check independence and limitations.")
            add("Kruskal-Wallis test", "Compares distributions of independent groups; median interpretation requires assumptions.")
    elif y_type == "continuous" and x_type == "continuous" and objective in ("correlation", "regression"):
        if objective == "correlation":
            add("Correlation analysis", "Pearson measures linear association; Spearman measures rank association.")
        else:
            add("Simple linear regression", "Check linearity, residuals, independence and variance assumptions.")
        warnings.append("Association does not establish causation; normality of X or Y alone is not the regression criterion.")
    elif y_type == "discrete" and objective in ("association", "proportion") and x_type in ("discrete", None):
        if objective == "association":
            if x_type is None:
                missing.append("Specify a second categorical variable.")
            elif paired is None:
                missing.append("Specify whether the categorical observations are paired or independent.")
            elif paired:
                warnings.append("Paired categorical data need a dedicated method, such as McNemar's test for binary pairs.")
            else:
                add("Chi-square test of independence", "For categorical variables and independent observations; check expected counts.")
        elif binary is None:
            missing.append("Specify whether the outcome has exactly two categories.")
        elif not binary:
            warnings.append("A proportion test requires a clearly defined binary outcome.")
        elif samples is None:
            missing.append("Specify the number of samples or groups.")
        elif samples == 1:
            add("1-Proportion test", "Compare one binary proportion with a specified value; check approximation requirements.")
        elif x_type is None:
            missing.append("Specify a discrete grouping variable.")
        elif paired is None:
            missing.append("Specify whether the binary observations are paired or independent.")
        elif paired:
            warnings.append("Paired binary proportions need a dedicated paired analysis.")
        elif samples == 2:
            add("2-Proportion test", "Compare independent binary proportions; check approximation requirements.")
        else:
            warnings.append("More than two proportions require a suitable multi-group analysis.")
    else:
        warnings.append("This combination is outside the scope of this guide; consult a statistician.")
    return MethodSelection(tuple(candidates), tuple(missing), tuple(warnings))
