"""Descriptive statistics, normality tests, capability and plots."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from os import PathLike
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import normal_ad


@dataclass(frozen=True)
class DescriptiveResult:
    n: int
    mean: float
    median: float
    std_dev: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    p10: float
    p50: float
    p90: float
    skewness: float
    kurtosis: float


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float


@dataclass(frozen=True)
class OneSampleTResult:
    n: int
    mean: float
    std_dev: float
    standard_error: float
    hypothesized_mean: float
    alternative: Literal["two-sided", "less", "greater"]
    confidence_level: float
    confidence_interval: tuple[float, float]
    statistic: float
    degrees_of_freedom: int
    p_value: float
    name: str

    def __str__(self) -> str:
        operators = {"two-sided": "not =", "less": "<", "greater": ">"}
        level = f"{self.confidence_level * 100:g}%"
        if self.alternative == "two-sided":
            interval_header = f"{level} CI"
            interval_top = ""
            interval_value = f"({self.confidence_interval[0]:.2f}, {self.confidence_interval[1]:.2f})"
        else:
            interval_top = f"{level} {'Lower' if self.alternative == 'greater' else 'Upper'}"
            interval_header = "Bound"
            bound = self.confidence_interval[0 if self.alternative == "greater" else 1]
            interval_value = f"{bound:.2f}"
        headers = ["Variable", "N", "Mean", "StDev", "SE Mean", interval_header, "T", "P"]
        values = [
            self.name, str(self.n), f"{self.mean:.2f}", f"{self.std_dev:.2f}",
            f"{self.standard_error:.2f}", interval_value, f"{self.statistic:.2f}", f"{self.p_value:.3f}",
        ]
        widths = [max(len(header), len(value)) for header, value in zip(headers, values)]
        widths[5] = max(widths[5], len(interval_top))

        def row(items: list[str]) -> str:
            return "  ".join(
                item.ljust(width) if index == 0 else item.rjust(width)
                for index, (item, width) in enumerate(zip(items, widths))
            ).rstrip()

        lines = [
            f"One-Sample T: {self.name}",
            "",
            f"Test of mu = {self.hypothesized_mean:g} vs {operators[self.alternative]} {self.hypothesized_mean:g}",
            "",
        ]
        if interval_top:
            lines.append(row(["", "", "", "", "", interval_top, "", ""]))
        lines.extend((row(headers), row(values)))
        relation = {
            "two-sided": "différente de",
            "less": "inférieure à",
            "greater": "supérieure à",
        }[self.alternative]
        decision = "permettent" if self.p_value < 1 - self.confidence_level else "ne permettent pas"
        lines.extend((
            "",
            f"Conclusion (seuil de {(1 - self.confidence_level) * 100:g} %) : "
            f"les données {decision} de conclure que la moyenne de {self.name} "
            f"est {relation} {self.hypothesized_mean}.",
        ))
        return "\n".join(lines)


@dataclass(frozen=True)
class RunsResult:
    runs: int
    n_above: int
    n_below: int
    statistic: float
    p_value: float


@dataclass(frozen=True)
class RunChartResult:
    median: float
    runs_about_median: int
    expected_runs_about_median: float
    longest_run_about_median: int
    p_clustering: float | None
    p_mixtures: float | None
    runs_up_down: int
    expected_runs_up_down: float | None
    longest_run_up_down: int
    p_trends: float | None
    p_oscillation: float | None


@dataclass(frozen=True)
class AndersonResult:
    statistic: float
    critical_values: tuple[float, ...]
    significance_levels: tuple[float, ...]


@dataclass(frozen=True)
class CapabilityPerformance:
    ppm_below_lsl: float | None
    ppm_above_usl: float | None
    ppm_total: float


@dataclass(frozen=True)
class CapabilityResult:
    lsl: float | None
    usl: float | None
    n: int
    mean: float
    std_within: float
    std_overall: float
    cp: float | None
    cpl: float | None
    cpu: float | None
    cpk: float
    pp: float | None
    ppl: float | None
    ppu: float | None
    ppk: float
    observed: CapabilityPerformance
    expected_within: CapabilityPerformance
    expected_overall: CapabilityPerformance


@dataclass(frozen=True)
class SigmaResult:
    dpo: float
    dpmo: float
    z_lt: float
    z_st: float


def sigma_level(
    dpo: float | None = None,
    *,
    defects: int | None = None,
    opportunities_per_unit: int | None = None,
    units: int | None = None,
) -> SigmaResult:
    """Convert a defect probability to long- and short-term sigma levels."""
    counts = (defects, opportunities_per_unit, units)
    if dpo is not None:
        if any(value is not None for value in counts):
            raise ValueError("provide either dpo or defects, opportunities_per_unit and units")
        if isinstance(dpo, bool) or not isinstance(dpo, Real) or not isfinite(dpo) or not 0 <= dpo <= 1:
            raise ValueError("dpo must be a finite number between 0 and 1")
        probability = float(dpo)
    else:
        if any(value is None for value in counts):
            raise ValueError("provide either dpo or defects, opportunities_per_unit and units")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in counts):
            raise ValueError("defects, opportunities_per_unit and units must be integers")
        if defects is None or opportunities_per_unit is None or units is None:
            raise ValueError("defects, opportunities_per_unit and units are required")
        if defects < 0 or opportunities_per_unit <= 0 or units <= 0:
            raise ValueError("defects must be non-negative and opportunities_per_unit and units must be positive")
        total = opportunities_per_unit * units
        if defects > total:
            raise ValueError("defects cannot exceed the total number of opportunities")
        probability = defects / total
    z_lt = float(stats.norm.isf(probability))
    return SigmaResult(probability, probability * 1_000_000, z_lt, z_lt + 1.5)


def _group_indices(groups: Iterable[str | Real], size: int) -> dict[str | Real, list[int]]:
    group_values = list(groups)
    if len(group_values) != size:
        raise ValueError("groups must have the same length as the data")
    grouped: dict[str | Real, list[int]] = {}
    for index, group in enumerate(group_values):
        if not (
            isinstance(group, str) and group.strip()
            or isinstance(group, Real) and not isinstance(group, bool) and isfinite(group)
        ):
            raise ValueError("groups must contain non-empty strings or finite numbers")
        grouped.setdefault(group, []).append(index)
    return grouped


class Analysis:
    """Analyze a one-dimensional sequence of finite observations."""

    def __init__(self, data: Iterable[float]):
        values = np.asarray(list(data), dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("data must be a non-empty one-dimensional sequence")
        if not np.isfinite(values).all():
            raise ValueError("data must contain only finite numbers")
        self._data = values.copy()
        self._name = "Data"

    @classmethod
    def from_excel(
        cls, path: str | PathLike[str], column: str, *, sheet_name: str | int = 0
    ) -> "Analysis":
        """Load a numeric column from an Excel worksheet in its original order."""
        frame = pd.read_excel(path, sheet_name=sheet_name, keep_default_na=False, na_values=[""])
        if column not in frame.columns:
            raise ValueError(f"column {column!r} not found in Excel worksheet")
        values = pd.to_numeric(frame[column].dropna(), errors="raise")
        analysis = cls(values)
        analysis._name = column
        return analysis

    def _require(self, minimum: int, nonconstant: bool = False) -> None:
        if self._data.size < minimum:
            raise ValueError(f"at least {minimum} observations are required")
        if nonconstant and np.ptp(self._data) == 0:
            raise ValueError("data must not be constant")

    def descriptive_statistics(self) -> DescriptiveResult:
        self._require(2)
        d = self._data
        constant = np.ptp(d) == 0
        return DescriptiveResult(
            n=int(d.size),
            mean=float(np.mean(d)),
            median=float(np.median(d)),
            std_dev=float(np.std(d, ddof=1)),
            minimum=float(np.min(d)),
            maximum=float(np.max(d)),
            q1=float(np.percentile(d, 25)),
            q3=float(np.percentile(d, 75)),
            p10=float(np.percentile(d, 10)),
            p50=float(np.percentile(d, 50)),
            p90=float(np.percentile(d, 90)),
            skewness=float("nan") if constant else float(stats.skew(d)),
            kurtosis=float("nan") if constant else float(stats.kurtosis(d)),
        )

    def run_test(self) -> RunsResult:
        median = np.median(self._data)
        values = self._data[self._data != median]
        above = values > median
        n_above = int(np.sum(above))
        n_below = int(values.size - n_above)
        if n_above == 0 or n_below == 0 or values.size < 3:
            raise ValueError("run test requires at least three non-median observations on both sides")
        runs = int(1 + np.count_nonzero(above[1:] != above[:-1]))
        z, p_value = runstest_1samp(values, cutoff=median)
        return RunsResult(runs, n_above, n_below, float(z), float(p_value))

    def run_chart_statistics(self) -> RunChartResult:
        """Return run-chart statistics for individual observations."""
        values = self._data
        n = len(values)
        median = float(np.median(values))
        above = values > median
        lengths = np.diff(np.r_[0, np.flatnonzero(above[1:] != above[:-1]) + 1, n])
        runs = len(lengths)
        m = int(above.sum())
        below = n - m
        expected = 1 + 2 * m * below / n
        variance = 2 * m * below * (2 * m * below - n) / (n**2 * (n - 1)) if n > 1 else 0
        p_clustering = p_mixtures = None
        if variance > 0:
            z = (runs - expected) / np.sqrt(variance)
            p_clustering = float(stats.norm.cdf(z))
            p_mixtures = float(stats.norm.sf(z))

        directions = np.diff(values) > 0
        direction_lengths = np.diff(
            np.r_[0, np.flatnonzero(directions[1:] != directions[:-1]) + 1, len(directions)]
        ) if directions.size else np.array([], dtype=int)
        runs_up_down = len(direction_lengths)
        if runs_up_down:
            direction_lengths[0] += 1
        longest_up_down = int(direction_lengths.max()) if runs_up_down else 0
        expected_up_down = (2 * n - 1) / 3 if n >= 3 else None
        p_trends = p_oscillation = None
        if expected_up_down is not None:
            z = (runs_up_down - expected_up_down) / np.sqrt((16 * n - 29) / 90)
            p_trends = float(stats.norm.cdf(z))
            p_oscillation = float(stats.norm.sf(z))
        return RunChartResult(
            median, runs, float(expected), int(lengths.max()), p_clustering, p_mixtures,
            runs_up_down, expected_up_down, longest_up_down, p_trends, p_oscillation,
        )

    def one_sample_t_test(
        self,
        hypothesized_mean: float,
        *,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        confidence_level: float = 0.95,
        name: str | None = None,
    ) -> OneSampleTResult:
        """Test the population mean and calculate its confidence interval."""
        self._require(2, nonconstant=True)
        if (isinstance(hypothesized_mean, bool) or not isinstance(hypothesized_mean, Real)
                or not isfinite(hypothesized_mean)):
            raise ValueError("hypothesized_mean must be a finite number")
        if alternative not in ("two-sided", "less", "greater"):
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
        if (isinstance(confidence_level, bool) or not isinstance(confidence_level, Real)
                or not isfinite(confidence_level) or not 0 < confidence_level < 1):
            raise ValueError("confidence_level must be a finite number between 0 and 1, exclusive")
        if name is not None and (not isinstance(name, str) or not name.strip() or "\n" in name or "\r" in name):
            raise ValueError("name must be a non-empty single-line string")
        n = int(self._data.size)
        mean = float(np.mean(self._data))
        std_dev = float(np.std(self._data, ddof=1))
        standard_error = std_dev / np.sqrt(n)
        if not isfinite(standard_error) or standard_error <= 0 or not isfinite(mean):
            raise ValueError("sample mean and standard error must be finite, and standard error must be positive")
        result = stats.ttest_1samp(self._data, hypothesized_mean, alternative=alternative)
        interval = result.confidence_interval(confidence_level=confidence_level)
        return OneSampleTResult(
            n, mean, std_dev, float(standard_error), float(hypothesized_mean),
            alternative, float(confidence_level), (float(interval.low), float(interval.high)),
            float(result.statistic), n - 1, float(result.pvalue), self._name if name is None else name,
        )

    def shapiro_test(self) -> TestResult:
        self._require(3, nonconstant=True)
        result = stats.shapiro(self._data)
        return TestResult(float(result.statistic), float(result.pvalue))

    def anderson_test(self) -> AndersonResult:
        self._require(5, nonconstant=True)
        result = stats.anderson(self._data, dist="norm")
        return AndersonResult(
            float(result.statistic),
            tuple(float(value) for value in result.critical_values),
            tuple(float(value) for value in result.significance_level),
        )

    def capability(self, lsl: float | None = None, usl: float | None = None) -> CapabilityResult:
        """Calculate normal capability indices for individual observations."""
        self._require(2, nonconstant=True)
        if lsl is None and usl is None:
            raise ValueError("at least one specification limit is required")
        if any(limit is not None and not np.isfinite(limit) for limit in (lsl, usl)):
            raise ValueError("specification limits must be finite")
        if lsl is not None and usl is not None and lsl >= usl:
            raise ValueError("lsl must be less than usl")

        mean = float(np.mean(self._data))
        std_overall = float(np.std(self._data, ddof=1))
        std_within = float(np.mean(np.abs(np.diff(self._data))) / 1.128)

        def indices(sigma: float) -> tuple[float | None, float | None, float | None, float]:
            lower = float((mean - lsl) / (3 * sigma)) if lsl is not None else None
            upper = float((usl - mean) / (3 * sigma)) if usl is not None else None
            spread = float((usl - lsl) / (6 * sigma)) if lsl is not None and usl is not None else None
            available = [value for value in (lower, upper) if value is not None]
            return spread, lower, upper, min(available)

        def performance(below: float | None, above: float | None) -> CapabilityPerformance:
            return CapabilityPerformance(below, above, (below or 0.0) + (above or 0.0))

        def expected(sigma: float) -> CapabilityPerformance:
            below = float(stats.norm.cdf(lsl, loc=mean, scale=sigma) * 1_000_000) if lsl is not None else None
            above = float(stats.norm.sf(usl, loc=mean, scale=sigma) * 1_000_000) if usl is not None else None
            return performance(below, above)

        cp, cpl, cpu, cpk = indices(std_within)
        pp, ppl, ppu, ppk = indices(std_overall)
        below = float(np.mean(self._data < lsl) * 1_000_000) if lsl is not None else None
        above = float(np.mean(self._data > usl) * 1_000_000) if usl is not None else None
        return CapabilityResult(
            float(lsl) if lsl is not None else None,
            float(usl) if usl is not None else None,
            int(self._data.size), mean, std_within, std_overall,
            cp, cpl, cpu, cpk, pp, ppl, ppu, ppk,
            performance(below, above), expected(std_within), expected(std_overall),
        )

    def capability_analysis_normal(
        self, lsl: float | None = None, usl: float | None = None, *,
        name: str | None = None, bins: int | Literal["auto"] = "auto",
    ) -> Figure:
        """Plot normal capability analysis for individual observations."""
        if bins != "auto" and (isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1):
            raise ValueError("bins must be 'auto' or a positive integer")
        result = self.capability(lsl=lsl, usl=usl)
        figure = plt.figure(figsize=(12, 8), layout="constrained")
        grid = figure.add_gridspec(2, 3, width_ratios=(1.35, 2.8, 1.35), height_ratios=(3, 0.9))
        process = figure.add_subplot(grid[0, 0])
        ax = figure.add_subplot(grid[0, 1])
        capability = figure.add_subplot(grid[0, 2])
        performance_grid = grid[1, :].subgridspec(1, 3)
        performance_axes = [figure.add_subplot(performance_grid[0, i]) for i in range(3)]
        title = f"Process Capability of {name}" if name else "Process Capability"
        figure.suptitle(title, fontsize=16)
        ax.hist(self._data, bins=bins, density=True, color="#b8bec7", edgecolor="#555555")
        limits = [limit for limit in (result.lsl, result.usl) if limit is not None]
        lo = min(float(np.min(self._data)), result.mean - 4 * result.std_overall,
                 result.mean - 4 * result.std_within, *limits)
        hi = max(float(np.max(self._data)), result.mean + 4 * result.std_overall,
                 result.mean + 4 * result.std_within, *limits)
        x = np.linspace(lo, hi, 500)
        ax.plot(x, stats.norm.pdf(x, result.mean, result.std_within), color="#b24542", label="Within")
        ax.plot(x, stats.norm.pdf(x, result.mean, result.std_overall), color="#333333",
                linestyle="--", label="Overall")
        for label, limit in (("LSL", result.lsl), ("USL", result.usl)):
            if limit is not None:
                ax.axvline(limit, color="#b24542", linestyle="--", linewidth=1.2)
                ax.text(limit, 1.01, label, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", color="#b24542", fontsize="small")
        ax.set(xlabel=name or "Value", ylabel="Density")
        ax.legend(fontsize="small", loc="upper right")

        def display(value: float | None, digits: int = 3) -> str:
            return "N/A" if value is None else f"{value:.{digits}f}"

        box = {"boxstyle": "square,pad=0.55", "facecolor": "white", "edgecolor": "#777777"}
        process.axis("off")
        process.text(
            0, 1,
            f"Process Data\n\nLSL             {display(result.lsl)}\nUSL             {display(result.usl)}\n"
            f"Sample N        {result.n}\nMean            {result.mean:.4f}\n"
            f"StDev (Within)  {result.std_within:.4f}\nStDev (Overall) {result.std_overall:.4f}",
            va="top", fontsize=9, fontfamily="monospace", bbox=box,
        )
        capability.axis("off")
        capability.text(
            0, 1,
            f"Potential (Within)\n\nCp   {display(result.cp)}\nCPL  {display(result.cpl)}\n"
            f"CPU  {display(result.cpu)}\nCpk  {result.cpk:.3f}\n\n"
            f"Overall Capability\n\nPp   {display(result.pp)}\nPPL  {display(result.ppl)}\n"
            f"PPU  {display(result.ppu)}\nPpk  {result.ppk:.3f}",
            va="top", fontsize=9, fontfamily="monospace", bbox=box,
        )
        for panel, label, item in zip(
            performance_axes,
            ("Observed Performance", "Expected Within", "Expected Overall"),
            (result.observed, result.expected_within, result.expected_overall),
        ):
            panel.axis("off")
            panel.text(
                0.05, 0.8,
                f"{label}\n\nPPM < LSL  {display(item.ppm_below_lsl, 2)}\n"
                f"PPM > USL  {display(item.ppm_above_usl, 2)}\nPPM Total  {display(item.ppm_total, 2)}",
                va="top", fontsize=9, fontfamily="monospace", bbox=box,
            )
        return figure

    def histogram(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        ax.hist(self._data, bins="auto", edgecolor="black")
        ax.set(title="Histogram", xlabel="Value", ylabel="Frequency")
        return ax

    def boxplot(
        self, ax: Axes | None = None, *, name: str | None = None,
        groups: Iterable[str | Real] | None = None, group_name: str | None = None,
        reference_lines: Iterable[float] | None = None,
    ) -> Axes:
        """Plot quartiles and 1.5-IQR whiskers, optionally grouped by category."""
        reference_values = [] if reference_lines is None else list(reference_lines)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
               for value in reference_values):
            raise ValueError("reference lines must be finite numbers")
        if groups is None:
            samples = [self._data]
            labels = []
        else:
            grouped = _group_indices(groups, self._data.size)
            labels = [str(group) for group in grouped]
            samples = [self._data[indices] for indices in grouped.values()]

        statistics = []
        for sample in samples:
            q1, median, q3 = np.quantile(sample, [0.25, 0.5, 0.75], method="weibull")
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            inside = sample[(sample >= lower) & (sample <= upper)]
            outliers = sample[(sample < lower) | (sample > upper)]
            statistics.append({
                "q1": q1, "med": median, "q3": q3,
                "whislo": np.min(inside), "whishi": np.max(inside),
                "fliers": outliers,
            })
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        ax.bxp(
            statistics, showfliers=True, patch_artist=True, widths=0.3,
            boxprops={"facecolor": "#86add8", "edgecolor": "#333333"},
            medianprops={"color": "#333333"},
            whiskerprops={"color": "#333333"},
            capprops={"color": "#333333"},
            flierprops={"marker": "*", "markeredgecolor": "#333333", "markerfacecolor": "#333333"},
        )
        for value in reference_values:
            ax.axhline(value, color="#bd3d37", linestyle="--", linewidth=1.2, zorder=1)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels)
        ax.set(title=f"Boxplot of {name}" if name else "Boxplot",
               ylabel=name or "Value", xlabel=group_name or "")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        return ax

    def run_chart(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            figure, ax = plt.subplots(figsize=(10, 6))
            figure.subplots_adjust(bottom=0.33)
        result = self.run_chart_statistics()
        ax.plot(
            np.arange(1, self._data.size + 1), self._data,
            marker="o", markersize=4, color="#345b9a", linewidth=1.2,
        )
        ax.axhline(result.median, color="#666666", linestyle="--", label=f"Median = {result.median:.4g}")
        ax.set(title="Run Chart", xlabel="Observation", ylabel="Value")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.legend()

        def display(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.4f}"

        median_summary = (
            f"Runs about median: {result.runs_about_median}\n"
            f"Expected runs: {result.expected_runs_about_median:.4f}\n"
            f"Longest run: {result.longest_run_about_median}\n"
            f"P-value, clustering: {display(result.p_clustering)}\n"
            f"P-value, mixtures: {display(result.p_mixtures)}"
        )
        direction_summary = (
            f"Runs up or down: {result.runs_up_down}\n"
            f"Expected runs: {display(result.expected_runs_up_down)}\n"
            f"Longest run: {result.longest_run_up_down}\n"
            f"P-value, trends: {display(result.p_trends)}\n"
            f"P-value, oscillation: {display(result.p_oscillation)}"
        )
        for x, summary in ((0, median_summary), (0.53, direction_summary)):
            ax.text(x, -0.17, summary, transform=ax.transAxes, va="top", fontsize="small",
                    bbox={"boxstyle": "square,pad=0.4", "facecolor": "white", "edgecolor": "#999999"})
        if np.any(np.diff(self._data) == 0):
            ax.text(0, -0.44, "Ties can affect the up/down approximation.",
                    transform=ax.transAxes, va="top", fontsize="small")
        return ax

    def normality_test(self, ax: Axes | None = None) -> Axes:
        """Plot normality and, for five or more observations, the Anderson-Darling test."""
        self._require(2, nonconstant=True)
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6), layout="constrained")

        values = np.sort(self._data)
        n = values.size
        mean = float(np.mean(values))
        std_dev = float(np.std(values, ddof=1))
        probabilities = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        quantiles = stats.norm.ppf(probabilities)
        ax.scatter(values, quantiles, s=30, color="#bd3d37", zorder=3, label="Observations")

        percent_ticks = np.array([0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9])
        tick_positions = stats.norm.ppf(percent_ticks / 100)
        z_min = min(float(tick_positions[1]), float(quantiles[0]) - 0.15)
        z_max = max(float(tick_positions[-2]), float(quantiles[-1]) + 0.15)
        z = np.linspace(z_min, z_max, 300)
        fitted = mean + std_dev * z
        ax.plot(fitted, z, color="#345b9a", linewidth=1.5, label="Normal fit")
        if n >= 3:
            margin = stats.norm.ppf(0.975) * std_dev * np.sqrt(1 / n + z**2 / (2 * (n - 1)))
            ax.plot(fitted - margin, z, color="#345b9a", linewidth=1, linestyle="--", label="Approx. 95% pointwise CI")
            ax.plot(fitted + margin, z, color="#345b9a", linewidth=1, linestyle="--")

        summary = f"N       {n}\nMean    {mean:.4g}\nStDev   {std_dev:.4g}"
        if n >= 5:
            statistic, p_value = normal_ad(values)
            summary += f"\nAD      {statistic:.4g}\nP-value {p_value:.4g}"
        ax.text(
            1.02, 0.98, summary, transform=ax.transAxes, va="top",
            fontfamily="monospace", fontsize="small",
            bbox={"boxstyle": "square,pad=0.4", "facecolor": "white", "edgecolor": "#999999"},
        )
        ax.set_yticks(tick_positions, [f"{value:g}" for value in percent_ticks])
        ax.set_ylim(z_min, z_max)
        ax.set(title="Normal Probability Plot", xlabel="Value", ylabel="Percent")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="lower right", fontsize="small")
        return ax

