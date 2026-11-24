"""Pareto chart for categorical observations or summarized counts."""

from collections.abc import Iterable, Mapping
from math import isfinite
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.textpath import TextPath
from matplotlib.transforms import ScaledTranslation


def pareto_chart(
    data: Iterable[str | Real] | Mapping[str | Real, Real],
    frequencies: Iterable[Real] | None = None,
    *,
    combine_after: float | None = 95,
    name: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot category counts and cumulative percentages in Pareto order."""
    if combine_after is not None:
        if (
            isinstance(combine_after, bool)
            or not isinstance(combine_after, Real)
            or not isfinite(combine_after)
            or not 0 < combine_after <= 100
        ):
            raise ValueError("combine_after must be a finite percentage greater than 0 and at most 100")
        threshold = float(combine_after)
        if threshold < 1:
            threshold *= 100
    else:
        threshold = None

    if isinstance(data, Mapping):
        if frequencies is not None:
            raise ValueError("frequencies cannot be supplied with a mapping")
        labels = list(data.keys())
        counts = list(data.values())
    else:
        labels = list(data)
        counts = [1] * len(labels) if frequencies is None else list(frequencies)
    if not labels or len(labels) != len(counts):
        raise ValueError("data and frequencies must be non-empty and have the same length")

    totals: dict[str | Real, int] = {}
    for label, count in zip(labels, counts):
        if not (
            isinstance(label, str) and label.strip()
            or isinstance(label, Real) and not isinstance(label, bool) and isfinite(label)
        ):
            raise ValueError("categories must be non-empty strings or finite numbers")
        if (
            isinstance(count, bool)
            or not isinstance(count, Real)
            or not isfinite(count)
            or count < 0
            or count != int(count)
        ):
            raise ValueError("frequencies must be finite non-negative integers")
        totals[label] = totals.get(label, 0) + int(count)

    ordered = sorted(totals.items(), key=lambda item: -item[1])
    total = sum(totals.values())
    if total == 0:
        raise ValueError("at least one frequency must be positive")

    if threshold is not None:
        cumulative = 0
        for index, (_, count) in enumerate(ordered):
            cumulative += count
            if cumulative / total * 100 > threshold:
                remaining = ordered[index + 1:]
                if remaining:
                    other_label = "Other"
                    while other_label in totals:
                        other_label += " (combined)"
                    ordered = ordered[:index + 1] + [(other_label, sum(count for _, count in remaining))]
                break

    labels = [str(label) for label, _ in ordered]
    counts = [count for _, count in ordered]
    percentages = [count / total * 100 for count in counts]
    cumulative_percentages = np.cumsum(percentages)
    cumulative_percentages[-1] = 100.0
    positions = np.arange(len(labels))
    vertical_labels = len(labels) > 6 or max(map(len, labels)) > 12
    label_height = max(
        (TextPath((0, 0), label, size=10).get_extents().width for label in labels),
        default=0,
    ) if vertical_labels else 12
    table_offset = label_height + 18

    if ax is None:
        bottom_inches = (table_offset + 55) / 72
        height = max(6, bottom_inches + 4)
        width = max(10, min(22, len(labels) * 0.8 + 2))
        figure, ax = plt.subplots(figsize=(width, height))
        figure.subplots_adjust(left=0.15, right=0.88, bottom=bottom_inches / height)
    ax.bar(positions, counts, color="#5286bd", edgecolor="black", linewidth=0.7)
    ax.set_xticks(positions, labels, rotation=90 if vertical_labels else 0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(0, total)
    ax.set_ylabel("Count")
    ax.set_title(f"Pareto Chart of {name}" if name else "Pareto Chart")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    percent_ax = ax.twinx()
    percent_ax.plot(positions, cumulative_percentages, color="#c83b37", marker="o", linewidth=1.5)
    percent_ax.set_ylim(0, 100)
    percent_ax.set_ylabel("Percent")
    percent_ax.set_yticks(np.arange(0, 101, 20))

    for row, values in enumerate((counts, percentages, cumulative_percentages)):
        offset = ScaledTranslation(0, -(table_offset + row * 14) / 72, ax.figure.dpi_scale_trans)
        transform = ax.get_xaxis_transform() + offset
        for position, value in zip(positions, values):
            text = str(value) if row == 0 else f"{value:.1f}"
            ax.text(position, 0, text, transform=transform, ha="center", va="top", fontsize="small")
        heading = ("Count", "Percent", "Cum %")[row]
        ax.text(-0.02, 0, heading, transform=ax.transAxes + offset,
                ha="right", va="top", fontsize="small")
    return ax
