"""Scatterplots of paired observations."""

from collections.abc import Iterable
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import stats

from .analysis import _group_indices


def scatter_plot(
    x: Iterable[float],
    y: Iterable[float],
    *,
    x_name: str = "X",
    y_name: str = "Y",
    groups: Iterable[str | Real] | None = None,
    group_name: str | None = None,
    regression: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot paired numeric observations, optionally with groups and linear fits."""
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size == 0 or x_values.size != y_values.size:
        raise ValueError("x and y must be non-empty one-dimensional sequences of equal length")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("x and y must contain only finite numbers")
    if not isinstance(regression, bool):
        raise ValueError("regression must be a boolean")

    grouped = {None: list(range(x_values.size))} if groups is None else _group_indices(groups, x_values.size)
    if regression:
        for indices in grouped.values():
            sample = x_values[indices]
            if sample.size < 2 or np.ptp(sample) == 0:
                raise ValueError("linear regression requires at least two observations with different x values in each group")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for group, indices in grouped.items():
        x_sample = x_values[indices]
        y_sample = y_values[indices]
        points = ax.scatter(x_sample, y_sample, s=30, label=str(group) if group is not None else None)
        if regression:
            fit = stats.linregress(x_sample, y_sample)
            line_x = np.array([x_sample.min(), x_sample.max()])
            ax.plot(line_x, fit.intercept + fit.slope * line_x, color=points.get_facecolor()[0])

    ax.set(title=f"Scatterplot of {y_name} vs {x_name}", xlabel=x_name, ylabel=y_name)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    if groups is not None:
        ax.legend(title=group_name)
    return ax
