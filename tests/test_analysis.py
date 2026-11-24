import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.sandbox.stats.runs import runstest_1samp

from py6sigma import Analysis, pareto_chart, scatter_plot, sigma_level


DATA = [12, 13, 11, 15, 18, 14, 16, 11, 13, 17]


def test_one_sample_t_test_two_sided():
    result = Analysis([1, 2, 3, 4, 5]).one_sample_t_test(2)
    assert result.n == 5
    assert result.mean == pytest.approx(3)
    assert result.std_dev == pytest.approx(math.sqrt(2.5))
    assert result.standard_error == pytest.approx(math.sqrt(0.5))
    assert result.hypothesized_mean == 2
    assert result.degrees_of_freedom == 4
    assert result.statistic == pytest.approx(1.414213562373095)
    assert result.p_value == pytest.approx(0.2301996410804989)
    assert result.confidence_interval == pytest.approx((1.0367568385, 4.9632431615))


@pytest.mark.parametrize("alternative,p_value,interval", [
    ("less", 0.8849001794597505, (-math.inf, 4.507443319062323)),
    ("greater", 0.11509982054024945, (1.4925566809376771, math.inf)),
])
def test_one_sample_t_test_one_sided(alternative, p_value, interval):
    result = Analysis([1, 2, 3, 4, 5]).one_sample_t_test(2, alternative=alternative)
    assert result.alternative == alternative
    assert result.p_value == pytest.approx(p_value)
    assert result.confidence_interval == pytest.approx(interval)


def test_one_sample_t_test_confidence_level():
    analysis = Analysis([1, 2, 3, 4, 5])
    low, high = analysis.one_sample_t_test(2, confidence_level=0.90).confidence_interval
    assert (low, high) == pytest.approx((1.4925566809, 4.5074433191))


def test_one_sample_t_test_minitab_output_from_excel(tmp_path):
    path = tmp_path / "input.xlsx"
    pd.DataFrame({"Speed": [1, 2, 3, 4, 5]}).to_excel(path, index=False)
    result = Analysis.from_excel(path, column="Speed").one_sample_t_test(2, alternative="greater")
    assert str(result) == (
        "One-Sample T: Speed\n\n"
        "Test of mu = 2 vs > 2\n\n"
        "                                   95% Lower\n"
        "Variable  N  Mean  StDev  SE Mean      Bound     T      P\n"
        "Speed     5  3.00   1.58     0.71       1.49  1.41  0.115\n\n"
        "Conclusion (seuil de 5 %) : les données ne permettent pas de conclure "
        "que la moyenne de Speed est supérieure à 2.0."
    )
    assert result.p_value == pytest.approx(0.11509982054024945)


def test_one_sample_t_test_display_other_alternatives():
    analysis = Analysis([1, 2, 3, 4, 5])
    assert "90% CI" in str(analysis.one_sample_t_test(2, confidence_level=0.9, name="Output"))
    assert "One-Sample T: Output" in str(analysis.one_sample_t_test(2, name="Output"))
    assert "95% Upper" in str(analysis.one_sample_t_test(2, alternative="less"))
    assert "vs < 2" in str(analysis.one_sample_t_test(2, alternative="less"))


@pytest.mark.parametrize("alternative,mean,expected", [
    ("less", 4, "permettent de conclure que la moyenne de Data est inférieure à 4.0"),
    ("greater", 2, "permettent de conclure que la moyenne de Data est supérieure à 2.0"),
    ("two-sided", 1, "permettent de conclure que la moyenne de Data est différente de 1.0"),
    ("less", 2, "ne permettent pas de conclure que la moyenne de Data est inférieure à 2.0"),
    ("two-sided", 2, "ne permettent pas de conclure que la moyenne de Data est différente de 2.0"),
])
def test_one_sample_t_test_conclusion(alternative, mean, expected):
    result = Analysis([1, 2, 3, 4, 5]).one_sample_t_test(
        mean, alternative=alternative, confidence_level=0.8
    )
    assert f"\n\nConclusion (seuil de 20 %) : les données {expected}." in str(result)


@pytest.mark.parametrize("data,mean,kwargs", [
    ([1], 0, {}),
    ([1, 1], 0, {}),
    ([1, 2], float("nan"), {}),
    ([1, 2], float("inf"), {}),
    ([1, 2], True, {}),
    ([1, 2], 0, {"alternative": "invalid"}),
    ([1, 2], 0, {"confidence_level": 0}),
    ([1, 2], 0, {"confidence_level": 1}),
    ([1, 2], 0, {"confidence_level": 95}),
    ([1, 2], 0, {"confidence_level": float("nan")}),
    ([1, 2], 0, {"confidence_level": True}),
    ([1, 2], 0, {"name": ""}),
    ([1, 2], 0, {"name": "a\nb"}),
])
def test_one_sample_t_test_rejects_invalid_input(data, mean, kwargs):
    with pytest.raises(ValueError):
        Analysis(data).one_sample_t_test(mean, **kwargs)


def test_scatter_plot_paired_coordinates_and_supplied_axes():
    fig, ax = plt.subplots()
    assert scatter_plot([3, 1, 2], [30, 10, 20], x_name="Input", y_name="Output", ax=ax) is ax
    fig.canvas.draw()
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [[3, 30], [1, 10], [2, 20]])
    assert not ax.lines
    assert ax.get_title() == "Scatterplot of Output vs Input"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Input", "Output")
    plt.close(fig)


def test_scatter_plot_groups_and_regression():
    ax = scatter_plot([1, 1, 3, 3], [3, 10, 7, 14],
                      groups=["B", "A", "B", "A"], group_name="Team", regression=True)
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [[1, 3], [3, 7]])
    np.testing.assert_allclose(ax.collections[1].get_offsets(), [[1, 10], [3, 14]])
    assert ax.lines[0].get_ydata() == pytest.approx([3, 7])
    assert ax.lines[1].get_ydata() == pytest.approx([10, 14])
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["B", "A"]
    assert ax.get_legend().get_title().get_text() == "Team"
    plt.close(ax.figure)


@pytest.mark.parametrize("x,y,kwargs", [
    ([], [], {}),
    ([1, 2], [1], {}),
    ([[1, 2]], [[3, 4]], {}),
    ([1, float("nan")], [2, 3], {}),
    ([1, 2], [3, float("inf")], {}),
    ([1, 2], [3, 4], {"groups": ["A"]}),
    ([1, 2], [3, 4], {"groups": ["A", None]}),
    ([1], [3], {"regression": True}),
    ([1, 1], [3, 4], {"regression": True}),
    ([1, 2], [3, 4], {"groups": ["A", "B"], "regression": True}),
    ([1, 2], [3, 4], {"regression": 1}),
])
def test_scatter_plot_rejects_invalid_input(x, y, kwargs):
    with pytest.raises(ValueError):
        scatter_plot(x, y, **kwargs)


def test_pareto_chart_counts_percentages_and_raw_data():
    ax = pareto_chart(["Scratch", "Peel", "Scratch", "Smudge", "Peel", "Scratch"],
                      combine_after=None, name="Defects")
    ax.figure.canvas.draw()
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Scratch", "Peel", "Smudge"]
    assert [bar.get_height() for bar in ax.patches] == [3, 2, 1]
    assert ax.figure.axes[1].lines[0].get_ydata() == pytest.approx([50, 83.333333, 100])
    assert ax.get_title() == "Pareto Chart of Defects"
    assert [text.get_text() for text in ax.texts[:3]] == ["3", "2", "1"]
    plt.close(ax.figure)


def test_pareto_chart_summarized_data_and_strict_threshold():
    fig, ax = plt.subplots()
    result = pareto_chart(
        ["A", "B", "A", "C", "D", "E"],
        [25, 30, 15, 20, 5, 5],
        combine_after=0.9,
        ax=ax,
    )
    assert result is ax
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["A", "B", "C", "D", "Other"]
    assert [bar.get_height() for bar in ax.patches] == [40, 30, 20, 5, 5]
    assert fig.axes[1].lines[0].get_ydata() == pytest.approx([40, 70, 90, 95, 100])
    plt.close(fig)


def test_pareto_chart_other_is_last_and_distinct():
    ax = pareto_chart({"Other": 50, "A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
                      combine_after=50)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Other", "A", "Other (combined)"]
    assert [bar.get_height() for bar in ax.patches] == [50, 10, 40]
    plt.close(ax.figure)


@pytest.mark.parametrize("data,frequencies,threshold", [
    ([], None, 95),
    (["A", "B"], [1], 95),
    (["A"], [0], 95),
    (["A"], [-1], 95),
    (["A"], [1.5], 95),
    (["A"], [float("nan")], 95),
    ([None], None, 95),
    ([""], None, 95),
    ([float("inf")], None, 95),
    ({"A": 1}, [1], 95),
    (["A"], None, 0),
    (["A"], None, 101),
    (["A"], None, True),
])
def test_pareto_chart_rejects_invalid_input(data, frequencies, threshold):
    with pytest.raises(ValueError):
        pareto_chart(data, frequencies, combine_after=threshold)


def test_sigma_level_from_dpo_or_counts():
    from_dpo = sigma_level(dpo=0.02275)
    from_counts = sigma_level(defects=91, opportunities_per_unit=4, units=1000)
    assert from_counts == from_dpo
    assert from_dpo.dpmo == pytest.approx(22750)
    assert from_dpo.z_lt == pytest.approx(2, abs=1e-4)
    assert from_dpo.z_st == pytest.approx(3.5, abs=1e-4)
    assert sigma_level(3.4e-6).z_lt == pytest.approx(4.5, abs=0.002)
    assert sigma_level(3.4e-6).z_st == pytest.approx(6.0, abs=0.002)
    result = sigma_level(defects=125, units=2000, opportunities_per_unit=5)
    assert result.dpo == pytest.approx(0.0125)
    assert result.dpmo == pytest.approx(12500)
    assert result.z_lt == pytest.approx(2.241403, abs=1e-6)
    assert result.z_st == pytest.approx(3.741403, abs=1e-6)


def test_sigma_level_zero_and_full_defects():
    assert sigma_level(0).z_lt == math.inf
    assert sigma_level(0).z_st == math.inf
    assert sigma_level(1).z_lt == -math.inf
    assert sigma_level(1).z_st == -math.inf


@pytest.mark.parametrize("kwargs", [
    {},
    {"dpo": -0.01},
    {"dpo": 1.01},
    {"dpo": float("nan")},
    {"dpo": float("inf")},
    {"dpo": True},
    {"dpo": 0.1, "defects": 1, "opportunities_per_unit": 1, "units": 10},
    {"defects": 1, "opportunities_per_unit": 1},
    {"defects": 11, "opportunities_per_unit": 1, "units": 10},
    {"defects": -1, "opportunities_per_unit": 1, "units": 10},
    {"defects": 1, "opportunities_per_unit": 0, "units": 10},
    {"defects": 1, "opportunities_per_unit": 1, "units": 0},
    {"defects": 1.5, "opportunities_per_unit": 1, "units": 10},
])
def test_sigma_level_rejects_invalid_input(kwargs):
    with pytest.raises(ValueError):
        sigma_level(**kwargs)


def test_descriptive_and_capability():
    analysis = Analysis(DATA)
    result = analysis.descriptive_statistics()
    assert result.n == 10
    assert result.mean == pytest.approx(14)
    assert result.std_dev == pytest.approx(np.std(DATA, ddof=1))
    assert result.q1 == pytest.approx(np.percentile(DATA, 25))
    cap = analysis.capability(10, 20)
    within = np.mean(np.abs(np.diff(DATA))) / 1.128
    assert cap.cp == pytest.approx(10 / (6 * within))
    assert cap.cpk == pytest.approx(4 / (3 * within))
    assert cap.pp == pytest.approx(10 / (6 * result.std_dev))
    assert cap.ppk == pytest.approx(4 / (3 * result.std_dev))


def test_normal_capability_two_limits_and_ppm():
    result = Analysis([1, 2, 3, 4, 5]).capability(lsl=2, usl=4)
    assert result.std_within == pytest.approx(1 / 1.128)
    assert result.std_overall == pytest.approx(math.sqrt(2.5))
    assert result.cp == pytest.approx(0.376)
    assert result.cpk == pytest.approx(0.376)
    assert result.pp == pytest.approx(0.2108185107)
    assert result.ppk == pytest.approx(0.2108185107)
    assert result.observed.ppm_below_lsl == 200000
    assert result.observed.ppm_above_usl == 200000
    assert result.observed.ppm_total == 400000
    assert result.expected_within.ppm_total == pytest.approx(259319.9191)
    assert result.expected_overall.ppm_total == pytest.approx(527089.2569)


def test_normal_capability_one_limit_and_figure():
    analysis = Analysis([1, 2, 3, 4, 5])
    upper = analysis.capability(usl=2)
    assert upper.cp is None and upper.pp is None
    assert upper.cpl is None and upper.ppl is None
    assert upper.cpk == upper.cpu
    assert upper.ppk == upper.ppu
    assert upper.cpk < 0
    assert upper.observed.ppm_total == 600000
    assert upper.observed.ppm_below_lsl is None
    lower = analysis.capability(lsl=4)
    assert lower.cpk == lower.cpl
    assert lower.ppk == lower.ppl
    assert lower.observed.ppm_total == 600000
    assert lower.observed.ppm_above_usl is None
    figure = analysis.capability_analysis_normal(usl=2, name="Time")
    figure.canvas.draw()
    assert len(figure.axes) == 6
    assert figure.axes[1].get_xlabel() == "Time"
    assert figure._suptitle.get_text() == "Process Capability of Time"
    assert "N/A" in figure.axes[0].texts[0].get_text()
    assert "600000.00" in figure.axes[3].texts[0].get_text()
    plt.close(figure)


def test_normal_capability_histogram_bins():
    analysis = Analysis([1, 2, 3, 4, 5])
    figure = analysis.capability_analysis_normal(usl=4, bins=8)
    ax = figure.axes[1]
    assert len(ax.patches) == 8
    assert ax.patches[0].get_width() == pytest.approx(0.5)
    plt.close(figure)
    for bins in (0, -1, True):
        with pytest.raises(ValueError):
            analysis.capability_analysis_normal(usl=4, bins=bins)


def test_run_test_excludes_median_and_counts_runs():
    data = [1, 2, 3, 3, 4, 5, 1]
    result = Analysis(data).run_test()
    z, p = runstest_1samp([1, 2, 4, 5, 1], cutoff=3)
    assert (result.runs, result.n_below, result.n_above) == (3, 3, 2)
    assert (result.statistic, result.p_value) == pytest.approx((z, p))


def test_normality_tests():
    analysis = Analysis(DATA)
    shapiro = stats.shapiro(DATA)
    assert analysis.shapiro_test().p_value == pytest.approx(shapiro.pvalue)
    anderson = stats.anderson(DATA, dist="norm")
    result = analysis.anderson_test()
    assert result.statistic == pytest.approx(anderson.statistic)
    assert result.critical_values == pytest.approx(anderson.critical_values)
    assert result.significance_levels == pytest.approx(anderson.significance_level)


@pytest.mark.parametrize("data", [[], [1, float("nan")], [1, float("inf")], [[1, 2]]])
def test_invalid_data(data):
    with pytest.raises(ValueError):
        Analysis(data)


def test_undefined_and_invalid_analyses():
    constant = Analysis([1, 1, 1, 1, 1])
    assert math.isnan(constant.descriptive_statistics().skewness)
    for method in (constant.run_test, constant.shapiro_test, constant.anderson_test, constant.normality_test):
        with pytest.raises(ValueError):
            method()
    with pytest.raises(ValueError):
        constant.capability(0, 2)
    for limits in [(2, 2), (3, 2), (0, float("inf"))]:
        with pytest.raises(ValueError):
            Analysis(DATA).capability(*limits)
    with pytest.raises(ValueError):
        Analysis(DATA).capability()
    with pytest.raises(ValueError):
        Analysis([1, 2]).shapiro_test()
    with pytest.raises(ValueError):
        Analysis([1, 2, 3, 4]).anderson_test()


def test_plots_use_supplied_axes():
    analysis = Analysis(DATA)
    for method in (analysis.histogram, analysis.boxplot, analysis.run_chart, analysis.normality_test):
        fig, ax = plt.subplots()
        assert method(ax) is ax
        plt.close(fig)


def test_boxplot_uses_minitab_quartiles():
    ax = Analysis([0, 2, 4, 6, 8, 10, 12, 14, 16, 24]).boxplot(name="Measurements")
    ax.figure.canvas.draw()
    vertices = ax.patches[0].get_path().vertices
    assert vertices[:, 1].min() == pytest.approx(3.5)
    assert vertices[:, 1].max() == pytest.approx(14.5)
    assert ax.lines[4].get_ydata() == pytest.approx([9, 9])
    assert ax.get_title() == "Boxplot of Measurements"
    plt.close(ax.figure)


def test_boxplot_whiskers_and_outliers_with_zero_iqr():
    ax = Analysis([1] * 9 + [0]).boxplot()
    assert ax.lines[0].get_ydata() == pytest.approx([1, 1])
    assert ax.lines[1].get_ydata() == pytest.approx([1, 1])
    assert ax.lines[5].get_ydata() == pytest.approx([0])
    assert ax.lines[5].get_marker() == "*"
    plt.close(ax.figure)


def test_boxplot_includes_whisker_boundary():
    # Q1 = 1, Q3 = 3, so the upper fence is exactly 6.
    ax = Analysis([1, 1, 1, 2, 2, 3, 3, 3, 6]).boxplot()
    assert ax.lines[1].get_ydata() == pytest.approx([3, 6])
    assert len(ax.lines[5].get_ydata()) == 0
    plt.close(ax.figure)


def test_boxplot_groups_preserve_alignment_and_order():
    ax = Analysis([10, 1, 20, 2, 30, 3]).boxplot(
        name="TAT", groups=["B", "A", "B", "A", "B", "A"], group_name="Team"
    )
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["B", "A"]
    assert ax.lines[4].get_ydata() == pytest.approx([20, 20])
    assert ax.lines[10].get_ydata() == pytest.approx([2, 2])
    assert ax.get_xlabel() == "Team"
    plt.close(ax.figure)


def test_boxplot_reference_lines():
    ax = Analysis([1, 2, 3, 4]).boxplot(reference_lines=[0, 40])
    assert ax.lines[-2].get_ydata() == [0, 0]
    assert ax.lines[-1].get_ydata() == [40, 40]
    assert ax.get_ylim()[0] <= 0
    assert ax.get_ylim()[1] >= 40
    plt.close(ax.figure)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "40"])
def test_boxplot_rejects_invalid_reference_lines(value):
    with pytest.raises(ValueError):
        Analysis([1, 2]).boxplot(reference_lines=[value])


@pytest.mark.parametrize("groups", [["A"], ["A", None], ["A", ""], ["A", float("nan")]])
def test_boxplot_rejects_invalid_groups(groups):
    with pytest.raises(ValueError):
        Analysis([1, 2]).boxplot(groups=groups)


def test_normality_test_coordinates_and_summary():
    analysis = Analysis(DATA)
    ax = analysis.normality_test()
    ordered = np.sort(DATA)
    quantiles = stats.norm.ppf((np.arange(1, 11) - 0.3) / 10.4)
    np.testing.assert_allclose(ax.collections[0].get_offsets(), np.column_stack((ordered, quantiles)))
    fit = ax.lines[0]
    assert fit.get_xdata() == pytest.approx(np.mean(DATA) + np.std(DATA, ddof=1) * fit.get_ydata())
    assert len(ax.lines) == 3
    statistic, p_value = normal_ad(np.asarray(DATA))
    summary = ax.texts[0].get_text()
    assert f"AD      {statistic:.4g}" in summary
    assert f"P-value {p_value:.4g}" in summary
    assert ax.get_ylabel() == "Percent"
    assert ax.get_title() == "Normal Probability Plot"
    plt.close(ax.figure)


def test_run_chart_statistics_example():
    data = [2, 5, 4, 3, 7, 9, 8, 1, 6]
    result = Analysis(data).run_chart_statistics()
    assert result.runs_about_median == 4
    assert result.expected_runs_about_median == pytest.approx(5.444444)
    assert result.longest_run_about_median == 4
    assert result.p_clustering == pytest.approx(0.1482, abs=0.0001)
    assert result.p_mixtures == pytest.approx(0.8518, abs=0.0001)
    assert result.runs_up_down == 5
    assert result.expected_runs_up_down == pytest.approx(5.666667)
    assert result.longest_run_up_down == 2
    assert result.p_trends == pytest.approx(0.2777, abs=0.0001)
    assert result.p_oscillation == pytest.approx(0.7223, abs=0.0001)


def test_run_chart_ties_and_degenerate_data():
    result = Analysis([1, 2, 2, 3]).run_chart_statistics()
    assert result.runs_about_median == 2
    assert result.longest_run_about_median == 3
    assert result.runs_up_down == 3
    ax = Analysis([1, 1, 1]).run_chart()
    assert "N/A" in ax.texts[0].get_text()
    assert "Ties" in ax.texts[2].get_text()
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    assert all(text.get_window_extent(renderer).y0 >= 0 for text in ax.texts)
    plt.close(ax.figure)
    result = Analysis([1]).run_chart_statistics()
    assert result.p_clustering is None
    assert result.p_trends is None


def test_normality_test_small_samples():
    ax = Analysis([1, 2]).normality_test()
    assert len(ax.lines) == 1
    assert "P-value" not in ax.texts[0].get_text()
    plt.close(ax.figure)
    ax = Analysis([1, 2, 3]).normality_test()
    assert len(ax.lines) == 3
    assert "P-value" not in ax.texts[0].get_text()
    plt.close(ax.figure)


def test_from_excel_and_chart_names(tmp_path):
    path = tmp_path / "input.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Other": [0]}).to_excel(writer, sheet_name="Other", index=False)
        pd.DataFrame({"Value": [3, None, 1, 2]}).to_excel(writer, sheet_name="Data", index=False)
    analysis = Analysis.from_excel(path, column="Value", sheet_name="Data")
    ax = analysis.run_chart()
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [3, 1, 2])
    plt.close(ax.figure)
    ax = analysis.normality_test()
    np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 0], [1, 2, 3])
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="column"):
        Analysis.from_excel(path, column="Missing")
    with pytest.raises(ValueError, match="Worksheet"):
        Analysis.from_excel(path, column="Value", sheet_name="Missing")


def test_from_excel_rejects_non_numeric_cells(tmp_path):
    path = tmp_path / "input.xlsx"
    pd.DataFrame({"Value": [1, "bad", 3]}).to_excel(path, index=False)
    with pytest.raises(ValueError):
        Analysis.from_excel(path, column="Value")
