import pytest

from py6sigma import select_statistical_method, xy_table_choice


def choose(**kwargs):
    return select_statistical_method(**kwargs)


def test_table_reflects_selector_and_distinguishes_paired_samples():
    table = xy_table_choice()
    rows = [line.split(" | ") for line in table.splitlines() if " | " in line]
    paired = [row for row in rows if "paired samples, non-normal" in row[0]]
    independent = [row for row in rows if "two independent groups, non-normal" in row[0]]
    assert any("Wilcoxon signed-rank test" in row[1] for row in paired)
    assert all("Mann-Whitney" not in row[1] for row in paired)
    assert any("Mann-Whitney" in row[1] for row in independent)
    assert any("1-Sample t test" in row[1] and row[2].strip() == "Yes" for row in rows)


def test_one_sample_t_is_available_and_non_normal_changes_estimand():
    normal = choose(y_type="continuous", x_type=None, objective="compare", samples=1, distribution="normal")
    assert [(item.name, item.available) for item in normal.candidates] == [("1-Sample t test", True)]
    non_normal = choose(y_type="continuous", x_type=None, objective="compare", samples=1, distribution="non-normal")
    assert non_normal.candidates[0].name == "1-Sample Sign test"
    assert "does not test the mean" in non_normal.candidates[0].note
    assert not non_normal.candidates[0].available


def test_paired_and_independent_non_normal_are_not_confused():
    common = dict(y_type="continuous", x_type="discrete", objective="compare", samples=2, distribution="non-normal")
    paired = choose(**common, paired=True)
    independent = choose(**common, paired=False)
    assert [item.name for item in paired.candidates] == [
        "1-Sample Sign test on paired differences", "Wilcoxon signed-rank test"
    ]
    assert [item.name for item in independent.candidates] == ["Mann-Whitney test"]
    assert "shape assumptions" in independent.candidates[0].note


def test_missing_design_information_does_not_select_a_test():
    args = dict(y_type="continuous", x_type="discrete", objective="compare", samples=2)
    result = choose(**args)
    assert not result.candidates
    assert "paired or independent" in result.missing[0]
    result = choose(**args, paired=True)
    assert not result.candidates
    assert "paired differences" in result.missing[0]


def test_variance_needs_independent_groups_and_identifies_normal_sensitivity():
    args = dict(y_type="continuous", x_type="discrete", objective="variance", samples=2)
    paired = choose(**args, paired=True, distribution="normal")
    assert not paired.candidates
    assert "paired observations" in paired.warnings[-1]
    independent = choose(**args, paired=False, distribution="normal")
    assert [candidate.name for candidate in independent.candidates] == [
        "Test for equal variances (Levene)", "2-Variance test (F)"
    ]
    assert "sensitive to non-normality" in independent.candidates[1].note
    one = choose(y_type="continuous", x_type=None, objective="variance", samples=1, distribution="non-normal")
    assert not one.candidates


def test_discrete_inputs_need_binary_and_pairing_details():
    args = dict(y_type="discrete", x_type="discrete", objective="proportion", samples=2)
    assert not choose(**args).candidates
    assert "two categories" in choose(**args).missing[0]
    assert not choose(**args, binary=True).candidates
    assert "paired" in choose(**args, binary=True).missing[0]
    assert [c.name for c in choose(**args, binary=True, paired=False).candidates] == ["2-Proportion test"]
    assert not choose(**args, binary=True, paired=True).candidates


def test_continuous_relationship_and_unknown_combinations():
    result = choose(y_type="continuous", x_type="continuous", objective="regression")
    assert [candidate.name for candidate in result.candidates] == ["Simple linear regression"]
    assert "does not establish causation" in result.warnings[-1]
    result = choose(y_type="discrete", x_type="continuous", objective="regression")
    assert not result.candidates
    assert "outside the scope" in result.warnings[-1]


@pytest.mark.parametrize("kwargs", [
    {"y_type": "invalid"},
    {"x_type": "invalid"},
    {"objective": "invalid"},
    {"distribution": "invalid"},
    {"samples": 0},
    {"samples": True},
    {"paired": 1},
    {"binary": "yes"},
    {"samples": 1, "paired": False},
])
def test_invalid_inputs_are_rejected(kwargs):
    args = dict(y_type="continuous", x_type="discrete", objective="compare")
    args.update(kwargs)
    with pytest.raises(ValueError):
        choose(**args)
