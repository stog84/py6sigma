import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["py6sigma", "py6sigma.cli"])
def test_cli_help(module):
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: py6sigma" in result.stdout
    assert "Lean Six Sigma" in result.stdout
    assert "Library capabilities" in result.stdout
    assert "one-sample t-test" in result.stdout
    assert "Pareto" in result.stdout
    assert "Python API" in result.stdout
    assert "--file" in result.stdout
    assert "--column" in result.stdout


@pytest.mark.parametrize("module", ["py6sigma", "py6sigma.cli"])
def test_csv_cli(tmp_path, module):
    data = tmp_path / "data.csv"
    data.write_text("CycleTime\n12\n13\n11\n15\n18\n14\n16\n11\n13\n17\n", encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    cmd = [sys.executable, "-m", module, "--file", str(data), "--column", "CycleTime", "--no-plots"]
    result = subprocess.run(cmd + ["--lsl", "10", "--usl", "20"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "Descriptive statistics" in result.stdout
    assert "Runs test" in result.stdout
    assert "Anderson-Darling test" in result.stdout
    assert "Capability" in result.stdout
    result = subprocess.run(cmd + ["--lsl", "10"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "Capability" in result.stdout
