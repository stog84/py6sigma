import os
import subprocess
import sys


def test_csv_cli(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("CycleTime\n12\n13\n11\n15\n18\n14\n16\n11\n13\n17\n", encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    cmd = [sys.executable, "-m", "py6sigma.cli", "--file", str(data), "--column", "CycleTime", "--no-plots"]
    result = subprocess.run(cmd + ["--lsl", "10", "--usl", "20"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "Descriptive statistics" in result.stdout
    assert "Runs test" in result.stdout
    assert "Anderson-Darling test" in result.stdout
    assert "Capability" in result.stdout
    result = subprocess.run(cmd + ["--lsl", "10"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "Capability" in result.stdout
