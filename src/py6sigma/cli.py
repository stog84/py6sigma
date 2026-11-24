"""Command-line analysis of a CSV column."""

import argparse
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd

from .analysis import Analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a numeric CSV column.")
    parser.add_argument("--file", required=True, help="CSV file")
    parser.add_argument("--column", required=True, help="Column name")
    parser.add_argument("--lsl", type=float, help="Lower specification limit")
    parser.add_argument("--usl", type=float, help="Upper specification limit")
    parser.add_argument("--no-plots", action="store_true", help="Do not display plots")
    args = parser.parse_args()
    try:
        frame = pd.read_csv(args.file)
        if args.column not in frame.columns:
            raise ValueError(f"column {args.column!r} not found in CSV")
        data = pd.to_numeric(frame[args.column].dropna(), errors="raise")
        analysis = Analysis(data)
        results = [
            ("Descriptive statistics", analysis.descriptive_statistics()),
            ("Runs test", analysis.run_test()),
            ("Shapiro-Wilk test", analysis.shapiro_test()),
            ("Anderson-Darling test", analysis.anderson_test()),
        ]
        if args.lsl is not None or args.usl is not None:
            results.append(("Capability", analysis.capability(args.lsl, args.usl)))
        for title, result in results:
            print(f"\n{title}")
            for name, value in asdict(result).items():
                print(f"  {name}: {value}")
        if not args.no_plots:
            analysis.histogram()
            analysis.boxplot()
            analysis.run_chart()
            analysis.normality_test()
            plt.show()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
