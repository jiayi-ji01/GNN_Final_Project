import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import E0_OUTPUT_DIR, setup_environment
from src.experiments import run_E0_evaluation
from src.plotting import (
    plot_e0_overall_figure,
    plot_feature_residuals,
    plot_truth_vs_median_scatter,
)
from src.workflow import train_workflow


def main():
    setup_environment(seed=42)
    workflow, history = train_workflow(seed=42)
    case_df, recovery_df, coverage_df, feature_case_df, feature_summary_df = run_E0_evaluation(
        workflow=workflow,
        n_test=100,
        num_posterior_samples=2000,
        n_ppc_draws=100,
        seed=2026,
    )

    plot_truth_vs_median_scatter(case_df)
    plot_feature_residuals(feature_case_df)
    plot_e0_overall_figure(
        case_df,
        coverage_df,
        feature_summary_df,
        out_path=f"{E0_OUTPUT_DIR}/E0_matched_overall_baseline_figure.png",
    )

    os.makedirs(E0_OUTPUT_DIR, exist_ok=True)
    case_df.to_csv(f"{E0_OUTPUT_DIR}/E0_matched_case_level.csv", index=False)
    recovery_df.to_csv(f"{E0_OUTPUT_DIR}/E0_matched_recovery_summary.csv", index=False)
    coverage_df.to_csv(f"{E0_OUTPUT_DIR}/E0_matched_coverage_summary.csv", index=False)
    feature_case_df.to_csv(f"{E0_OUTPUT_DIR}/E0_matched_ppc_feature_case_level.csv", index=False)
    feature_summary_df.to_csv(f"{E0_OUTPUT_DIR}/E0_matched_ppc_feature_summary.csv", index=False)
    print(recovery_df.round(4))
    print(coverage_df.round(4))


if __name__ == "__main__":
    main()
