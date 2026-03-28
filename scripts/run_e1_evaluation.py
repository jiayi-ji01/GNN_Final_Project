import os

from src.config import E1_OUTPUT_DIR, get_e1_curve_level_output_dir, setup_environment
from src.diagnostics import evaluate_curve_level_ppc
from src.experiments import run_E1_evaluation
from src.plotting import (
    plot_e0_overall_figure,
    plot_feature_residuals,
    plot_truth_vs_median_scatter,
    plot_curve_level_ppc_summary,
)
from src.simulators import simulate_case_E1, simulate_curve
from src.workflow import train_workflow


def main():
    setup_environment(seed=42)
    workflow, history = train_workflow(seed=42)

    case_df, recovery_df, coverage_df, feature_case_df, feature_summary_df = run_E1_evaluation(
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
        out_path=f"{E1_OUTPUT_DIR}/E1_structural_overall_figure.png",
    )

    os.makedirs(E1_OUTPUT_DIR, exist_ok=True)
    case_df.to_csv(f"{E1_OUTPUT_DIR}/E1_case_level.csv", index=False)
    recovery_df.to_csv(f"{E1_OUTPUT_DIR}/E1_recovery_summary.csv", index=False)
    coverage_df.to_csv(f"{E1_OUTPUT_DIR}/E1_coverage_summary.csv", index=False)
    feature_case_df.to_csv(f"{E1_OUTPUT_DIR}/E1_ppc_feature_case_level.csv", index=False)
    feature_summary_df.to_csv(f"{E1_OUTPUT_DIR}/E1_ppc_feature_summary.csv", index=False)

    curve_case_df, curve_summary_df = evaluate_curve_level_ppc(
        workflow=workflow,
        case_simulator=simulate_case_E1,
        predictive_curve_simulator=simulate_curve,
        n_test=100,
        num_posterior_samples=1000,
        n_ppc_draws=100,
        seed=2026,
        split_frac=0.5,
    )
    curve_out_dir = get_e1_curve_level_output_dir()
    os.makedirs(curve_out_dir, exist_ok=True)
    curve_case_df.to_csv(f"{curve_out_dir}/E1_curve_case_level.csv", index=False)
    curve_summary_df.to_csv(f"{curve_out_dir}/E1_curve_summary.csv", index=False)
    plot_curve_level_ppc_summary(curve_case_df, severity_label="structural", out_dir=curve_out_dir)

    print(recovery_df.round(4))
    print(coverage_df.round(4))


if __name__ == "__main__":
    main()
