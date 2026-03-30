import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    E1_OUTPUT_DIR,
    get_e1_curve_level_output_dir,
    get_e1_reference_ablation_output_dir,
    get_e1_severity_output_dir,
    get_e1_severity_summary_output_dir,
    setup_environment,
)
from src.diagnostics import evaluate_curve_level_ppc
from src.experiments import (
    build_E1_reference_ablation_summary,
    build_E1_results_report,
    build_E1_severity_summary,
    make_E1_case_simulator,
    run_E1_evaluation,
    run_E1_example_case,
    run_extended_E1_feature_evaluation,
)
from src.plotting import (
    plot_E1_ablation_comparison,
    plot_E1_severity_3panel_summary,
    plot_E1_severity_single_line,
    plot_e1_example_case,
    plot_e1_overall_figure,
    plot_feature_residuals,
    plot_truth_vs_median_scatter,
    plot_curve_level_ppc_summary,
)
from src.simulators import simulate_curve
from src.workflow import train_workflow


def main():
    setup_environment(seed=42)
    workflow_flow, history = train_workflow(seed=42)

    example_case = run_E1_example_case()
    os.makedirs(E1_OUTPUT_DIR, exist_ok=True)
    plot_e1_example_case(
        example_case["seir_traj"],
        example_case["observed_curve"],
        example_case["time_grid"],
        example_case["sigma"],
        out_path=f"{E1_OUTPUT_DIR}/E1_example_case.png",
    )

    case_df, recovery_df, coverage_df, feature_case_df, feature_summary_df = run_E1_evaluation(
        workflow=workflow_flow,
        n_test=100,
        num_posterior_samples=2000,
        n_ppc_draws=100,
        seed=2026,
    )

    plot_truth_vs_median_scatter(case_df)
    plot_feature_residuals(feature_case_df)
    plot_e1_overall_figure(
        case_df,
        coverage_df,
        feature_summary_df,
        severity_label="structural",
        out_path=f"{E1_OUTPUT_DIR}/E1_structural_overall_figure.png",
    )

    os.makedirs(E1_OUTPUT_DIR, exist_ok=True)
    case_df.to_csv(f"{E1_OUTPUT_DIR}/E1_case_level.csv", index=False)
    recovery_df.to_csv(f"{E1_OUTPUT_DIR}/E1_recovery_summary.csv", index=False)
    coverage_df.to_csv(f"{E1_OUTPUT_DIR}/E1_coverage_summary.csv", index=False)
    feature_case_df.to_csv(f"{E1_OUTPUT_DIR}/E1_ppc_feature_case_level.csv", index=False)
    feature_summary_df.to_csv(f"{E1_OUTPUT_DIR}/E1_ppc_feature_summary.csv", index=False)

    ext_case_df, ext_summary_df = run_extended_E1_feature_evaluation(
        workflow=workflow_flow,
        n_test=100,
        num_posterior_samples=1000,
        n_ppc_draws=100,
        seed=2026,
    )
    ext_case_df.to_csv(f"{E1_OUTPUT_DIR}/E1_ppc_feature_case_level_extended.csv", index=False)
    ext_summary_df.to_csv(f"{E1_OUTPUT_DIR}/E1_ppc_feature_summary_extended.csv", index=False)

    curve_case_df, curve_summary_df = evaluate_curve_level_ppc(
        workflow=workflow_flow,
        case_simulator=make_E1_case_simulator(),
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
    plot_curve_level_ppc_summary(
        curve_case_df,
        severity_label="structural",
        experiment_label="E1",
        out_dir=curve_out_dir,
    )

    report_text = build_E1_results_report(
        recovery_df=recovery_df,
        coverage_df=coverage_df,
        feature_summary_df=feature_summary_df,
        curve_summary_df=curve_summary_df,
    )
    with open(f"{E1_OUTPUT_DIR}/E1_results_report.md", "w") as f:
        f.write(report_text)

    severity_outputs = {}
    for severity in ["mild", "medium", "strong"]:
        result = run_E1_evaluation(
            workflow=workflow_flow,
            severity=severity,
            reference_mode="full_curve",
            n_test=100,
            num_posterior_samples=2000,
            n_ppc_draws=100,
            seed=2026,
        )
        severity_outputs[severity] = {
            "case_df": result[0],
            "recovery_df": result[1],
            "coverage_df": result[2],
            "feature_case_df": result[3],
            "feature_summary_df": result[4],
        }
        out_dir = get_e1_severity_output_dir(severity)
        os.makedirs(out_dir, exist_ok=True)
        result[0].to_csv(f"{out_dir}/E1_case_level.csv", index=False)
        result[1].to_csv(f"{out_dir}/E1_recovery_summary.csv", index=False)
        result[2].to_csv(f"{out_dir}/E1_coverage_summary.csv", index=False)
        result[3].to_csv(f"{out_dir}/E1_ppc_feature_case_level.csv", index=False)
        result[4].to_csv(f"{out_dir}/E1_ppc_feature_summary.csv", index=False)
        plot_e1_overall_figure(
            result[0],
            result[2],
            result[4],
            severity_label=severity,
            out_path=f"{out_dir}/E1_overall_figure.png",
        )

    severity_summary_df = build_E1_severity_summary(
        mild_recovery_df_E1=severity_outputs["mild"]["recovery_df"],
        mild_coverage_df_E1=severity_outputs["mild"]["coverage_df"],
        mild_feature_summary_df_E1=severity_outputs["mild"]["feature_summary_df"],
        medium_recovery_df_E1=severity_outputs["medium"]["recovery_df"],
        medium_coverage_df_E1=severity_outputs["medium"]["coverage_df"],
        medium_feature_summary_df_E1=severity_outputs["medium"]["feature_summary_df"],
        strong_recovery_df_E1=severity_outputs["strong"]["recovery_df"],
        strong_coverage_df_E1=severity_outputs["strong"]["coverage_df"],
        strong_feature_summary_df_E1=severity_outputs["strong"]["feature_summary_df"],
    )
    severity_summary_dir = get_e1_severity_summary_output_dir()
    os.makedirs(severity_summary_dir, exist_ok=True)
    severity_summary_df.to_csv(f"{severity_summary_dir}/E1_severity_summary.csv", index=False)
    plot_E1_severity_single_line(
        severity_summary_df,
        metric="beta_empirical_90",
        out_path=f"{severity_summary_dir}/E1_beta_empirical90_vs_severity.png",
    )
    plot_E1_severity_single_line(
        severity_summary_df,
        metric="R0_empirical_90",
        out_path=f"{severity_summary_dir}/E1_R0_empirical90_vs_severity.png",
    )
    plot_E1_severity_single_line(
        severity_summary_df,
        metric="beta_bias_median",
        out_path=f"{severity_summary_dir}/E1_beta_bias_median_vs_severity.png",
    )
    plot_E1_severity_3panel_summary(
        severity_summary_df,
        out_path=f"{severity_summary_dir}/E1_severity_3panel_summary.png",
    )

    reference_outputs = {}
    for reference_mode in ["full_curve", "early_phase"]:
        result = run_E1_evaluation(
            workflow=workflow_flow,
            severity="medium",
            reference_mode=reference_mode,
            n_test=100,
            num_posterior_samples=2000,
            n_ppc_draws=100,
            seed=2026,
        )
        reference_outputs[reference_mode] = {
            "case_df": result[0],
            "recovery_df": result[1],
            "coverage_df": result[2],
            "feature_case_df": result[3],
            "feature_summary_df": result[4],
        }
        out_dir = get_e1_reference_ablation_output_dir(reference_mode)
        os.makedirs(out_dir, exist_ok=True)
        result[0].to_csv(f"{out_dir}/E1_case_level.csv", index=False)
        result[1].to_csv(f"{out_dir}/E1_recovery_summary.csv", index=False)
        result[2].to_csv(f"{out_dir}/E1_coverage_summary.csv", index=False)
        result[3].to_csv(f"{out_dir}/E1_ppc_feature_case_level.csv", index=False)
        result[4].to_csv(f"{out_dir}/E1_ppc_feature_summary.csv", index=False)

    reference_summary_df = build_E1_reference_ablation_summary(
        full_curve_recovery_df_E1=reference_outputs["full_curve"]["recovery_df"],
        full_curve_coverage_df_E1=reference_outputs["full_curve"]["coverage_df"],
        full_curve_feature_summary_df_E1=reference_outputs["full_curve"]["feature_summary_df"],
        early_phase_recovery_df_E1=reference_outputs["early_phase"]["recovery_df"],
        early_phase_coverage_df_E1=reference_outputs["early_phase"]["coverage_df"],
        early_phase_feature_summary_df_E1=reference_outputs["early_phase"]["feature_summary_df"],
    )
    reference_dir = get_e1_reference_ablation_output_dir()
    os.makedirs(reference_dir, exist_ok=True)
    reference_summary_df.to_csv(f"{reference_dir}/E1_ablation_reference_summary.csv", index=False)
    plot_E1_ablation_comparison(
        reference_summary_df,
        group_col="reference_mode",
        title_prefix="E1 reference ablation",
        out_path=f"{reference_dir}/E1_ablation_reference_comparison.png",
    )

    print(recovery_df.round(4))
    print(coverage_df.round(4))


if __name__ == "__main__":
    main()
