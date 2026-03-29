import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    E3_OUTPUT_DIR,
    get_e3_caselevel_analysis_output_dir,
    get_e3_curve_level_output_dir,
    get_e3_curve_level_tau_output_dir,
    get_e3_severity_output_dir,
    get_e3_severity_summary_output_dir,
    get_e3_truth_region_output_dir,
    setup_environment,
)
from src.diagnostics import evaluate_curve_level_ppc, evaluate_curve_level_ppc_tau_aligned
from src.experiments import (
    build_E3_caselevel_correlation_summary,
    build_E3_controlled_cases,
    build_E3_eta_tau_bin_summaries,
    build_E3_severity_summary,
    build_E3_truth_region_tables,
    run_E3_evaluation,
    run_E3_example_case,
)
from src.plotting import (
    plot_E3_eta_bin_panels,
    plot_E3_severity_lines,
    plot_E3_tau_bin_panels,
    plot_E3_truth_group_panels,
    plot_curve_level_ppc_summary,
    plot_e3_controlled_grid,
    plot_e3_example_case,
    plot_e3_overall_figure,
    plot_feature_residuals,
    scatter_with_group_colors,
)
from src.simulators import simulate_case_E3, simulate_curve
from src.workflow import train_workflow


def main():
    setup_environment(seed=42)
    workflow, history = train_workflow(seed=42)

    os.makedirs(E3_OUTPUT_DIR, exist_ok=True)

    example_case = run_E3_example_case()
    plot_e3_example_case(example_case, out_path=f"{E3_OUTPUT_DIR}/E3_example_case.png")

    controlled_cases = build_E3_controlled_cases()
    plot_e3_controlled_grid(controlled_cases, out_path=f"{E3_OUTPUT_DIR}/E3_controlled_severity_grid.png")

    outputs = {}
    for severity in ["mild", "medium", "strong"]:
        result = run_E3_evaluation(
            workflow=workflow,
            severity=severity,
            n_test=100,
            num_posterior_samples=2000,
            n_ppc_draws=100,
            seed=2028,
        )
        outputs[severity] = {
            "case_df": result[0],
            "recovery_df": result[1],
            "coverage_df": result[2],
            "feature_case_df": result[3],
            "feature_summary_df": result[4],
        }

        out_dir = get_e3_severity_output_dir(severity)
        os.makedirs(out_dir, exist_ok=True)
        result[0].to_csv(f"{out_dir}/E3_case_level.csv", index=False)
        result[1].to_csv(f"{out_dir}/E3_recovery_summary.csv", index=False)
        result[2].to_csv(f"{out_dir}/E3_coverage_summary.csv", index=False)
        result[3].to_csv(f"{out_dir}/E3_ppc_feature_case_level.csv", index=False)
        result[4].to_csv(f"{out_dir}/E3_ppc_feature_summary.csv", index=False)

        plot_e3_overall_figure(
            result[0],
            result[2],
            result[4],
            severity_label=severity,
            out_path=f"{out_dir}/E3_overall_figure.png",
        )
        plot_feature_residuals(
            result[3],
            title_prefix=f"E3 {severity}",
        ).savefig(f"{out_dir}/E3_feature_residuals.png", dpi=200, bbox_inches="tight")

        curve_case_df, curve_summary_df = evaluate_curve_level_ppc(
            workflow=workflow,
            case_simulator=lambda rng=None, sev=severity: simulate_case_E3(rng=rng, severity=sev),
            predictive_curve_simulator=simulate_curve,
            n_test=100,
            num_posterior_samples=1000,
            n_ppc_draws=100,
            seed=2030,
            split_mode="fraction",
            split_frac=0.5,
        )
        curve_out_dir = get_e3_curve_level_output_dir(severity)
        os.makedirs(curve_out_dir, exist_ok=True)
        curve_case_df.to_csv(f"{curve_out_dir}/E3_curve_case_level_{severity}.csv", index=False)
        curve_summary_df.to_csv(f"{curve_out_dir}/E3_curve_summary_{severity}.csv", index=False)
        plot_curve_level_ppc_summary(
            curve_case_df,
            severity_label=severity,
            experiment_label="E3",
            out_dir=curve_out_dir,
            split_mode="fraction",
        )

        curve_case_df_tau, curve_summary_df_tau = evaluate_curve_level_ppc_tau_aligned(
            workflow=workflow,
            case_simulator=lambda rng=None, sev=severity: simulate_case_E3(rng=rng, severity=sev),
            predictive_curve_simulator=simulate_curve,
            n_test=100,
            num_posterior_samples=1000,
            n_ppc_draws=100,
            seed=2032,
        )
        curve_tau_out_dir = get_e3_curve_level_tau_output_dir(severity)
        os.makedirs(curve_tau_out_dir, exist_ok=True)
        curve_case_df_tau.to_csv(f"{curve_tau_out_dir}/E3_curve_case_level_tau_aligned_{severity}.csv", index=False)
        curve_summary_df_tau.to_csv(f"{curve_tau_out_dir}/E3_curve_summary_tau_aligned_{severity}.csv", index=False)
        plot_curve_level_ppc_summary(
            curve_case_df_tau,
            severity_label=severity,
            experiment_label="E3_tau",
            out_dir=curve_tau_out_dir,
            split_mode="tau",
        )

    severity_summary_df = build_E3_severity_summary(
        mild_recovery_df_E3=outputs["mild"]["recovery_df"],
        mild_coverage_df_E3=outputs["mild"]["coverage_df"],
        mild_feature_summary_df_E3=outputs["mild"]["feature_summary_df"],
        medium_recovery_df_E3=outputs["medium"]["recovery_df"],
        medium_coverage_df_E3=outputs["medium"]["coverage_df"],
        medium_feature_summary_df_E3=outputs["medium"]["feature_summary_df"],
        strong_recovery_df_E3=outputs["strong"]["recovery_df"],
        strong_coverage_df_E3=outputs["strong"]["coverage_df"],
        strong_feature_summary_df_E3=outputs["strong"]["feature_summary_df"],
    )
    severity_dir = get_e3_severity_summary_output_dir()
    os.makedirs(severity_dir, exist_ok=True)
    severity_summary_df.to_csv(f"{severity_dir}/E3_severity_summary.csv", index=False)
    plot_E3_severity_lines(severity_summary_df, out_dir=severity_dir)

    case_all = pd.concat(
        [outputs["mild"]["case_df"], outputs["medium"]["case_df"], outputs["strong"]["case_df"]],
        axis=0,
        ignore_index=True,
    )
    case_all["severity"] = pd.Categorical(case_all["severity"], categories=["mild", "medium", "strong"], ordered=True)

    caselevel_dir = get_e3_caselevel_analysis_output_dir()
    os.makedirs(caselevel_dir, exist_ok=True)
    case_all.to_csv(f"{caselevel_dir}/E3_case_level_all.csv", index=False)

    corr_df = build_E3_caselevel_correlation_summary(case_all)
    corr_df.to_csv(f"{caselevel_dir}/E3_caselevel_correlation_summary.csv", index=False)

    scatter_with_group_colors(
        case_all,
        x="eta",
        y="R0_err_median",
        title="E3 case-level: eta vs R0 median error",
        xlabel="eta (post-change reporting factor)",
        ylabel="R0 median error",
        out_path=f"{caselevel_dir}/E3_eta_vs_R0_err_median.png",
    )
    scatter_with_group_colors(
        case_all,
        x="eta",
        y="gamma_err_median",
        title="E3 case-level: eta vs gamma median error",
        xlabel="eta (post-change reporting factor)",
        ylabel="gamma median error",
        out_path=f"{caselevel_dir}/E3_eta_vs_gamma_err_median.png",
    )
    scatter_with_group_colors(
        case_all,
        x="tau",
        y="R0_err_median",
        title="E3 case-level: tau vs R0 median error",
        xlabel="tau (change point day)",
        ylabel="R0 median error",
        out_path=f"{caselevel_dir}/E3_tau_vs_R0_err_median.png",
    )
    scatter_with_group_colors(
        case_all,
        x="eta",
        y="R0_covered90",
        title="E3 case-level: eta vs R0 90% coverage indicator",
        xlabel="eta (post-change reporting factor)",
        ylabel="R0 covered by 90% interval",
        out_path=f"{caselevel_dir}/E3_eta_vs_R0_covered90.png",
    )

    case_binned, eta_bin_summary, tau_bin_summary = build_E3_eta_tau_bin_summaries(case_all)
    eta_bin_summary.to_csv(f"{caselevel_dir}/E3_eta_bin_summary.csv", index=False)
    tau_bin_summary.to_csv(f"{caselevel_dir}/E3_tau_bin_summary.csv", index=False)
    plot_E3_eta_bin_panels(eta_bin_summary, out_path=f"{caselevel_dir}/E3_eta_bin_panels.png")
    plot_E3_tau_bin_panels(tau_bin_summary, out_path=f"{caselevel_dir}/E3_tau_bin_panels.png")

    truth_tables = build_E3_truth_region_tables(case_all)
    truth_dir = get_e3_truth_region_output_dir()
    os.makedirs(truth_dir, exist_ok=True)
    truth_tables["truth_df"].to_csv(f"{truth_dir}/E3_truth_df.csv", index=False)
    truth_tables["gamma_group_summary"].to_csv(f"{truth_dir}/E3_summary_by_true_gamma_group.csv", index=False)
    truth_tables["R0_group_summary"].to_csv(f"{truth_dir}/E3_summary_by_true_R0_group.csv", index=False)
    truth_tables["beta_group_summary"].to_csv(f"{truth_dir}/E3_summary_by_true_beta_group.csv", index=False)
    truth_tables["severity_gamma_group_summary"].to_csv(f"{truth_dir}/E3_severity_x_true_gamma_group.csv", index=False)
    truth_tables["severity_R0_group_summary"].to_csv(f"{truth_dir}/E3_severity_x_true_R0_group.csv", index=False)
    truth_tables["pivot_R0_cov_by_gamma"].to_csv(f"{truth_dir}/E3_pivot_R0_cov_by_gamma.csv")
    truth_tables["pivot_gamma_cov_by_gamma"].to_csv(f"{truth_dir}/E3_pivot_gamma_cov_by_gamma.csv")
    truth_tables["pivot_R0_cov_by_R0"].to_csv(f"{truth_dir}/E3_pivot_R0_cov_by_R0.csv")
    plot_E3_truth_group_panels(
        truth_tables["gamma_group_summary"],
        truth_tables["R0_group_summary"],
        truth_tables["beta_group_summary"],
        out_dir=truth_dir,
    )

    print(severity_summary_df.round(4))
    print(corr_df.round(4))


if __name__ == "__main__":
    main()
