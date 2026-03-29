import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    get_e2_curve_level_output_dir,
    get_e2_curve_level_tau_output_dir,
    get_e2_reference_sensitivity_output_dir,
    get_e2_severity_output_dir,
    get_e2_severity_summary_output_dir,
    setup_environment,
)
from src.diagnostics import evaluate_curve_level_ppc
from src.experiments import (
    build_E2_reference_sensitivity_summary,
    build_E2_severity_summary,
    build_E2_severity_summary_with_uncertainty,
    run_E2_evaluation,
    run_extended_E2_feature_evaluation,
)
from src.plotting import (
    plot_E2_reference_sensitivity,
    plot_E2_severity_lines,
    plot_curve_level_ppc_summary,
    plot_e2_overall_figure,
    plot_feature_residuals,
)
from src.simulators import simulate_case_E2, simulate_curve
from src.workflow import train_workflow


def main():
    setup_environment(seed=42)
    workflow, history = train_workflow(seed=42)

    outputs = {}
    for severity in ["mild", "medium", "strong"]:
        result = run_E2_evaluation(
            workflow=workflow,
            severity=severity,
            beta_ref_mode="time_avg",
            n_test=100,
            num_posterior_samples=2000,
            n_ppc_draws=100,
            seed=2027,
        )
        outputs[severity] = {
            "case_df": result[0],
            "recovery_df": result[1],
            "coverage_df": result[2],
            "feature_case_df": result[3],
            "feature_summary_df": result[4],
        }

        out_dir = get_e2_severity_output_dir(severity)
        os.makedirs(out_dir, exist_ok=True)
        result[0].to_csv(f"{out_dir}/E2_case_level.csv", index=False)
        result[1].to_csv(f"{out_dir}/E2_recovery_summary.csv", index=False)
        result[2].to_csv(f"{out_dir}/E2_coverage_summary.csv", index=False)
        result[3].to_csv(f"{out_dir}/E2_ppc_feature_case_level.csv", index=False)
        result[4].to_csv(f"{out_dir}/E2_ppc_feature_summary.csv", index=False)

        plot_e2_overall_figure(
            result[0],
            result[2],
            result[4],
            severity_label=severity,
            out_path=f"{out_dir}/E2_overall_figure.png",
        )
        plot_feature_residuals(
            result[3],
            title_prefix=f"E2 {severity}",
        ).savefig(f"{out_dir}/E2_feature_residuals.png", dpi=200, bbox_inches="tight")

        ext_case_df, ext_summary_df = run_extended_E2_feature_evaluation(
            workflow=workflow,
            severity=severity,
            n_test=100,
            num_posterior_samples=1000,
            n_ppc_draws=100,
            seed=2026,
        )
        ext_case_df.to_csv(f"{out_dir}/E2_ppc_feature_case_level_extended.csv", index=False)
        ext_summary_df.to_csv(f"{out_dir}/E2_ppc_feature_summary_extended.csv", index=False)

        curve_case_df, curve_summary_df = evaluate_curve_level_ppc(
            workflow=workflow,
            case_simulator=lambda rng=None, sev=severity: simulate_case_E2(rng=rng, severity=sev),
            predictive_curve_simulator=simulate_curve,
            n_test=100,
            num_posterior_samples=1000,
            n_ppc_draws=100,
            seed=2026,
            split_mode="fraction",
            split_frac=0.5,
        )
        curve_out_dir = get_e2_curve_level_output_dir(severity)
        os.makedirs(curve_out_dir, exist_ok=True)
        curve_case_df.to_csv(f"{curve_out_dir}/E2_curve_case_level_{severity}.csv", index=False)
        curve_summary_df.to_csv(f"{curve_out_dir}/E2_curve_summary_{severity}.csv", index=False)
        plot_curve_level_ppc_summary(
            curve_case_df,
            severity_label=severity,
            experiment_label="E2",
            out_dir=curve_out_dir,
            split_mode="fraction",
        )

        curve_case_df_tau, curve_summary_df_tau = evaluate_curve_level_ppc(
            workflow=workflow,
            case_simulator=lambda rng=None, sev=severity: simulate_case_E2(rng=rng, severity=sev),
            predictive_curve_simulator=simulate_curve,
            n_test=100,
            num_posterior_samples=1000,
            n_ppc_draws=100,
            seed=2026,
            split_mode="tau",
            split_frac=0.5,
        )
        curve_tau_out_dir = get_e2_curve_level_tau_output_dir(severity)
        os.makedirs(curve_tau_out_dir, exist_ok=True)
        curve_case_df_tau.to_csv(f"{curve_tau_out_dir}/E2_curve_case_level_tau_{severity}.csv", index=False)
        curve_summary_df_tau.to_csv(f"{curve_tau_out_dir}/E2_curve_summary_tau_{severity}.csv", index=False)
        plot_curve_level_ppc_summary(
            curve_case_df_tau,
            severity_label=severity,
            experiment_label="E2_tau",
            out_dir=curve_tau_out_dir,
            split_mode="tau",
        )

    summary_df = build_E2_severity_summary(
        mild_recovery_df_E2=outputs["mild"]["recovery_df"],
        mild_coverage_df_E2=outputs["mild"]["coverage_df"],
        mild_feature_summary_df_E2=outputs["mild"]["feature_summary_df"],
        medium_recovery_df_E2=outputs["medium"]["recovery_df"],
        medium_coverage_df_E2=outputs["medium"]["coverage_df"],
        medium_feature_summary_df_E2=outputs["medium"]["feature_summary_df"],
        strong_recovery_df_E2=outputs["strong"]["recovery_df"],
        strong_coverage_df_E2=outputs["strong"]["coverage_df"],
        strong_feature_summary_df_E2=outputs["strong"]["feature_summary_df"],
    )
    summary_unc_df = build_E2_severity_summary_with_uncertainty(outputs)
    severity_summary_dir = get_e2_severity_summary_output_dir()
    os.makedirs(severity_summary_dir, exist_ok=True)
    summary_df.to_csv(f"{severity_summary_dir}/E2_severity_summary.csv", index=False)
    summary_unc_df.to_csv(f"{severity_summary_dir}/E2_severity_summary_with_uncertainty.csv", index=False)
    plot_E2_severity_lines(summary_unc_df, out_dir=severity_summary_dir)

    ref_df = build_E2_reference_sensitivity_summary(
        workflow=workflow,
        n_test=100,
        num_posterior_samples=2000,
        n_ppc_draws=100,
        seed=2027,
    )
    ref_dir = get_e2_reference_sensitivity_output_dir()
    os.makedirs(ref_dir, exist_ok=True)
    ref_df.to_csv(f"{ref_dir}/E2_reference_sensitivity_summary.csv", index=False)
    plot_E2_reference_sensitivity(ref_df, out_dir=ref_dir)

    print(summary_unc_df.round(4))
    print(ref_df.round(4))


if __name__ == "__main__":
    main()
