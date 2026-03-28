import os

from src.config import (
    get_e2_curve_level_output_dir,
    get_e2_severity_output_dir,
    get_e2_severity_summary_output_dir,
    setup_environment,
)
from src.diagnostics import evaluate_curve_level_ppc
from src.experiments import build_E2_severity_summary, run_E2_evaluation
from src.plotting import plot_E2_severity_lines, plot_curve_level_ppc_summary
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
            n_test=100,
            num_posterior_samples=2000,
            n_ppc_draws=100,
            seed=2027,
        )
        outputs[severity] = result

        case_df, recovery_df, coverage_df, feature_case_df, feature_summary_df = result
        out_dir = get_e2_severity_output_dir(severity)
        os.makedirs(out_dir, exist_ok=True)
        case_df.to_csv(f"{out_dir}/E2_case_level.csv", index=False)
        recovery_df.to_csv(f"{out_dir}/E2_recovery_summary.csv", index=False)
        coverage_df.to_csv(f"{out_dir}/E2_coverage_summary.csv", index=False)
        feature_case_df.to_csv(f"{out_dir}/E2_ppc_feature_case_level.csv", index=False)
        feature_summary_df.to_csv(f"{out_dir}/E2_ppc_feature_summary.csv", index=False)

        curve_case_df, curve_summary_df = evaluate_curve_level_ppc(
            workflow=workflow,
            case_simulator=lambda rng=None, sev=severity: simulate_case_E2(rng=rng, severity=sev),
            predictive_curve_simulator=simulate_curve,
            n_test=100,
            num_posterior_samples=1000,
            n_ppc_draws=100,
            seed=2026,
            split_frac=0.5,
        )
        curve_out_dir = get_e2_curve_level_output_dir(severity)
        os.makedirs(curve_out_dir, exist_ok=True)
        curve_case_df.to_csv(f"{curve_out_dir}/E2_curve_case_level_{severity}.csv", index=False)
        curve_summary_df.to_csv(f"{curve_out_dir}/E2_curve_summary_{severity}.csv", index=False)
        plot_curve_level_ppc_summary(curve_case_df, severity_label=severity, out_dir=curve_out_dir)

    summary_df = build_E2_severity_summary(
        mild_recovery_df_E2=outputs["mild"][1],
        mild_coverage_df_E2=outputs["mild"][2],
        mild_feature_summary_df_E2=outputs["mild"][4],
        medium_recovery_df_E2=outputs["medium"][1],
        medium_coverage_df_E2=outputs["medium"][2],
        medium_feature_summary_df_E2=outputs["medium"][4],
        strong_recovery_df_E2=outputs["strong"][1],
        strong_coverage_df_E2=outputs["strong"][2],
        strong_feature_summary_df_E2=outputs["strong"][4],
    )
    severity_summary_dir = get_e2_severity_summary_output_dir()
    os.makedirs(severity_summary_dir, exist_ok=True)
    summary_df.to_csv(f"{severity_summary_dir}/E2_severity_summary.csv", index=False)
    plot_E2_severity_lines(summary_df, out_dir=severity_summary_dir)
    print(summary_df.round(4))


if __name__ == "__main__":
    main()
