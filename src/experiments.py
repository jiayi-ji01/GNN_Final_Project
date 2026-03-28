import os

import numpy as np
import pandas as pd

from .config import TIME_GRID
from .diagnostics import evaluate_feature_level_ppc, evaluate_testset
from .simulators import simulate_case_E1, simulate_case_E2, simulate_case_matched, simulate_curve
from .workflow import sample_posterior


def run_E0_evaluation(
    workflow,
    n_test=100,
    num_posterior_samples=2000,
    n_ppc_draws=100,
    seed=2026,
):
    case_df, recovery_df, coverage_df = evaluate_testset(
        workflow=workflow,
        case_simulator=simulate_case_matched,
        n_test=n_test,
        num_posterior_samples=num_posterior_samples,
        seed=seed,
    )
    feature_case_df, feature_summary_df = evaluate_feature_level_ppc(
        workflow=workflow,
        case_simulator=simulate_case_matched,
        predictive_curve_simulator=simulate_curve,
        n_test=n_test,
        num_posterior_samples=min(1000, num_posterior_samples),
        n_ppc_draws=n_ppc_draws,
        seed=seed,
    )
    return case_df, recovery_df, coverage_df, feature_case_df, feature_summary_df


def run_single_case_posterior(
    workflow,
    true_beta=0.55,
    true_gamma=0.18,
    single_case_seed=2025,
    num_posterior_samples=4000,
    n_ppc=100,
    ppc_seed=2024,
):
    true_r0 = true_beta / true_gamma
    single_case_rng = np.random.default_rng(single_case_seed)
    observed_curve = simulate_curve(true_beta, true_gamma, rng=single_case_rng)

    posterior = sample_posterior(
        workflow=workflow,
        obs=observed_curve,
        num_samples=num_posterior_samples,
    )
    posterior_df = pd.DataFrame({
        "beta": posterior["beta"],
        "gamma": posterior["gamma"],
    })
    posterior_df["R0"] = posterior_df["beta"] / posterior_df["gamma"]

    summary_table = posterior_df[["beta", "gamma", "R0"]].describe(
        percentiles=[0.05, 0.5, 0.95]
    ).T
    summary_table = summary_table[["mean", "std", "5%", "50%", "95%"]]

    r0_ci = np.quantile(posterior_df["R0"], [0.05, 0.5, 0.95])
    prob_r0_gt_1 = float((posterior_df["R0"] > 1.0).mean())

    single_ppc_rng = np.random.default_rng(ppc_seed)
    draw_ids = single_ppc_rng.choice(len(posterior_df), size=n_ppc, replace=False)
    ppc_draws = posterior_df.iloc[draw_ids][["beta", "gamma"]].to_numpy()
    ppc_curves = np.stack(
        [simulate_curve(beta, gamma, rng=single_ppc_rng)[:, 0] for beta, gamma in ppc_draws],
        axis=0,
    )

    return {
        "true_beta": true_beta,
        "true_gamma": true_gamma,
        "true_r0": true_r0,
        "observed_curve": observed_curve,
        "posterior_df": posterior_df,
        "summary_table": summary_table,
        "r0_ci": r0_ci,
        "prob_r0_gt_1": prob_r0_gt_1,
        "ppc_curves": ppc_curves,
        "time_grid": TIME_GRID,
    }


def run_E1_evaluation(
    workflow,
    n_test=100,
    num_posterior_samples=2000,
    n_ppc_draws=100,
    seed=2026,
):
    case_df, recovery_df, coverage_df = evaluate_testset(
        workflow=workflow,
        case_simulator=simulate_case_E1,
        n_test=n_test,
        num_posterior_samples=num_posterior_samples,
        seed=seed,
    )
    feature_case_df, feature_summary_df = evaluate_feature_level_ppc(
        workflow=workflow,
        case_simulator=simulate_case_E1,
        predictive_curve_simulator=simulate_curve,
        n_test=n_test,
        num_posterior_samples=min(1000, num_posterior_samples),
        n_ppc_draws=n_ppc_draws,
        seed=seed,
    )
    return case_df, recovery_df, coverage_df, feature_case_df, feature_summary_df


def run_E2_evaluation(
    workflow,
    severity="strong",
    n_test=100,
    num_posterior_samples=2000,
    n_ppc_draws=100,
    seed=2027,
):
    def case_simulator(rng=None):
        return simulate_case_E2(rng=rng, severity=severity)

    case_df_E2, recovery_df_E2, coverage_df_E2 = evaluate_testset(
        workflow=workflow,
        case_simulator=case_simulator,
        n_test=n_test,
        num_posterior_samples=num_posterior_samples,
        seed=seed,
    )
    feature_case_df_E2, feature_summary_df_E2 = evaluate_feature_level_ppc(
        workflow=workflow,
        case_simulator=case_simulator,
        predictive_curve_simulator=simulate_curve,
        n_test=n_test,
        num_posterior_samples=min(1000, num_posterior_samples),
        n_ppc_draws=n_ppc_draws,
        seed=seed,
    )
    return (
        case_df_E2,
        recovery_df_E2,
        coverage_df_E2,
        feature_case_df_E2,
        feature_summary_df_E2,
    )


def run_extended_E2_feature_evaluation(
    workflow,
    severity="medium",
    n_test=100,
    num_posterior_samples=1000,
    n_ppc_draws=100,
    seed=2026,
    feature_names=None,
    extract_feature_kwargs=None,
):
    from .diagnostics import extended_evaluate_feature_level_ppc

    if feature_names is None:
        feature_names = [
            "peak_time",
            "peak_height",
            "cumulative_infected",
            "early_growth_slope",
            "post_peak_decline_slope",
            "late_phase_auc",
            "time_to_half_peak_after_peak",
            "late_to_early_mean_ratio",
        ]

    if extract_feature_kwargs is None:
        extract_feature_kwargs = {
            "early_days": 8,
            "post_peak_days": 8,
            "late_start_frac": 0.5,
        }

    return extended_evaluate_feature_level_ppc(
        workflow=workflow,
        case_simulator=lambda rng=None: simulate_case_E2(rng=rng, severity=severity),
        predictive_curve_simulator=simulate_curve,
        n_test=n_test,
        num_posterior_samples=num_posterior_samples,
        n_ppc_draws=n_ppc_draws,
        seed=seed,
        feature_names=feature_names,
        extract_feature_kwargs=extract_feature_kwargs,
    )


def extract_single_value(df, row_key_col, row_key, value_col):
    sub = df[df[row_key_col] == row_key]
    if len(sub) != 1:
        raise ValueError(
            f"Expected exactly one row where {row_key_col} == {row_key}, but got {len(sub)} rows."
        )
    return float(sub.iloc[0][value_col])


def build_E2_severity_summary(
    mild_recovery_df_E2,
    mild_coverage_df_E2,
    mild_feature_summary_df_E2,
    medium_recovery_df_E2,
    medium_coverage_df_E2,
    medium_feature_summary_df_E2,
    strong_recovery_df_E2,
    strong_coverage_df_E2,
    strong_feature_summary_df_E2,
):
    rows = []
    severity_blocks = [
        ("mild", mild_recovery_df_E2, mild_coverage_df_E2, mild_feature_summary_df_E2),
        ("medium", medium_recovery_df_E2, medium_coverage_df_E2, medium_feature_summary_df_E2),
        ("strong", strong_recovery_df_E2, strong_coverage_df_E2, strong_feature_summary_df_E2),
    ]
    for severity, recovery_df, coverage_df, feature_summary_df in severity_blocks:
        rows.append({
            "severity": severity,
            "beta_bias_median": extract_single_value(recovery_df, "quantity", "beta", "bias_median_est"),
            "beta_MAE": extract_single_value(recovery_df, "quantity", "beta", "MAE_median_est"),
            "beta_empirical_90": extract_single_value(coverage_df, "quantity", "beta", "empirical_90"),
            "gamma_empirical_90": extract_single_value(coverage_df, "quantity", "gamma", "empirical_90"),
            "R0_empirical_90": extract_single_value(coverage_df, "quantity", "R0", "empirical_90"),
            "peak_time_MAE": extract_single_value(feature_summary_df, "feature", "peak_time", "MAE_median_pred"),
            "early_growth_slope_empirical_50": extract_single_value(
                feature_summary_df, "feature", "early_growth_slope", "empirical_50"
            ),
        })

    summary_df = pd.DataFrame(rows)
    severity_order = ["mild", "medium", "strong"]
    summary_df["severity"] = pd.Categorical(summary_df["severity"], categories=severity_order, ordered=True)
    return summary_df.sort_values("severity").reset_index(drop=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
