import os

import numpy as np
import pandas as pd

from .config import TIME_GRID
from .diagnostics import (
    extended_extract_curve_features,
    evaluate_curve_level_ppc,
    evaluate_curve_level_ppc_tau_aligned,
    evaluate_feature_level_ppc,
    evaluate_testset,
    extended_evaluate_feature_level_ppc,
    wilson_interval,
)
from .simulators import (
    rho_t_piecewise,
    simulate_case_E1,
    simulate_case_E2,
    simulate_case_E3,
    simulate_case_matched,
    simulate_curve,
    solve_seir,
    solve_sir,
)
from .workflow import sample_posterior


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_single_value(df, row_key_col, row_key, value_col):
    sub = df[df[row_key_col] == row_key]
    if len(sub) != 1:
        raise ValueError(
            f"Expected exactly one row where {row_key_col} == {row_key}, but got {len(sub)} rows."
        )
    return float(sub.iloc[0][value_col])


def bootstrap_ci(values, stat_func=np.median, n_boot=2000, alpha=0.05, seed=12345):
    vals = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        boot_stats.append(stat_func(sample))
    lower = float(np.quantile(boot_stats, alpha / 2.0))
    upper = float(np.quantile(boot_stats, 1.0 - alpha / 2.0))
    return lower, upper


# E0


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


def run_e0_seed_sensitivity(
    train_seeds=(11, 22, 42),
    eval_seed=2026,
    n_test=100,
    num_posterior_samples=2000,
):
    rows = []
    from .workflow import train_workflow

    for train_seed in train_seeds:
        workflow, _ = train_workflow(seed=train_seed)
        case_df, recovery_df, coverage_df = evaluate_testset(
            workflow=workflow,
            case_simulator=simulate_case_matched,
            n_test=n_test,
            num_posterior_samples=num_posterior_samples,
            seed=eval_seed,
        )
        rows.append({
            "train_seed": train_seed,
            "beta_corr": extract_single_value(recovery_df, "quantity", "beta", "corr_truth_vs_median"),
            "gamma_corr": extract_single_value(recovery_df, "quantity", "gamma", "corr_truth_vs_median"),
            "R0_corr": extract_single_value(recovery_df, "quantity", "R0", "corr_truth_vs_median"),
            "beta_emp90": extract_single_value(coverage_df, "quantity", "beta", "empirical_90"),
            "gamma_emp90": extract_single_value(coverage_df, "quantity", "gamma", "empirical_90"),
            "R0_emp90": extract_single_value(coverage_df, "quantity", "R0", "empirical_90"),
        })
    return pd.DataFrame(rows)


def summarize_empirical_coverage_with_uncertainty(case_df, quantities=("beta", "gamma", "R0"), alpha=0.05):
    rows = []
    for q in quantities:
        k50 = int(case_df[f"{q}_covered50"].sum())
        k90 = int(case_df[f"{q}_covered90"].sum())
        n = len(case_df)
        emp50 = k50 / n
        emp90 = k90 / n
        ci50_low, ci50_high = wilson_interval(k50, n, alpha=alpha)
        ci90_low, ci90_high = wilson_interval(k90, n, alpha=alpha)
        rows.append({
            "quantity": q,
            "n_cases": n,
            "nominal_50": 0.50,
            "k_50": k50,
            "empirical_50": emp50,
            "diff_50": emp50 - 0.50,
            "ci95_50_low": ci50_low,
            "ci95_50_high": ci50_high,
            "nominal_90": 0.90,
            "k_90": k90,
            "empirical_90": emp90,
            "diff_90": emp90 - 0.90,
            "ci95_90_low": ci90_low,
            "ci95_90_high": ci90_high,
        })
    return pd.DataFrame(rows)


def add_latent_curve_features_to_case_df(case_df):
    df = case_df.copy()
    latent_rows = []
    for _, row in df.iterrows():
        latent_curve = solve_sir(row["true_beta"], row["true_gamma"])[:, 1]
        latent_rows.append(extended_extract_curve_features(latent_curve))
    latent_feat_df = pd.DataFrame(latent_rows).add_prefix("latent_")
    return pd.concat([df.reset_index(drop=True), latent_feat_df.reset_index(drop=True)], axis=1)


def summarize_gamma_by_bins(case_df, bin_col, n_bins=4, binning="quantile", alpha=0.05):
    df = case_df.copy()
    if binning == "quantile":
        df["bin"] = pd.qcut(df[bin_col], q=n_bins, duplicates="drop")
    elif binning == "equal_width":
        df["bin"] = pd.cut(df[bin_col], bins=n_bins)
    else:
        raise ValueError("binning must be 'quantile' or 'equal_width'.")

    rows = []
    for bin_value, sub in df.groupby("bin", observed=False):
        n = len(sub)
        k90 = int(sub["gamma_covered90"].sum())
        ci_low, ci_high = wilson_interval(k90, n, alpha=alpha)
        rows.append({
            "bin": str(bin_value),
            "n_cases": n,
            "gamma_MAE_median_est": float(sub["gamma_abs_err_median"].mean()),
            "gamma_bias_median_est": float(sub["gamma_err_median"].mean()),
            "gamma_empirical_90": float(sub["gamma_covered90"].mean()),
            "gamma_ci95_90_low": ci_low,
            "gamma_ci95_90_high": ci_high,
        })
    return pd.DataFrame(rows), df


def summarize_feature_coverage_with_uncertainty(feature_case_df, alpha=0.05):
    rows = []
    for feat_name, sub in feature_case_df.groupby("feature", observed=False):
        n = len(sub)
        k50 = int(sub["covered50"].sum())
        k90 = int(sub["covered90"].sum())
        emp50 = k50 / n
        emp90 = k90 / n
        ci50_low, ci50_high = wilson_interval(k50, n, alpha=alpha)
        ci90_low, ci90_high = wilson_interval(k90, n, alpha=alpha)
        rows.append({
            "feature": feat_name,
            "n_cases": n,
            "nominal_50": 0.50,
            "k_50": k50,
            "empirical_50": emp50,
            "diff_50": emp50 - 0.50,
            "ci95_50_low": ci50_low,
            "ci95_50_high": ci50_high,
            "nominal_90": 0.90,
            "k_90": k90,
            "empirical_90": emp90,
            "diff_90": emp90 - 0.90,
            "ci95_90_low": ci90_low,
            "ci95_90_high": ci90_high,
            "avg_pred_width50": sub["pred_width50"].mean(),
            "avg_pred_width90": sub["pred_width90"].mean(),
        })
    return pd.DataFrame(rows)


# E1


def make_E1_case_simulator(severity=None, reference_mode="full_curve", early_phase_days=12):
    def case_simulator(rng=None):
        return simulate_case_E1(
            rng=rng,
            severity=severity,
            reference_mode=reference_mode,
            early_phase_days=early_phase_days,
        )

    return case_simulator


def run_E1_evaluation(
    workflow,
    severity=None,
    reference_mode="full_curve",
    early_phase_days=12,
    n_test=100,
    num_posterior_samples=2000,
    n_ppc_draws=100,
    seed=2026,
):
    case_simulator = make_E1_case_simulator(
        severity=severity,
        reference_mode=reference_mode,
        early_phase_days=early_phase_days,
    )
    case_df, recovery_df, coverage_df = evaluate_testset(
        workflow=workflow,
        case_simulator=case_simulator,
        n_test=n_test,
        num_posterior_samples=num_posterior_samples,
        seed=seed,
    )
    feature_case_df, feature_summary_df = evaluate_feature_level_ppc(
        workflow=workflow,
        case_simulator=case_simulator,
        predictive_curve_simulator=simulate_curve,
        n_test=n_test,
        num_posterior_samples=min(1000, num_posterior_samples),
        n_ppc_draws=n_ppc_draws,
        seed=seed,
    )
    return case_df, recovery_df, coverage_df, feature_case_df, feature_summary_df


def run_E1_example_case(beta=0.55, gamma=0.18, sigma=0.35, seed=3030):
    rng = np.random.default_rng(seed)
    observed_curve = simulate_case_E1(rng=rng, beta=beta, gamma=gamma, sigma=sigma)["obs"]
    seir_traj = solve_seir(beta=beta, gamma=gamma, sigma=sigma)
    return {
        "beta": beta,
        "gamma": gamma,
        "sigma": sigma,
        "R0": beta / gamma,
        "observed_curve": observed_curve,
        "seir_traj": seir_traj,
        "time_grid": TIME_GRID,
    }


def run_extended_E1_feature_evaluation(
    workflow,
    severity=None,
    reference_mode="full_curve",
    early_phase_days=12,
    n_test=100,
    num_posterior_samples=1000,
    n_ppc_draws=100,
    seed=2026,
    feature_names=None,
    extract_feature_kwargs=None,
):
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

    case_simulator = make_E1_case_simulator(
        severity=severity,
        reference_mode=reference_mode,
        early_phase_days=early_phase_days,
    )
    return extended_evaluate_feature_level_ppc(
        workflow=workflow,
        case_simulator=case_simulator,
        predictive_curve_simulator=simulate_curve,
        n_test=n_test,
        num_posterior_samples=num_posterior_samples,
        n_ppc_draws=n_ppc_draws,
        seed=seed,
        feature_names=feature_names,
        extract_feature_kwargs=extract_feature_kwargs,
    )


def _build_E1_compact_summary_rows(blocks, key_name):
    rows = []
    for block_key, recovery_df, coverage_df, feature_summary_df in blocks:
        rows.append({
            key_name: block_key,
            "beta_bias_median": extract_single_value(recovery_df, "quantity", "beta", "bias_median_est"),
            "beta_MAE": extract_single_value(recovery_df, "quantity", "beta", "MAE_median_est"),
            "R0_MAE": extract_single_value(recovery_df, "quantity", "R0", "MAE_median_est"),
            "beta_empirical_90": extract_single_value(coverage_df, "quantity", "beta", "empirical_90"),
            "gamma_empirical_90": extract_single_value(coverage_df, "quantity", "gamma", "empirical_90"),
            "R0_empirical_90": extract_single_value(coverage_df, "quantity", "R0", "empirical_90"),
            "peak_time_MAE": extract_single_value(feature_summary_df, "feature", "peak_time", "MAE_median_pred"),
            "early_growth_slope_empirical_50": extract_single_value(
                feature_summary_df, "feature", "early_growth_slope", "empirical_50"
            ),
        })
    return pd.DataFrame(rows)


def build_E1_severity_summary(
    mild_recovery_df_E1,
    mild_coverage_df_E1,
    mild_feature_summary_df_E1,
    medium_recovery_df_E1,
    medium_coverage_df_E1,
    medium_feature_summary_df_E1,
    strong_recovery_df_E1,
    strong_coverage_df_E1,
    strong_feature_summary_df_E1,
):
    summary_df = _build_E1_compact_summary_rows(
        [
            ("mild", mild_recovery_df_E1, mild_coverage_df_E1, mild_feature_summary_df_E1),
            ("medium", medium_recovery_df_E1, medium_coverage_df_E1, medium_feature_summary_df_E1),
            ("strong", strong_recovery_df_E1, strong_coverage_df_E1, strong_feature_summary_df_E1),
        ],
        key_name="severity",
    )
    summary_df["severity"] = pd.Categorical(
        summary_df["severity"],
        categories=["mild", "medium", "strong"],
        ordered=True,
    )
    return summary_df.sort_values("severity").reset_index(drop=True)


def build_E1_reference_ablation_summary(
    full_curve_recovery_df_E1,
    full_curve_coverage_df_E1,
    full_curve_feature_summary_df_E1,
    early_phase_recovery_df_E1,
    early_phase_coverage_df_E1,
    early_phase_feature_summary_df_E1,
):
    summary_df = _build_E1_compact_summary_rows(
        [
            ("full_curve", full_curve_recovery_df_E1, full_curve_coverage_df_E1, full_curve_feature_summary_df_E1),
            ("early_phase", early_phase_recovery_df_E1, early_phase_coverage_df_E1, early_phase_feature_summary_df_E1),
        ],
        key_name="reference_mode",
    )
    summary_df["reference_mode"] = pd.Categorical(
        summary_df["reference_mode"],
        categories=["full_curve", "early_phase"],
        ordered=True,
    )
    return summary_df.sort_values("reference_mode").reset_index(drop=True)


def build_E1_posterior_family_ablation_summary(
    flow_recovery_df_E1,
    flow_coverage_df_E1,
    flow_feature_summary_df_E1,
    diag_recovery_df_E1,
    diag_coverage_df_E1,
    diag_feature_summary_df_E1,
):
    summary_df = _build_E1_compact_summary_rows(
        [
            ("flow", flow_recovery_df_E1, flow_coverage_df_E1, flow_feature_summary_df_E1),
            ("diag_gaussian", diag_recovery_df_E1, diag_coverage_df_E1, diag_feature_summary_df_E1),
        ],
        key_name="posterior_family",
    )
    summary_df["posterior_family"] = pd.Categorical(
        summary_df["posterior_family"],
        categories=["flow", "diag_gaussian"],
        ordered=True,
    )
    return summary_df.sort_values("posterior_family").reset_index(drop=True)


def build_E1_results_report(recovery_df, coverage_df, feature_summary_df, curve_summary_df):
    beta_bias = extract_single_value(recovery_df, "quantity", "beta", "bias_median_est")
    gamma_bias = extract_single_value(recovery_df, "quantity", "gamma", "bias_median_est")
    r0_bias = extract_single_value(recovery_df, "quantity", "R0", "bias_median_est")
    beta_cov90 = extract_single_value(coverage_df, "quantity", "beta", "empirical_90")
    gamma_cov90 = extract_single_value(coverage_df, "quantity", "gamma", "empirical_90")
    r0_cov90 = extract_single_value(coverage_df, "quantity", "R0", "empirical_90")
    peak_cov90 = extract_single_value(feature_summary_df, "feature", "peak_time", "empirical_90")
    early_cov50 = extract_single_value(feature_summary_df, "feature", "early_growth_slope", "empirical_50")
    curve_rmse = extract_single_value(curve_summary_df, "metric", "curve_rmse_full", "mean")
    pointwise_cov90 = extract_single_value(curve_summary_df, "metric", "pointwise_empirical_90", "mean")

    return f"""# E1 Structural Misspecification Report

## Setup

- Truth model: SEIR
- Inference model: SIR-trained BayesFlow amortizer
- Purpose: evaluate structural misspecification caused by ignoring the exposed compartment

## Key Findings

- Posterior median bias:
  - beta: {beta_bias:.4f}
  - gamma: {gamma_bias:.4f}
  - R0: {r0_bias:.4f}
- Empirical 90% coverage:
  - beta: {beta_cov90:.4f}
  - gamma: {gamma_cov90:.4f}
  - R0: {r0_cov90:.4f}
- Feature-level PPC:
  - peak_time empirical 90% coverage: {peak_cov90:.4f}
  - early_growth_slope empirical 50% coverage: {early_cov50:.4f}
- Curve-level PPC:
  - mean full-curve RMSE: {curve_rmse:.4f}
  - mean pointwise empirical 90% coverage: {pointwise_cov90:.4f}

## Interpretation

Under E1, the inference model is forced to explain SEIR-generated observations with a simpler SIR family. This primarily tests whether the missing latent exposure stage distorts parameter recovery, narrows posterior intervals too aggressively, or creates predictive mismatches that are more visible in feature-level or curve-level diagnostics than in matched settings.

In practice, the report should be interpreted by comparing transmission-related quantities, recovery-related quantities, and predictive diagnostics together rather than relying on a single metric alone.
"""


# E2


def run_E2_evaluation(
    workflow,
    severity="strong",
    beta_ref_mode="time_avg",
    n_test=100,
    num_posterior_samples=2000,
    n_ppc_draws=100,
    seed=2027,
):
    def case_simulator(rng=None):
        return simulate_case_E2(rng=rng, severity=severity, beta_ref_mode=beta_ref_mode)

    case_df, recovery_df, coverage_df = evaluate_testset(
        workflow=workflow,
        case_simulator=case_simulator,
        n_test=n_test,
        num_posterior_samples=num_posterior_samples,
        seed=seed,
    )
    feature_case_df, feature_summary_df = evaluate_feature_level_ppc(
        workflow=workflow,
        case_simulator=case_simulator,
        predictive_curve_simulator=simulate_curve,
        n_test=n_test,
        num_posterior_samples=min(1000, num_posterior_samples),
        n_ppc_draws=n_ppc_draws,
        seed=seed,
    )
    return case_df, recovery_df, coverage_df, feature_case_df, feature_summary_df


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


def build_E2_severity_summary_with_uncertainty(E2_results, alpha=0.05, n_boot=2000, seed=12345):
    rows = []
    severity_order = ["mild", "medium", "strong"]

    for sev in severity_order:
        recovery_df = E2_results[sev]["recovery_df"]
        coverage_df = E2_results[sev]["coverage_df"]
        feature_summary_df = E2_results[sev]["feature_summary_df"]
        case_df = E2_results[sev]["case_df"]

        beta_emp90 = float(coverage_df.loc[coverage_df["quantity"] == "beta", "empirical_90"].iloc[0])
        r0_emp90 = float(coverage_df.loc[coverage_df["quantity"] == "R0", "empirical_90"].iloc[0])
        beta_ci_low, beta_ci_high = wilson_interval(int(case_df["beta_covered90"].sum()), len(case_df), alpha=alpha)
        r0_ci_low, r0_ci_high = wilson_interval(int(case_df["R0_covered90"].sum()), len(case_df), alpha=alpha)

        beta_bias_vals = case_df["beta_err_median"].to_numpy(dtype=float)
        beta_bias_median = float(np.median(beta_bias_vals))
        beta_bias_ci_low, beta_bias_ci_high = bootstrap_ci(
            beta_bias_vals,
            stat_func=np.median,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed + 17 * (severity_order.index(sev) + 1),
        )

        rows.append({
            "severity": sev,
            "beta_bias_median": beta_bias_median,
            "beta_bias_ci_low": beta_bias_ci_low,
            "beta_bias_ci_high": beta_bias_ci_high,
            "beta_MAE": extract_single_value(recovery_df, "quantity", "beta", "MAE_median_est"),
            "beta_empirical_90": beta_emp90,
            "beta_empirical_90_ci_low": beta_ci_low,
            "beta_empirical_90_ci_high": beta_ci_high,
            "gamma_empirical_90": extract_single_value(coverage_df, "quantity", "gamma", "empirical_90"),
            "R0_empirical_90": r0_emp90,
            "R0_empirical_90_ci_low": r0_ci_low,
            "R0_empirical_90_ci_high": r0_ci_high,
            "peak_time_MAE": extract_single_value(feature_summary_df, "feature", "peak_time", "MAE_median_pred"),
            "early_growth_slope_empirical_50": extract_single_value(
                feature_summary_df, "feature", "early_growth_slope", "empirical_50"
            ),
        })

    summary_df = pd.DataFrame(rows)
    summary_df["severity"] = pd.Categorical(summary_df["severity"], categories=severity_order, ordered=True)
    return summary_df.sort_values("severity").reset_index(drop=True)


def build_E2_reference_sensitivity_summary(workflow, n_test=100, num_posterior_samples=2000, n_ppc_draws=100, seed=2027):
    rows = []
    for beta_ref_mode in ["time_avg", "infection_weighted"]:
        for sev in ["mild", "medium", "strong"]:
            case_df, recovery_df, coverage_df, _, _ = run_E2_evaluation(
                workflow=workflow,
                severity=sev,
                beta_ref_mode=beta_ref_mode,
                n_test=n_test,
                num_posterior_samples=num_posterior_samples,
                n_ppc_draws=n_ppc_draws,
                seed=seed,
            )
            rows.append({
                "severity": sev,
                "beta_ref_mode": beta_ref_mode,
                "beta_bias_median": extract_single_value(recovery_df, "quantity", "beta", "bias_median_est"),
                "beta_empirical_90": extract_single_value(coverage_df, "quantity", "beta", "empirical_90"),
                "R0_empirical_90": extract_single_value(coverage_df, "quantity", "R0", "empirical_90"),
            })

    ref_df = pd.DataFrame(rows)
    ref_df["severity"] = pd.Categorical(ref_df["severity"], categories=["mild", "medium", "strong"], ordered=True)
    return ref_df.sort_values(["beta_ref_mode", "severity"]).reset_index(drop=True)


# E3


def run_E3_example_case(beta=0.75, gamma=0.10, rho0=1.0, tau=18, eta=0.40, severity="strong", seed=4040):
    example_case = simulate_case_E3(
        rng=np.random.default_rng(seed),
        severity=severity,
        beta=beta,
        gamma=gamma,
        rho0=rho0,
        tau=tau,
        eta=eta,
    )
    rho_path = rho_t_piecewise(TIME_GRID, rho0=example_case["rho0"], tau=example_case["tau"], eta=example_case["eta"])
    latent_infected = solve_sir(beta=example_case["true_beta"], gamma=example_case["true_gamma"])[:, 1]
    observed_mean = rho_path * latent_infected
    return {
        "case": example_case,
        "rho_path": rho_path,
        "latent_infected": latent_infected,
        "observed_mean": observed_mean,
        "observed_curve": example_case["obs"][:, 0],
        "time_grid": TIME_GRID,
    }


def build_E3_controlled_cases():
    severity_list = ["mild", "medium", "strong"]
    fixed_eta_by_severity = {
        "mild": 0.90,
        "medium": 0.72,
        "strong": 0.40,
    }
    rows = []
    for row_idx, sev in enumerate(severity_list):
        case = simulate_case_E3(
            rng=np.random.default_rng(5000 + row_idx),
            severity=sev,
            beta=0.75,
            gamma=0.10,
            rho0=1.0,
            tau=18,
            eta=fixed_eta_by_severity[sev],
        )
        rho_path = rho_t_piecewise(TIME_GRID, rho0=case["rho0"], tau=case["tau"], eta=case["eta"])
        latent_infected = solve_sir(case["true_beta"], case["true_gamma"])[:, 1]
        rows.append({
            "severity": sev,
            "case": case,
            "rho_path": rho_path,
            "latent_infected": latent_infected,
            "observed_mean": rho_path * latent_infected,
            "observed_curve": case["obs"][:, 0],
        })
    return rows


def run_E3_evaluation(
    workflow,
    severity="strong",
    n_test=100,
    num_posterior_samples=2000,
    n_ppc_draws=100,
    seed=2028,
):
    def case_simulator(rng=None):
        return simulate_case_E3(rng=rng, severity=severity)

    case_df, recovery_df, coverage_df = evaluate_testset(
        workflow=workflow,
        case_simulator=case_simulator,
        n_test=n_test,
        num_posterior_samples=num_posterior_samples,
        seed=seed,
    )
    feature_case_df, feature_summary_df = evaluate_feature_level_ppc(
        workflow=workflow,
        case_simulator=case_simulator,
        predictive_curve_simulator=simulate_curve,
        n_test=n_test,
        num_posterior_samples=min(1000, num_posterior_samples),
        n_ppc_draws=n_ppc_draws,
        seed=seed,
    )
    return case_df, recovery_df, coverage_df, feature_case_df, feature_summary_df


def build_E3_severity_summary(
    mild_recovery_df_E3,
    mild_coverage_df_E3,
    mild_feature_summary_df_E3,
    medium_recovery_df_E3,
    medium_coverage_df_E3,
    medium_feature_summary_df_E3,
    strong_recovery_df_E3,
    strong_coverage_df_E3,
    strong_feature_summary_df_E3,
):
    rows = []
    severity_blocks = [
        ("mild", mild_recovery_df_E3, mild_coverage_df_E3, mild_feature_summary_df_E3),
        ("medium", medium_recovery_df_E3, medium_coverage_df_E3, medium_feature_summary_df_E3),
        ("strong", strong_recovery_df_E3, strong_coverage_df_E3, strong_feature_summary_df_E3),
    ]

    for severity, recovery_df, coverage_df, feature_summary_df in severity_blocks:
        rows.append({
            "severity": severity,
            "beta_bias_median": extract_single_value(recovery_df, "quantity", "beta", "bias_median_est"),
            "gamma_bias_median": extract_single_value(recovery_df, "quantity", "gamma", "bias_median_est"),
            "R0_bias_median": extract_single_value(recovery_df, "quantity", "R0", "bias_median_est"),
            "beta_empirical_90": extract_single_value(coverage_df, "quantity", "beta", "empirical_90"),
            "gamma_empirical_90": extract_single_value(coverage_df, "quantity", "gamma", "empirical_90"),
            "R0_empirical_90": extract_single_value(coverage_df, "quantity", "R0", "empirical_90"),
            "peak_height_empirical_90": extract_single_value(feature_summary_df, "feature", "peak_height", "empirical_90"),
            "cumulative_infected_empirical_90": extract_single_value(
                feature_summary_df, "feature", "cumulative_infected", "empirical_90"
            ),
            "early_growth_slope_empirical_90": extract_single_value(
                feature_summary_df, "feature", "early_growth_slope", "empirical_90"
            ),
        })

    summary_df = pd.DataFrame(rows)
    summary_df["severity"] = pd.Categorical(summary_df["severity"], categories=["mild", "medium", "strong"], ordered=True)
    return summary_df.sort_values("severity").reset_index(drop=True)


def build_E3_caselevel_correlation_summary(case_df):
    def safe_corr(x, y):
        x = pd.Series(x).astype(float)
        y = pd.Series(y).astype(float)
        if x.nunique() < 2 or y.nunique() < 2:
            return np.nan
        return x.corr(y)

    specs = [
        ("eta", "R0_err_median"),
        ("eta", "gamma_err_median"),
        ("eta", "R0_covered90"),
        ("tau", "R0_err_median"),
        ("tau", "R0_covered90"),
        ("tau", "gamma_err_median"),
    ]

    rows = []
    for x_col, y_col in specs:
        rows.append({
            "group": "pooled",
            "x": x_col,
            "y": y_col,
            "pearson_corr": safe_corr(case_df[x_col], case_df[y_col]),
        })

    for sev in ["mild", "medium", "strong"]:
        sub = case_df[case_df["severity"] == sev].copy()
        for x_col, y_col in specs:
            rows.append({
                "group": sev,
                "x": x_col,
                "y": y_col,
                "pearson_corr": safe_corr(sub[x_col], sub[y_col]),
            })
    return pd.DataFrame(rows)


def add_quantile_bins(df, col, q=4):
    out = df.copy()
    out[f"{col}_bin"] = pd.qcut(out[col], q=q, duplicates="drop")
    return out


def build_E3_eta_tau_bin_summaries(case_df):
    case_binned = add_quantile_bins(case_df, "eta", q=4)
    case_binned = add_quantile_bins(case_binned, "tau", q=4)

    eta_bin_summary = (
        case_binned.groupby("eta_bin", observed=False)
        .agg(
            n_cases=("case_id", "count"),
            eta_mean=("eta", "mean"),
            R0_err_median_mean=("R0_err_median", "mean"),
            gamma_err_median_mean=("gamma_err_median", "mean"),
            R0_covered90_rate=("R0_covered90", "mean"),
            gamma_covered90_rate=("gamma_covered90", "mean"),
        )
        .reset_index()
    )

    tau_bin_summary = (
        case_binned.groupby("tau_bin", observed=False)
        .agg(
            n_cases=("case_id", "count"),
            tau_mean=("tau", "mean"),
            R0_err_median_mean=("R0_err_median", "mean"),
            gamma_err_median_mean=("gamma_err_median", "mean"),
            R0_covered90_rate=("R0_covered90", "mean"),
            gamma_covered90_rate=("gamma_covered90", "mean"),
        )
        .reset_index()
    )

    return case_binned, eta_bin_summary, tau_bin_summary


def add_tertile_group(df, col, labels=("low", "mid", "high")):
    out = df.copy()
    out[f"{col}_group"] = pd.qcut(out[col], q=3, labels=labels, duplicates="drop")
    return out


def summarize_by_truth_group(df, group_col):
    return (
        df.groupby(group_col, observed=False)
        .agg(
            n_cases=("case_id", "count"),
            true_beta_mean=("true_beta", "mean"),
            true_gamma_mean=("true_gamma", "mean"),
            true_R0_mean=("true_R0", "mean"),
            beta_err_median_mean=("beta_err_median", "mean"),
            gamma_err_median_mean=("gamma_err_median", "mean"),
            R0_err_median_mean=("R0_err_median", "mean"),
            beta_covered90_rate=("beta_covered90", "mean"),
            gamma_covered90_rate=("gamma_covered90", "mean"),
            R0_covered90_rate=("R0_covered90", "mean"),
        )
        .reset_index()
    )


def summarize_by_severity_and_group(df, group_col):
    return (
        df.groupby(["severity", group_col], observed=False)
        .agg(
            n_cases=("case_id", "count"),
            gamma_err_median_mean=("gamma_err_median", "mean"),
            R0_err_median_mean=("R0_err_median", "mean"),
            gamma_covered90_rate=("gamma_covered90", "mean"),
            R0_covered90_rate=("R0_covered90", "mean"),
        )
        .reset_index()
    )


def build_E3_truth_region_tables(case_df):
    truth_df = add_tertile_group(case_df, "true_gamma")
    truth_df = add_tertile_group(truth_df, "true_R0")
    truth_df = add_tertile_group(truth_df, "true_beta")

    gamma_group_summary = summarize_by_truth_group(truth_df, "true_gamma_group")
    r0_group_summary = summarize_by_truth_group(truth_df, "true_R0_group")
    beta_group_summary = summarize_by_truth_group(truth_df, "true_beta_group")
    severity_gamma_group_summary = summarize_by_severity_and_group(truth_df, "true_gamma_group")
    severity_r0_group_summary = summarize_by_severity_and_group(truth_df, "true_R0_group")

    pivot_r0_cov_by_gamma = severity_gamma_group_summary.pivot(
        index="severity",
        columns="true_gamma_group",
        values="R0_covered90_rate",
    )
    pivot_gamma_cov_by_gamma = severity_gamma_group_summary.pivot(
        index="severity",
        columns="true_gamma_group",
        values="gamma_covered90_rate",
    )
    pivot_r0_cov_by_r0 = severity_r0_group_summary.pivot(
        index="severity",
        columns="true_R0_group",
        values="R0_covered90_rate",
    )

    return {
        "truth_df": truth_df,
        "gamma_group_summary": gamma_group_summary,
        "R0_group_summary": r0_group_summary,
        "beta_group_summary": beta_group_summary,
        "severity_gamma_group_summary": severity_gamma_group_summary,
        "severity_R0_group_summary": severity_r0_group_summary,
        "pivot_R0_cov_by_gamma": pivot_r0_cov_by_gamma,
        "pivot_gamma_cov_by_gamma": pivot_gamma_cov_by_gamma,
        "pivot_R0_cov_by_R0": pivot_r0_cov_by_r0,
    }
