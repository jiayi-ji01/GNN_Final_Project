import numpy as np
import pandas as pd

from .config import PRIOR_BOUNDS, TIME_GRID
from .workflow import from_unconstrained


def summarize_posterior_samples(beta_samples, gamma_samples):
    beta_samples = np.asarray(beta_samples).reshape(-1)
    gamma_samples = np.asarray(gamma_samples).reshape(-1)
    r0_samples = beta_samples / gamma_samples

    out = {}
    for name, arr in [("beta", beta_samples), ("gamma", gamma_samples), ("R0", r0_samples)]:
        q25, q50, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
        q05, q95 = np.quantile(arr, [0.05, 0.95])
        out[f"{name}_mean"] = float(np.mean(arr))
        out[f"{name}_median"] = float(q50)
        out[f"{name}_std"] = float(np.std(arr, ddof=1))
        out[f"{name}_ci50_low"] = float(q25)
        out[f"{name}_ci50_high"] = float(q75)
        out[f"{name}_ci90_low"] = float(q05)
        out[f"{name}_ci90_high"] = float(q95)
        out[f"{name}_width50"] = float(q75 - q25)
        out[f"{name}_width90"] = float(q95 - q05)
    return out


def evaluate_testset(
    workflow,
    case_simulator,
    n_test=50,
    num_posterior_samples=2000,
    seed=123,
    truth_keys=("beta", "gamma", "R0"),
):
    rng = np.random.default_rng(seed)
    rows = []

    for case_id in range(n_test):
        case = case_simulator(rng=rng)
        obs = case["obs"]

        samples = workflow.sample(
            num_samples=num_posterior_samples,
            conditions={"obs": obs[None, ...]},
            split=True,
        )

        z_beta_samps = np.asarray(samples["z_beta"]).reshape(-1)
        z_gamma_samps = np.asarray(samples["z_gamma"]).reshape(-1)
        beta_samps = from_unconstrained(z_beta_samps, *PRIOR_BOUNDS["beta"])
        gamma_samps = from_unconstrained(z_gamma_samps, *PRIOR_BOUNDS["gamma"])

        row = {"case_id": case_id}
        for k, v in case.items():
            if k != "obs":
                row[k] = v
        row.update(summarize_posterior_samples(beta_samps, gamma_samps))

        for q in truth_keys:
            true_key = f"true_{q}"
            if true_key in row:
                true_val = row[true_key]
                row[f"{q}_covered50"] = float(row[f"{q}_ci50_low"] <= true_val <= row[f"{q}_ci50_high"])
                row[f"{q}_covered90"] = float(row[f"{q}_ci90_low"] <= true_val <= row[f"{q}_ci90_high"])
                row[f"{q}_err_mean"] = row[f"{q}_mean"] - true_val
                row[f"{q}_err_median"] = row[f"{q}_median"] - true_val
                row[f"{q}_abs_err_median"] = abs(row[f"{q}_median"] - true_val)
                row[f"{q}_sq_err_median"] = (row[f"{q}_median"] - true_val) ** 2

        rows.append(row)

    case_df = pd.DataFrame(rows)

    recovery_rows = []
    for q in truth_keys:
        true_key = f"true_{q}"
        if true_key in case_df.columns:
            recovery_rows.append({
                "quantity": q,
                "bias_mean_est": case_df[f"{q}_err_mean"].mean(),
                "bias_median_est": case_df[f"{q}_err_median"].mean(),
                "MAE_median_est": case_df[f"{q}_abs_err_median"].mean(),
                "RMSE_median_est": np.sqrt(case_df[f"{q}_sq_err_median"].mean()),
                "corr_truth_vs_median": case_df[true_key].corr(case_df[f"{q}_median"]),
                "avg_post_std": case_df[f"{q}_std"].mean(),
            })
    recovery_df = pd.DataFrame(recovery_rows)

    coverage_rows = []
    for q in truth_keys:
        true_key = f"true_{q}"
        if true_key in case_df.columns:
            empirical_50 = case_df[f"{q}_covered50"].mean()
            empirical_90 = case_df[f"{q}_covered90"].mean()
            coverage_rows.append({
                "quantity": q,
                "nominal_50": 0.50,
                "empirical_50": empirical_50,
                "diff_50": empirical_50 - 0.50,
                "nominal_90": 0.90,
                "empirical_90": empirical_90,
                "diff_90": empirical_90 - 0.90,
                "avg_width50": case_df[f"{q}_width50"].mean(),
                "avg_width90": case_df[f"{q}_width90"].mean(),
            })
    coverage_df = pd.DataFrame(coverage_rows)
    return case_df, recovery_df, coverage_df


def extract_curve_features(curve_1d, time_grid=TIME_GRID, early_days=8):
    y = np.asarray(curve_1d).reshape(-1)
    t = np.asarray(time_grid).reshape(-1)
    peak_idx = int(np.argmax(y))
    peak_time = float(t[peak_idx])
    peak_height = float(y[peak_idx])
    cumulative_infected = float(np.trapz(y, t))
    k = min(early_days, len(y))
    slope = float(np.polyfit(t[:k], y[:k], deg=1)[0])
    return {
        "peak_time": peak_time,
        "peak_height": peak_height,
        "cumulative_infected": cumulative_infected,
        "early_growth_slope": slope,
    }


def evaluate_feature_level_ppc(
    workflow,
    case_simulator,
    predictive_curve_simulator,
    n_test=50,
    num_posterior_samples=1000,
    n_ppc_draws=100,
    seed=2026,
    feature_names=None,
):
    rng = np.random.default_rng(seed)
    rows = []

    if feature_names is None:
        feature_names = [
            "peak_time",
            "peak_height",
            "cumulative_infected",
            "early_growth_slope",
        ]

    for case_id in range(n_test):
        case = case_simulator(rng=rng)
        obs_full = case["obs"]
        obs = obs_full[:, 0]
        obs_feats = extract_curve_features(obs)

        samples = workflow.sample(
            num_samples=num_posterior_samples,
            conditions={"obs": obs_full[None, ...]},
            split=True,
        )
        z_beta_samps = np.asarray(samples["z_beta"]).reshape(-1)
        z_gamma_samps = np.asarray(samples["z_gamma"]).reshape(-1)
        beta_samps = from_unconstrained(z_beta_samps, *PRIOR_BOUNDS["beta"])
        gamma_samps = from_unconstrained(z_gamma_samps, *PRIOR_BOUNDS["gamma"])

        draw_idx = rng.choice(len(beta_samps), size=n_ppc_draws, replace=False)
        beta_draws = beta_samps[draw_idx]
        gamma_draws = gamma_samps[draw_idx]
        ppc_feature_rows = {k: [] for k in obs_feats.keys()}

        for b, g in zip(beta_draws, gamma_draws):
            pred_curve = predictive_curve_simulator(float(b), float(g), rng=rng)[:, 0]
            pred_feats = extract_curve_features(pred_curve)
            for feat_name, feat_val in pred_feats.items():
                ppc_feature_rows[feat_name].append(feat_val)

        for feat_name in feature_names:
            obs_val = obs_feats[feat_name]
            pred_vals = np.asarray(ppc_feature_rows[feat_name], dtype=float)
            q25, q50, q75 = np.quantile(pred_vals, [0.25, 0.50, 0.75])
            q05, q95 = np.quantile(pred_vals, [0.05, 0.95])
            row = {
                "case_id": case_id,
                "feature": feat_name,
                "observed_value": float(obs_val),
                "pred_mean": float(pred_vals.mean()),
                "pred_median": float(q50),
                "pred_std": float(pred_vals.std(ddof=1)),
                "pred_ci50_low": float(q25),
                "pred_ci50_high": float(q75),
                "pred_ci90_low": float(q05),
                "pred_ci90_high": float(q95),
                "pred_width50": float(q75 - q25),
                "pred_width90": float(q95 - q05),
                "covered50": float(q25 <= obs_val <= q75),
                "covered90": float(q05 <= obs_val <= q95),
                "err_mean": float(pred_vals.mean() - obs_val),
                "err_median": float(q50 - obs_val),
                "abs_err_median": float(abs(q50 - obs_val)),
                "sq_err_median": float((q50 - obs_val) ** 2),
            }
            for k, v in case.items():
                if k != "obs":
                    row[k] = v
            rows.append(row)

    feature_case_df = pd.DataFrame(rows)
    summary_rows = []
    for feat_name in feature_names:
        sub = feature_case_df[feature_case_df["feature"] == feat_name].copy()
        empirical_50 = sub["covered50"].mean()
        empirical_90 = sub["covered90"].mean()
        summary_rows.append({
            "feature": feat_name,
            "nominal_50": 0.50,
            "empirical_50": empirical_50,
            "diff_50": empirical_50 - 0.50,
            "nominal_90": 0.90,
            "empirical_90": empirical_90,
            "diff_90": empirical_90 - 0.90,
            "bias_mean_pred": sub["err_mean"].mean(),
            "bias_median_pred": sub["err_median"].mean(),
            "MAE_median_pred": sub["abs_err_median"].mean(),
            "RMSE_median_pred": np.sqrt(sub["sq_err_median"].mean()),
            "avg_pred_std": sub["pred_std"].mean(),
            "avg_width50": sub["pred_width50"].mean(),
            "avg_width90": sub["pred_width90"].mean(),
        })

    return feature_case_df, pd.DataFrame(summary_rows)


def extended_extract_curve_features(
    curve_1d,
    time_grid=TIME_GRID,
    early_days=8,
    post_peak_days=8,
    late_start_frac=0.5,
    eps=1e-8,
):
    y = np.asarray(curve_1d).reshape(-1).astype(float)
    t = np.asarray(time_grid).reshape(-1).astype(float)
    if len(y) != len(t):
        raise ValueError("curve_1d and time_grid must have the same length.")

    n = len(y)
    peak_idx = int(np.argmax(y))
    peak_time = float(t[peak_idx])
    peak_height = float(y[peak_idx])
    cumulative_infected = float(np.trapz(y, t))

    k_early = min(early_days, n)
    if k_early >= 2:
        early_growth_slope = float(np.polyfit(t[:k_early], y[:k_early], deg=1)[0])
    else:
        early_growth_slope = 0.0

    post_start = peak_idx
    post_end = min(peak_idx + post_peak_days + 1, n)
    x_post = t[post_start:post_end]
    y_post = y[post_start:post_end]
    if len(x_post) >= 2:
        post_peak_decline_slope = float(np.polyfit(x_post, y_post, deg=1)[0])
    else:
        post_peak_decline_slope = 0.0

    late_start_idx = int(np.floor(late_start_frac * n))
    late_start_idx = min(max(late_start_idx, 0), n - 1)
    late_phase_auc = float(np.trapz(y[late_start_idx:], t[late_start_idx:]))

    half_peak = 0.5 * peak_height
    after_peak = y[peak_idx:]
    below_half_idx = np.where(after_peak <= half_peak)[0]
    if len(below_half_idx) > 0:
        first_below = int(below_half_idx[0])
        time_to_half_peak_after_peak = float(t[peak_idx + first_below] - t[peak_idx])
    else:
        time_to_half_peak_after_peak = float(t[-1] - t[peak_idx])

    early_end_idx = max(1, int(np.floor((1.0 - late_start_frac) * n)))
    early_mean = float(np.mean(y[:early_end_idx]))
    late_mean = float(np.mean(y[late_start_idx:]))
    late_to_early_mean_ratio = float(late_mean / (early_mean + eps))

    return {
        "peak_time": peak_time,
        "peak_height": peak_height,
        "cumulative_infected": cumulative_infected,
        "early_growth_slope": early_growth_slope,
        "post_peak_decline_slope": post_peak_decline_slope,
        "late_phase_auc": late_phase_auc,
        "time_to_half_peak_after_peak": time_to_half_peak_after_peak,
        "late_to_early_mean_ratio": late_to_early_mean_ratio,
    }


def extended_evaluate_feature_level_ppc(
    workflow,
    case_simulator,
    predictive_curve_simulator,
    n_test=50,
    num_posterior_samples=1000,
    n_ppc_draws=100,
    seed=2026,
    feature_names=None,
    extract_feature_kwargs=None,
):
    rng = np.random.default_rng(seed)
    rows = []
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
        extract_feature_kwargs = {}

    for case_id in range(n_test):
        case = case_simulator(rng=rng)
        obs_full = case["obs"]
        obs = obs_full[:, 0]
        obs_feats = extended_extract_curve_features(obs, **extract_feature_kwargs)

        samples = workflow.sample(
            num_samples=num_posterior_samples,
            conditions={"obs": obs_full[None, ...]},
            split=True,
        )
        z_beta_samps = np.asarray(samples["z_beta"]).reshape(-1)
        z_gamma_samps = np.asarray(samples["z_gamma"]).reshape(-1)
        beta_samps = from_unconstrained(z_beta_samps, *PRIOR_BOUNDS["beta"])
        gamma_samps = from_unconstrained(z_gamma_samps, *PRIOR_BOUNDS["gamma"])

        n_available = len(beta_samps)
        n_draws = min(n_ppc_draws, n_available)
        draw_idx = rng.choice(n_available, size=n_draws, replace=False)
        beta_draws = beta_samps[draw_idx]
        gamma_draws = gamma_samps[draw_idx]
        ppc_feature_rows = {k: [] for k in obs_feats.keys()}

        for b, g in zip(beta_draws, gamma_draws):
            pred_curve = predictive_curve_simulator(float(b), float(g), rng=rng)[:, 0]
            pred_feats = extended_extract_curve_features(pred_curve, **extract_feature_kwargs)
            for feat_name, feat_val in pred_feats.items():
                ppc_feature_rows[feat_name].append(feat_val)

        for feat_name in feature_names:
            obs_val = obs_feats[feat_name]
            pred_vals = np.asarray(ppc_feature_rows[feat_name], dtype=float)
            q25, q50, q75 = np.quantile(pred_vals, [0.25, 0.50, 0.75])
            q05, q95 = np.quantile(pred_vals, [0.05, 0.95])
            row = {
                "case_id": case_id,
                "feature": feat_name,
                "observed_value": float(obs_val),
                "pred_mean": float(pred_vals.mean()),
                "pred_median": float(q50),
                "pred_std": float(pred_vals.std(ddof=1)),
                "pred_ci50_low": float(q25),
                "pred_ci50_high": float(q75),
                "pred_ci90_low": float(q05),
                "pred_ci90_high": float(q95),
                "pred_width50": float(q75 - q25),
                "pred_width90": float(q95 - q05),
                "covered50": float(q25 <= obs_val <= q75),
                "covered90": float(q05 <= obs_val <= q95),
                "err_mean": float(pred_vals.mean() - obs_val),
                "err_median": float(q50 - obs_val),
                "abs_err_median": float(abs(q50 - obs_val)),
                "sq_err_median": float((q50 - obs_val) ** 2),
            }
            for k, v in case.items():
                if k != "obs":
                    row[k] = v
            rows.append(row)

    feature_case_df = pd.DataFrame(rows)
    summary_rows = []
    for feat_name in feature_names:
        sub = feature_case_df[feature_case_df["feature"] == feat_name].copy()
        empirical_50 = sub["covered50"].mean()
        empirical_90 = sub["covered90"].mean()
        summary_rows.append({
            "feature": feat_name,
            "nominal_50": 0.50,
            "empirical_50": empirical_50,
            "diff_50": empirical_50 - 0.50,
            "nominal_90": 0.90,
            "empirical_90": empirical_90,
            "diff_90": empirical_90 - 0.90,
            "bias_mean_pred": sub["err_mean"].mean(),
            "bias_median_pred": sub["err_median"].mean(),
            "MAE_median_pred": sub["abs_err_median"].mean(),
            "RMSE_median_pred": np.sqrt(sub["sq_err_median"].mean()),
            "avg_pred_std": sub["pred_std"].mean(),
            "avg_width50": sub["pred_width50"].mean(),
            "avg_width90": sub["pred_width90"].mean(),
        })
    return feature_case_df, pd.DataFrame(summary_rows)


def summarize_predictive_curves(ppc_curves):
    ppc_curves = np.asarray(ppc_curves, dtype=float)
    return {
        "mean": np.mean(ppc_curves, axis=0),
        "median": np.median(ppc_curves, axis=0),
        "q05": np.quantile(ppc_curves, 0.05, axis=0),
        "q25": np.quantile(ppc_curves, 0.25, axis=0),
        "q75": np.quantile(ppc_curves, 0.75, axis=0),
        "q95": np.quantile(ppc_curves, 0.95, axis=0),
    }


def compute_curve_discrepancy(observed_curve, predictive_median_curve, split_frac=0.5):
    y_obs = np.asarray(observed_curve, dtype=float).reshape(-1)
    y_pred = np.asarray(predictive_median_curve, dtype=float).reshape(-1)
    if len(y_obs) != len(y_pred):
        raise ValueError("observed_curve and predictive_median_curve must have same length.")

    err = y_pred - y_obs
    abs_err = np.abs(err)
    sq_err = err ** 2
    n = len(y_obs)
    split_idx = int(np.floor(split_frac * n))
    split_idx = min(max(split_idx, 1), n - 1)

    err_early = err[:split_idx]
    err_late = err[split_idx:]
    abs_err_early = np.abs(err_early)
    abs_err_late = np.abs(err_late)
    sq_err_early = err_early ** 2
    sq_err_late = err_late ** 2

    return {
        "curve_bias_full": float(np.mean(err)),
        "curve_mae_full": float(np.mean(abs_err)),
        "curve_rmse_full": float(np.sqrt(np.mean(sq_err))),
        "curve_bias_early": float(np.mean(err_early)),
        "curve_mae_early": float(np.mean(abs_err_early)),
        "curve_rmse_early": float(np.sqrt(np.mean(sq_err_early))),
        "curve_bias_late": float(np.mean(err_late)),
        "curve_mae_late": float(np.mean(abs_err_late)),
        "curve_rmse_late": float(np.sqrt(np.mean(sq_err_late))),
        "late_minus_early_rmse": float(np.sqrt(np.mean(sq_err_late)) - np.sqrt(np.mean(sq_err_early))),
        "late_minus_early_mae": float(np.mean(abs_err_late) - np.mean(abs_err_early)),
    }


def evaluate_curve_level_ppc(
    workflow,
    case_simulator,
    predictive_curve_simulator,
    n_test=50,
    num_posterior_samples=1000,
    n_ppc_draws=100,
    seed=2026,
    split_frac=0.5,
):
    rng = np.random.default_rng(seed)
    rows = []

    for case_id in range(n_test):
        case = case_simulator(rng=rng)
        obs_full = case["obs"]
        obs = obs_full[:, 0]

        samples = workflow.sample(
            num_samples=num_posterior_samples,
            conditions={"obs": obs_full[None, ...]},
            split=True,
        )
        z_beta_samps = np.asarray(samples["z_beta"]).reshape(-1)
        z_gamma_samps = np.asarray(samples["z_gamma"]).reshape(-1)
        beta_samps = from_unconstrained(z_beta_samps, *PRIOR_BOUNDS["beta"])
        gamma_samps = from_unconstrained(z_gamma_samps, *PRIOR_BOUNDS["gamma"])

        n_available = len(beta_samps)
        n_draws = min(n_ppc_draws, n_available)
        draw_idx = rng.choice(n_available, size=n_draws, replace=False)
        beta_draws = beta_samps[draw_idx]
        gamma_draws = gamma_samps[draw_idx]

        ppc_curves = np.stack(
            [predictive_curve_simulator(float(b), float(g), rng=rng)[:, 0] for b, g in zip(beta_draws, gamma_draws)],
            axis=0,
        )
        curve_summary = summarize_predictive_curves(ppc_curves)
        pred_median_curve = curve_summary["median"]
        covered50_by_time = ((curve_summary["q25"] <= obs) & (obs <= curve_summary["q75"])).astype(float)
        covered90_by_time = ((curve_summary["q05"] <= obs) & (obs <= curve_summary["q95"])).astype(float)
        disc = compute_curve_discrepancy(obs, pred_median_curve, split_frac=split_frac)

        row = {
            "case_id": case_id,
            "pointwise_empirical_50": float(np.mean(covered50_by_time)),
            "pointwise_empirical_90": float(np.mean(covered90_by_time)),
        }
        row.update(disc)
        for k, v in case.items():
            if k != "obs":
                row[k] = v
        rows.append(row)

    curve_case_df = pd.DataFrame(rows)
    metric_names = [
        "pointwise_empirical_50",
        "pointwise_empirical_90",
        "curve_bias_full",
        "curve_mae_full",
        "curve_rmse_full",
        "curve_bias_early",
        "curve_mae_early",
        "curve_rmse_early",
        "curve_bias_late",
        "curve_mae_late",
        "curve_rmse_late",
        "late_minus_early_rmse",
        "late_minus_early_mae",
    ]

    summary_rows = []
    for metric in metric_names:
        vals = curve_case_df[metric].to_numpy(dtype=float)
        summary_rows.append({
            "metric": metric,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "median": float(np.median(vals)),
            "q05": float(np.quantile(vals, 0.05)),
            "q95": float(np.quantile(vals, 0.95)),
        })
    return curve_case_df, pd.DataFrame(summary_rows)
