import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import get_e2_severity_summary_output_dir, get_e3_severity_summary_output_dir


def plot_training_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], marker="o", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.show()


def plot_observed_curve(observed_curve, time_grid, true_beta=None, true_gamma=None, true_r0=None):
    plt.figure(figsize=(10, 4))
    plt.plot(time_grid, observed_curve[:, 0], lw=2)
    plt.xlabel("Day")
    plt.ylabel("Infected fraction")
    plt.title("Observed infected-fraction curve")
    plt.tight_layout()
    plt.show()

    if true_beta is not None:
        print("True beta:", true_beta)
    if true_gamma is not None:
        print("True gamma:", true_gamma)
    if true_r0 is not None:
        print("True R0:", round(true_r0, 3))


def plot_single_case_posterior(posterior_df, true_beta, true_gamma, true_r0):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.histplot(posterior_df["beta"], kde=True, ax=axes[0], color="#1f77b4")
    axes[0].axvline(true_beta, color="black", linestyle="--", label="true beta")
    axes[0].set_title("Posterior of beta")
    axes[0].legend()

    sns.histplot(posterior_df["gamma"], kde=True, ax=axes[1], color="#ff7f0e")
    axes[1].axvline(true_gamma, color="black", linestyle="--", label="true gamma")
    axes[1].set_title("Posterior of gamma")
    axes[1].legend()

    sns.histplot(posterior_df["R0"], kde=True, ax=axes[2], color="#2ca02c")
    axes[2].axvline(true_r0, color="black", linestyle="--", label="true R0")
    axes[2].axvline(1.0, color="#c44e52", linestyle=":", label="R0 = 1")
    axes[2].set_title("Posterior of R0")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    sns.jointplot(data=posterior_df, x="beta", y="gamma", kind="kde", fill=True)
    plt.show()


def plot_single_case_ppc(ppc_curves, observed_curve, time_grid):
    plt.figure(figsize=(10, 5))
    for curve in ppc_curves:
        plt.plot(time_grid, curve, color="#4c72b0", alpha=0.08)
    plt.plot(time_grid, observed_curve[:, 0], color="black", lw=2.5, label="observed curve")
    plt.xlabel("Day")
    plt.ylabel("Infected fraction")
    plt.title("Posterior predictive check")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_e1_example_case(seir_traj, observed_curve, time_grid, sigma, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(time_grid, seir_traj[:, 1], lw=2, label="E(t)")
    axes[0].plot(time_grid, seir_traj[:, 2], lw=2, label="I(t)")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Fraction")
    axes[0].set_title(f"E1 SEIR latent dynamics | sigma={sigma:.3f}")
    axes[0].legend()

    axes[1].plot(time_grid, observed_curve[:, 0], lw=2)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Infected fraction")
    axes[1].set_title("Observed curve from SEIR model")

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_e1_overall_figure(case_df, coverage_df, feature_summary_df, severity_label="medium", out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    x = case_df["true_beta"]
    y = case_df["beta_median"]
    ax.scatter(x, y, alpha=0.7, color="#1f77b4")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("Pseudo-true SIR beta")
    ax.set_ylabel("Posterior median beta")
    ax.set_title(f"E1 {severity_label}: beta recovery")

    ax = axes[0, 1]
    x = case_df["true_gamma"]
    y = case_df["gamma_median"]
    ax.scatter(x, y, alpha=0.7, color="#ff7f0e")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("Pseudo-true SIR gamma")
    ax.set_ylabel("Posterior median gamma")
    ax.set_title(f"E1 {severity_label}: gamma recovery")

    ax = axes[1, 0]
    param_cov_plot = coverage_df.melt(
        id_vars="quantity",
        value_vars=["empirical_50", "empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.barplot(data=param_cov_plot, x="quantity", y="coverage", hue="interval", ax=ax)
    ax.axhline(0.50, color="gray", linestyle="--", lw=1.0)
    ax.axhline(0.90, color="gray", linestyle=":", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Coverage")
    ax.set_title(f"E1 {severity_label}: parameter coverage")

    ax = axes[1, 1]
    feat_cov_plot = feature_summary_df.melt(
        id_vars="feature",
        value_vars=["empirical_50", "empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.barplot(data=feat_cov_plot, x="feature", y="coverage", hue="interval", ax=ax)
    ax.axhline(0.50, color="gray", linestyle="--", lw=1.0)
    ax.axhline(0.90, color="gray", linestyle=":", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Coverage")
    ax.set_title(f"E1 {severity_label}: feature-level PPC coverage")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E1_severity_single_line(summary_df, metric, out_path=None):
    configs = {
        "beta_empirical_90": {
            "title": "E1: beta empirical 90% coverage vs severity",
            "ylabel": "Empirical 90% coverage",
            "ref": 0.90,
            "ylim": (0, 1.02),
        },
        "R0_empirical_90": {
            "title": "E1: R0 empirical 90% coverage vs severity",
            "ylabel": "Empirical 90% coverage",
            "ref": 0.90,
            "ylim": (0, 1.02),
        },
        "beta_bias_median": {
            "title": "E1: beta bias (median estimator) vs severity",
            "ylabel": "Bias of posterior median beta",
            "ref": 0.0,
            "ylim": None,
        },
    }
    cfg = configs[metric]
    plt.figure(figsize=(6, 4))
    plt.plot(summary_df["severity"].astype(str), summary_df[metric], marker="o", lw=2)
    plt.axhline(cfg["ref"], color="gray", linestyle="--", lw=1.2)
    plt.xlabel("Severity")
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title"])
    if cfg["ylim"] is not None:
        plt.ylim(*cfg["ylim"])
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_E1_severity_3panel_summary(summary_df, out_path=None):
    x_labels = summary_df["severity"].astype(str).tolist()
    x = np.arange(len(x_labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    panels = [
        ("beta_empirical_90", "beta empirical 90%", 0.90, "Empirical 90% coverage", (0, 1.02)),
        ("R0_empirical_90", "R0 empirical 90%", 0.90, "Empirical 90% coverage", (0, 1.02)),
        ("beta_bias_median", "beta bias (median)", 0.0, "Bias of posterior median beta", None),
    ]
    for ax, (col, title, ref_line, ylab, ylim) in zip(axes, panels):
        y = summary_df[col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=2)
        ax.axhline(ref_line, color="gray", linestyle="--", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Severity")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E1_ablation_comparison(summary_df, group_col, title_prefix="E1 ablation", out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    panels = [
        ("beta_empirical_90", "beta empirical 90% coverage", 0.90, "Coverage"),
        ("R0_empirical_90", "R0 empirical 90% coverage", 0.90, "Coverage"),
        ("beta_bias_median", "beta bias (median)", 0.0, "Bias"),
        ("gamma_empirical_90", "gamma empirical 90% coverage", 0.90, "Coverage"),
    ]

    x_order = summary_df[group_col].astype(str).drop_duplicates().tolist()
    for ax, (metric, title, ref_line, ylab) in zip(axes, panels):
        sns.barplot(data=summary_df, x=group_col, y=metric, order=x_order, ax=ax, color="#4c72b0")
        ax.axhline(ref_line, color="gray", linestyle="--", lw=1.2)
        ax.set_title(f"{title_prefix}: {title}")
        ax.set_xlabel(group_col)
        ax.set_ylabel(ylab)
        if "coverage" in title:
            ax.set_ylim(0, 1.02)
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_e2_example_case(example_case_E2, time_grid, beta_path, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(time_grid, beta_path, lw=2)
    axes[0].axvline(example_case_E2["tau"], color="black", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("beta(t)")
    axes[0].set_title(
        f"E2 beta(t) schedule | severity={example_case_E2['severity']}\n"
        f"beta0={example_case_E2['beta0']:.3f}, kappa={example_case_E2['kappa']:.3f}, tau={example_case_E2['tau']}"
    )

    axes[1].plot(time_grid, example_case_E2["obs"][:, 0], lw=2)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Infected fraction")
    axes[1].set_title("Observed curve from time-varying beta(t) model")

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_e2_overall_figure(case_df, coverage_df, feature_summary_df, severity_label="mild", out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    x = case_df["true_beta"]
    y = case_df["beta_median"]
    ax.scatter(x, y, alpha=0.7, color="#1f77b4")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("Reference true beta (time-avg beta(t))")
    ax.set_ylabel("Posterior median beta")
    ax.set_title(f"E2 {severity_label}: beta recovery")

    ax = axes[0, 1]
    x = case_df["true_gamma"]
    y = case_df["gamma_median"]
    ax.scatter(x, y, alpha=0.7, color="#ff7f0e")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("True gamma")
    ax.set_ylabel("Posterior median gamma")
    ax.set_title(f"E2 {severity_label}: gamma recovery")

    ax = axes[1, 0]
    param_cov_plot = coverage_df.melt(
        id_vars="quantity",
        value_vars=["empirical_50", "empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.barplot(data=param_cov_plot, x="quantity", y="coverage", hue="interval", ax=ax)
    ax.axhline(0.50, color="gray", linestyle="--", lw=1.0)
    ax.axhline(0.90, color="gray", linestyle=":", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Coverage")
    ax.set_title(f"E2 {severity_label}: parameter coverage")

    ax = axes[1, 1]
    feat_cov_plot = feature_summary_df.melt(
        id_vars="feature",
        value_vars=["empirical_50", "empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.barplot(data=feat_cov_plot, x="feature", y="coverage", hue="interval", ax=ax)
    ax.axhline(0.50, color="gray", linestyle="--", lw=1.0)
    ax.axhline(0.90, color="gray", linestyle=":", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Coverage")
    ax.set_title(f"E2 {severity_label}: feature-level PPC coverage")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_e3_example_case(example_case, out_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    case = example_case["case"]
    time_grid = example_case["time_grid"]
    axes[0].plot(time_grid, example_case["rho_path"], lw=2)
    axes[0].axvline(case["tau"], color="black", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("rho(t)")
    axes[0].set_title(
        "E3 rho(t) schedule | illustrative strong case\n"
        f"rho0={case['rho0']:.2f}, eta={case['eta']:.3f}, tau={case['tau']}"
    )

    axes[1].plot(time_grid, example_case["latent_infected"], lw=2, label="latent infected I(t)")
    axes[1].plot(time_grid, example_case["observed_mean"], lw=2, linestyle="--", label="rho(t) * I(t)")
    axes[1].axvline(case["tau"], color="black", linestyle="--", lw=1.2)
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Latent curve vs distorted observation mean")
    axes[1].legend()

    axes[2].plot(time_grid, example_case["observed_curve"], lw=2, label="observed curve")
    axes[2].plot(time_grid, example_case["observed_mean"], lw=2, linestyle="--", alpha=0.8, label="noiseless mean")
    axes[2].axvline(case["tau"], color="black", linestyle="--", lw=1.2)
    axes[2].set_xlabel("Day")
    axes[2].set_ylabel("Observed infected fraction")
    axes[2].set_title("Observed curve under E3")
    axes[2].legend()

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_e3_controlled_grid(controlled_cases, out_path=None):
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))

    for row_idx, block in enumerate(controlled_cases):
        sev = block["severity"]
        case = block["case"]

        ax = axes[row_idx, 0]
        ax.plot(range(len(block["rho_path"])), block["rho_path"], lw=2)
        ax.axvline(case["tau"], color="black", linestyle="--", lw=1.2)
        ax.set_title(f"{sev}: rho(t)")
        ax.set_xlabel("Day")
        ax.set_ylabel("rho(t)")

        ax = axes[row_idx, 1]
        ax.plot(range(len(block["latent_infected"])), block["latent_infected"], lw=2, label="latent I(t)")
        ax.plot(range(len(block["observed_mean"])), block["observed_mean"], lw=2, linestyle="--", label="rho(t) * I(t)")
        ax.axvline(case["tau"], color="black", linestyle="--", lw=1.2)
        ax.set_title(f"{sev}: latent vs distorted mean")
        ax.set_xlabel("Day")
        ax.set_ylabel("Fraction")
        if row_idx == 0:
            ax.legend()

        ax = axes[row_idx, 2]
        ax.plot(range(len(block["observed_curve"])), block["observed_curve"], lw=2, label="observed")
        ax.plot(range(len(block["observed_mean"])), block["observed_mean"], lw=2, linestyle="--", alpha=0.8, label="noiseless mean")
        ax.axvline(case["tau"], color="black", linestyle="--", lw=1.2)
        ax.set_title(f"{sev}: obs curve | eta={case['eta']:.3f}, tau={case['tau']}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Observed infected fraction")
        if row_idx == 0:
            ax.legend()

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_e3_overall_figure(case_df, coverage_df, feature_summary_df, severity_label, out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    x = case_df["true_beta"]
    y = case_df["beta_median"]
    ax.scatter(x, y, alpha=0.7, color="#1f77b4")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("True beta")
    ax.set_ylabel("Posterior median beta")
    ax.set_title(f"E3 {severity_label}: beta recovery")

    ax = axes[0, 1]
    x = case_df["true_gamma"]
    y = case_df["gamma_median"]
    ax.scatter(x, y, alpha=0.7, color="#ff7f0e")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("True gamma")
    ax.set_ylabel("Posterior median gamma")
    ax.set_title(f"E3 {severity_label}: gamma recovery")

    ax = axes[1, 0]
    param_cov_plot = coverage_df.melt(
        id_vars="quantity",
        value_vars=["empirical_50", "empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.barplot(data=param_cov_plot, x="quantity", y="coverage", hue="interval", ax=ax)
    ax.axhline(0.50, color="gray", linestyle="--", lw=1.0)
    ax.axhline(0.90, color="gray", linestyle=":", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Coverage")
    ax.set_title(f"E3 {severity_label}: parameter coverage")

    ax = axes[1, 1]
    feat_cov_plot = feature_summary_df.melt(
        id_vars="feature",
        value_vars=["empirical_50", "empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.barplot(data=feat_cov_plot, x="feature", y="coverage", hue="interval", ax=ax)
    ax.axhline(0.50, color="gray", linestyle="--", lw=1.0)
    ax.axhline(0.90, color="gray", linestyle=":", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Coverage")
    ax.set_title(f"E3 {severity_label}: feature-level PPC coverage")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_truth_vs_median_scatter(case_df, quantities=("beta", "gamma", "R0")):
    fig, axes = plt.subplots(1, len(quantities), figsize=(16, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    if len(quantities) == 1:
        axes = [axes]
    for ax, q, color in zip(axes, quantities, colors):
        x = case_df[f"true_{q}"]
        y = case_df[f"{q}_median"]
        ax.scatter(x, y, alpha=0.7, color=color)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
        ax.set_xlabel(f"True {q}")
        ax.set_ylabel(f"Posterior median {q}")
        ax.set_title(f"Truth vs posterior median: {q}")
    plt.tight_layout()
    plt.show()
    return fig


def plot_feature_residuals(feature_case_df, feature_order=None, title_prefix=None):
    plot_df = feature_case_df.copy()
    plot_df["residual"] = plot_df["err_median"]
    if feature_order is None:
        feature_order = [
            "peak_time",
            "peak_height",
            "cumulative_infected",
            "early_growth_slope",
        ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, feat in zip(axes, feature_order):
        sub = plot_df[plot_df["feature"] == feat].copy()
        sns.boxplot(data=sub, y="residual", ax=ax)
        ax.axhline(0.0, color="black", linestyle="--", lw=1.2)
        if title_prefix is None:
            ax.set_title(f"Residuals: {feat}")
        else:
            ax.set_title(f"{title_prefix}: residuals | {feat}")
        ax.set_xlabel("")
        ax.set_ylabel("Predictive median - observed")
    plt.tight_layout()
    plt.show()
    return fig


def plot_coverage_barplot(summary_df, key="feature", title="Coverage", ylim=(0, 1.05)):
    cov_plot_df = summary_df.melt(
        id_vars=key,
        value_vars=["empirical_50", "empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=cov_plot_df, x=key, y="coverage", hue="interval")
    plt.axhline(0.50, color="gray", linestyle="--", lw=1.0)
    plt.axhline(0.90, color="gray", linestyle=":", lw=1.0)
    plt.title(title)
    plt.xlabel(key.capitalize())
    plt.ylabel("Coverage")
    plt.ylim(*ylim)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_e0_overall_figure(case_df, coverage_df, feature_summary_df, out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    x = case_df["true_beta"]
    y = case_df["beta_median"]
    ax.scatter(x, y, alpha=0.7, color="#1f77b4")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("True beta")
    ax.set_ylabel("Posterior median beta")
    ax.set_title("E0 matched: beta recovery")

    ax = axes[0, 1]
    x = case_df["true_gamma"]
    y = case_df["gamma_median"]
    ax.scatter(x, y, alpha=0.7, color="#ff7f0e")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("True gamma")
    ax.set_ylabel("Posterior median gamma")
    ax.set_title("E0 matched: gamma recovery")

    ax = axes[1, 0]
    param_cov_plot = coverage_df.melt(
        id_vars="quantity",
        value_vars=["empirical_50", "empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.barplot(data=param_cov_plot, x="quantity", y="coverage", hue="interval", ax=ax)
    ax.axhline(0.50, color="gray", linestyle="--", lw=1.0)
    ax.axhline(0.90, color="gray", linestyle=":", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Coverage")
    ax.set_title("E0 matched: parameter coverage")

    ax = axes[1, 1]
    feat_cov_plot = feature_summary_df.melt(
        id_vars="feature",
        value_vars=["empirical_50", "empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.barplot(data=feat_cov_plot, x="feature", y="coverage", hue="interval", ax=ax)
    ax.axhline(0.50, color="gray", linestyle="--", lw=1.0)
    ax.axhline(0.90, color="gray", linestyle=":", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Coverage")
    ax.set_title("E0 matched: feature-level PPC coverage")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_parameter_coverage_with_ci(coverage_unc_df, out_path=None):
    plot_rows = []
    for _, row in coverage_unc_df.iterrows():
        plot_rows.append({
            "quantity": row["quantity"],
            "interval": "50%",
            "nominal": row["nominal_50"],
            "empirical": row["empirical_50"],
            "ci_low": row["ci95_50_low"],
            "ci_high": row["ci95_50_high"],
        })
        plot_rows.append({
            "quantity": row["quantity"],
            "interval": "90%",
            "nominal": row["nominal_90"],
            "empirical": row["empirical_90"],
            "ci_low": row["ci95_90_low"],
            "ci_high": row["ci95_90_high"],
        })

    plot_df = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = []
    x_pos = []
    y_vals = []
    yerr_low = []
    yerr_high = []
    nominal_vals = []
    colors = []

    idx = 0
    for q in plot_df["quantity"].unique():
        sub = plot_df[plot_df["quantity"] == q].copy()
        for _, r in sub.iterrows():
            x_labels.append(f"{q}\n{r['interval']}")
            x_pos.append(idx)
            y_vals.append(r["empirical"])
            emp = float(r["empirical"])
            ci_low = min(float(r["ci_low"]), emp)
            ci_high = max(float(r["ci_high"]), emp)
            yerr_low.append(max(emp - ci_low, 0.0))
            yerr_high.append(max(ci_high - emp, 0.0))
            nominal_vals.append(r["nominal"])
            colors.append("#4c72b0" if r["interval"] == "50%" else "#dd8452")
            idx += 1

    ax.bar(x_pos, y_vals, color=colors, alpha=0.85)
    ax.errorbar(x_pos, y_vals, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", elinewidth=1.5, capsize=4)
    for x, nom in zip(x_pos, nominal_vals):
        ax.hlines(y=nom, xmin=x - 0.35, xmax=x + 0.35, colors="gray", linestyles="--", linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Coverage")
    ax.set_title("E0 matched: empirical parameter coverage with Wilson 95% CI")
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_gamma_stratified_diagnostics(strat_df, title_prefix="Gamma stratified diagnostics", out_path=None):
    plot_df = strat_df.copy().reset_index(drop=True)
    x = np.arange(len(plot_df))
    labels = plot_df["bin"].astype(str).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].bar(x, plot_df["gamma_MAE_median_est"], color="#4c72b0", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Gamma MAE")
    axes[0].set_title(f"{title_prefix}: MAE")

    axes[1].bar(x, plot_df["gamma_bias_median_est"], color="#dd8452", alpha=0.85)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Gamma bias")
    axes[1].set_title(f"{title_prefix}: bias")

    y = plot_df["gamma_empirical_90"].to_numpy(dtype=float)
    ci_low = np.minimum(plot_df["gamma_ci95_90_low"].to_numpy(dtype=float), y)
    ci_high = np.maximum(plot_df["gamma_ci95_90_high"].to_numpy(dtype=float), y)
    yerr_low = np.clip(y - ci_low, 0.0, None)
    yerr_high = np.clip(ci_high - y, 0.0, None)

    axes[2].bar(x, y, color="#55a868", alpha=0.85)
    axes[2].errorbar(x, y, yerr=np.vstack([yerr_low, yerr_high]), fmt="none", ecolor="black", elinewidth=1.5, capsize=4)
    axes[2].axhline(0.90, color="gray", linestyle="--", linewidth=1.2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].set_ylabel("Gamma 90% coverage")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title(f"{title_prefix}: 90% coverage")

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E2_severity_lines(summary_df, out_dir=None):
    if out_dir is None:
        out_dir = get_e2_severity_summary_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    x_labels = summary_df["severity"].astype(str).tolist()
    x = list(range(len(x_labels)))
    has_ci = {"beta_empirical_90_ci_low", "beta_empirical_90_ci_high", "R0_empirical_90_ci_low", "R0_empirical_90_ci_high", "beta_bias_ci_low", "beta_bias_ci_high"}.issubset(summary_df.columns)

    def _line_plot(y_col, title, ylab, out_name, ref_line=None, ci_low=None, ci_high=None, ylim=None):
        y = summary_df[y_col].to_numpy(dtype=float)
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o", lw=2)
        if has_ci and ci_low is not None and ci_high is not None:
            y_low = summary_df[ci_low].to_numpy(dtype=float)
            y_high = summary_df[ci_high].to_numpy(dtype=float)
            yerr = np.vstack([y - y_low, y_high - y])
            plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4)
        if ref_line is not None:
            plt.axhline(ref_line, color="gray", linestyle="--", lw=1.2)
        plt.xticks(x, x_labels)
        plt.xlabel("Severity")
        plt.ylabel(ylab)
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{out_name}", dpi=200, bbox_inches="tight")
        plt.show()

    _line_plot(
        "beta_empirical_90",
        "E2: beta empirical 90% coverage vs severity",
        "Empirical 90% coverage",
        "E2_beta_empirical90_vs_severity.png",
        ref_line=0.90,
        ci_low="beta_empirical_90_ci_low",
        ci_high="beta_empirical_90_ci_high",
        ylim=(0, 1.02),
    )
    _line_plot(
        "R0_empirical_90",
        "E2: R0 empirical 90% coverage vs severity",
        "Empirical 90% coverage",
        "E2_R0_empirical90_vs_severity.png",
        ref_line=0.90,
        ci_low="R0_empirical_90_ci_low",
        ci_high="R0_empirical_90_ci_high",
        ylim=(0, 1.02),
    )
    _line_plot(
        "beta_bias_median",
        "E2: beta bias (median estimator) vs severity",
        "Bias of posterior median beta",
        "E2_beta_bias_median_vs_severity.png",
        ref_line=0.0,
        ci_low="beta_bias_ci_low",
        ci_high="beta_bias_ci_high",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    panels = [
        ("beta_empirical_90", "beta empirical 90%", 0.90, "Empirical 90% coverage", "beta_empirical_90_ci_low", "beta_empirical_90_ci_high", (0, 1.02)),
        ("R0_empirical_90", "R0 empirical 90%", 0.90, "Empirical 90% coverage", "R0_empirical_90_ci_low", "R0_empirical_90_ci_high", (0, 1.02)),
        ("beta_bias_median", "beta bias (median)", 0.0, "Bias of posterior median beta", "beta_bias_ci_low", "beta_bias_ci_high", None),
    ]
    for ax, (col, title, ref_line, ylab, ci_low, ci_high, ylim) in zip(axes, panels):
        y = summary_df[col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=2)
        if has_ci:
            y_low = summary_df[ci_low].to_numpy(dtype=float)
            y_high = summary_df[ci_high].to_numpy(dtype=float)
            yerr = np.vstack([y - y_low, y_high - y])
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4)
        ax.axhline(ref_line, color="gray", linestyle="--", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Severity")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/E2_severity_3panel_summary.png", dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E2_severity_single_line(summary_df, metric, out_path=None):
    configs = {
        "beta_empirical_90": {
            "title": "E2: beta empirical 90% coverage vs severity",
            "ylabel": "Empirical 90% coverage",
            "ref": 0.90,
            "ylim": (0, 1.02),
            "ci_low": "beta_empirical_90_ci_low",
            "ci_high": "beta_empirical_90_ci_high",
        },
        "R0_empirical_90": {
            "title": "E2: R0 empirical 90% coverage vs severity",
            "ylabel": "Empirical 90% coverage",
            "ref": 0.90,
            "ylim": (0, 1.02),
            "ci_low": "R0_empirical_90_ci_low",
            "ci_high": "R0_empirical_90_ci_high",
        },
        "beta_bias_median": {
            "title": "E2: beta bias (median estimator) vs severity",
            "ylabel": "Bias of posterior median beta",
            "ref": 0.0,
            "ylim": None,
            "ci_low": "beta_bias_ci_low",
            "ci_high": "beta_bias_ci_high",
        },
    }
    cfg = configs[metric]
    x_labels = summary_df["severity"].astype(str).tolist()
    x = np.arange(len(x_labels))
    y = summary_df[metric].to_numpy(dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", lw=2)
    if cfg["ci_low"] in summary_df.columns and cfg["ci_high"] in summary_df.columns:
        y_low = summary_df[cfg["ci_low"]].to_numpy(dtype=float)
        y_high = summary_df[cfg["ci_high"]].to_numpy(dtype=float)
        yerr = np.vstack([y - y_low, y_high - y])
        plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4)
    plt.axhline(cfg["ref"], color="gray", linestyle="--", lw=1.2)
    plt.xticks(x, x_labels)
    plt.xlabel("Severity")
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title"])
    if cfg["ylim"] is not None:
        plt.ylim(*cfg["ylim"])
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_E2_severity_3panel_summary(summary_df, out_path=None):
    x_labels = summary_df["severity"].astype(str).tolist()
    x = np.arange(len(x_labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    panels = [
        ("beta_empirical_90", "beta empirical 90%", 0.90, "Empirical 90% coverage", "beta_empirical_90_ci_low", "beta_empirical_90_ci_high", (0, 1.02)),
        ("R0_empirical_90", "R0 empirical 90%", 0.90, "Empirical 90% coverage", "R0_empirical_90_ci_low", "R0_empirical_90_ci_high", (0, 1.02)),
        ("beta_bias_median", "beta bias (median)", 0.0, "Bias of posterior median beta", "beta_bias_ci_low", "beta_bias_ci_high", None),
    ]
    has_ci = {"beta_empirical_90_ci_low", "beta_empirical_90_ci_high", "R0_empirical_90_ci_low", "R0_empirical_90_ci_high", "beta_bias_ci_low", "beta_bias_ci_high"}.issubset(summary_df.columns)
    for ax, (col, title, ref_line, ylab, ci_low, ci_high, ylim) in zip(axes, panels):
        y = summary_df[col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=2)
        if has_ci:
            y_low = summary_df[ci_low].to_numpy(dtype=float)
            y_high = summary_df[ci_high].to_numpy(dtype=float)
            yerr = np.vstack([y - y_low, y_high - y])
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4)
        ax.axhline(ref_line, color="gray", linestyle="--", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Severity")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E2_reference_sensitivity(ref_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.lineplot(data=ref_df, x="severity", y="beta_empirical_90", hue="beta_ref_mode", marker="o", ax=axes[0])
    axes[0].axhline(0.90, color="gray", linestyle="--", lw=1.2)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("beta empirical 90% coverage")
    axes[0].set_ylabel("Coverage")
    axes[0].set_xlabel("Severity")

    sns.lineplot(data=ref_df, x="severity", y="beta_bias_median", hue="beta_ref_mode", marker="o", ax=axes[1])
    axes[1].axhline(0.0, color="gray", linestyle="--", lw=1.2)
    axes[1].set_title("beta median bias")
    axes[1].set_ylabel("Bias")
    axes[1].set_xlabel("Severity")

    sns.lineplot(data=ref_df, x="severity", y="R0_empirical_90", hue="beta_ref_mode", marker="o", ax=axes[2])
    axes[2].axhline(0.90, color="gray", linestyle="--", lw=1.2)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("R0 empirical 90% coverage")
    axes[2].set_ylabel("Coverage")
    axes[2].set_xlabel("Severity")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/E2_reference_sensitivity_3panel.png", dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E3_severity_lines(summary_df, out_dir=None):
    if out_dir is None:
        out_dir = get_e3_severity_summary_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    panels = [
        ("beta_empirical_90", 0.90, "Empirical 90% coverage", "E3: beta empirical 90% coverage vs severity", "E3_beta_empirical90_vs_severity.png"),
        ("gamma_empirical_90", 0.90, "Empirical 90% coverage", "E3: gamma empirical 90% coverage vs severity", "E3_gamma_empirical90_vs_severity.png"),
        ("R0_empirical_90", 0.90, "Empirical 90% coverage", "E3: R0 empirical 90% coverage vs severity", "E3_R0_empirical90_vs_severity.png"),
        ("gamma_bias_median", 0.0, "Bias of posterior median gamma", "E3: gamma bias (median estimator) vs severity", "E3_gamma_bias_median_vs_severity.png"),
    ]

    for col, ref_line, ylab, title, out_name in panels:
        plt.figure(figsize=(6, 4))
        plt.plot(summary_df["severity"].astype(str), summary_df[col], marker="o", lw=2)
        plt.axhline(ref_line, color="gray", linestyle="--", lw=1.2)
        plt.xlabel("Severity")
        plt.ylabel(ylab)
        plt.title(title)
        if "coverage" in ylab.lower():
            plt.ylim(0, 1.02)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{out_name}", dpi=200, bbox_inches="tight")
        plt.show()


def plot_E3_severity_single_line(summary_df, metric, out_path=None):
    configs = {
        "beta_empirical_90": {
            "title": "E3: beta empirical 90% coverage vs severity",
            "ylabel": "Empirical 90% coverage",
            "ref": 0.90,
            "ylim": (0, 1.02),
        },
        "gamma_empirical_90": {
            "title": "E3: gamma empirical 90% coverage vs severity",
            "ylabel": "Empirical 90% coverage",
            "ref": 0.90,
            "ylim": (0, 1.02),
        },
        "R0_empirical_90": {
            "title": "E3: R0 empirical 90% coverage vs severity",
            "ylabel": "Empirical 90% coverage",
            "ref": 0.90,
            "ylim": (0, 1.02),
        },
        "gamma_bias_median": {
            "title": "E3: gamma bias (median estimator) vs severity",
            "ylabel": "Bias of posterior median gamma",
            "ref": 0.0,
            "ylim": None,
        },
    }
    cfg = configs[metric]
    plt.figure(figsize=(6, 4))
    plt.plot(summary_df["severity"].astype(str), summary_df[metric], marker="o", lw=2)
    plt.axhline(cfg["ref"], color="gray", linestyle="--", lw=1.2)
    plt.xlabel("Severity")
    plt.ylabel(cfg["ylabel"])
    plt.title(cfg["title"])
    if cfg["ylim"] is not None:
        plt.ylim(*cfg["ylim"])
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_curve_level_ppc_summary(curve_case_df, severity_label="medium", experiment_label="E2", out_dir=None, split_mode="fraction"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    title_prefix = f"{experiment_label} {severity_label}"
    if split_mode == "tau":
        first_label = "pre-change"
        second_label = "post-change"
        diff_label = "post minus pre RMSE"
    else:
        first_label = "first-window"
        second_label = "second-window"
        diff_label = "second minus first RMSE"

    sns.boxplot(data=curve_case_df, y="curve_rmse_full", ax=axes[0])
    axes[0].set_title(f"{title_prefix}: curve RMSE (full)")
    axes[0].set_ylabel("RMSE")

    rmse_long = curve_case_df.melt(
        value_vars=["curve_rmse_first", "curve_rmse_second"],
        var_name="window",
        value_name="rmse",
    )
    rmse_long["window"] = rmse_long["window"].map({
        "curve_rmse_first": first_label,
        "curve_rmse_second": second_label,
    })
    sns.boxplot(data=rmse_long, x="window", y="rmse", ax=axes[1])
    axes[1].set_title(f"{title_prefix}: {first_label} vs {second_label} RMSE")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMSE")

    sns.boxplot(data=curve_case_df, y="second_minus_first_rmse", ax=axes[2])
    axes[2].axhline(0.0, color="black", linestyle="--", lw=1.2)
    axes[2].set_title(f"{title_prefix}: {diff_label}")
    axes[2].set_ylabel(diff_label)

    cov_long = curve_case_df.melt(
        value_vars=["pointwise_empirical_50", "pointwise_empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.boxplot(data=cov_long, x="interval", y="coverage", ax=axes[3])
    axes[3].axhline(0.50, color="gray", linestyle="--", lw=1.0)
    axes[3].axhline(0.90, color="gray", linestyle=":", lw=1.0)
    axes[3].set_title(f"{title_prefix}: pointwise curve coverage")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("Coverage")

    plt.tight_layout()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{out_dir}/{experiment_label}_curve_level_ppc_{severity_label}.png", dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def scatter_with_group_colors(df, x, y, title, xlabel=None, ylabel=None, out_path=None):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="severity",
        hue_order=["mild", "medium", "strong"],
        alpha=0.75,
    )
    plt.title(title)
    plt.xlabel(xlabel if xlabel is not None else x)
    plt.ylabel(ylabel if ylabel is not None else y)
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_E3_eta_bin_panels(eta_bin_summary, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(eta_bin_summary["eta_mean"], eta_bin_summary["R0_err_median_mean"], marker="o", lw=2)
    axes[0].axhline(0.0, color="gray", linestyle="--", lw=1.0)
    axes[0].set_xlabel("Mean eta in bin")
    axes[0].set_ylabel("Mean R0 median error")
    axes[0].set_title("E3: eta bin vs R0 median error")

    axes[1].plot(eta_bin_summary["eta_mean"], eta_bin_summary["R0_covered90_rate"], marker="o", lw=2)
    axes[1].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[1].set_xlabel("Mean eta in bin")
    axes[1].set_ylabel("R0 90% coverage rate")
    axes[1].set_title("E3: eta bin vs R0 90% coverage")

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E3_tau_bin_panels(tau_bin_summary, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(tau_bin_summary["tau_mean"], tau_bin_summary["R0_err_median_mean"], marker="o", lw=2)
    axes[0].axhline(0.0, color="gray", linestyle="--", lw=1.0)
    axes[0].set_xlabel("Mean tau in bin")
    axes[0].set_ylabel("Mean R0 median error")
    axes[0].set_title("E3: tau bin vs R0 median error")

    axes[1].plot(tau_bin_summary["tau_mean"], tau_bin_summary["R0_covered90_rate"], marker="o", lw=2)
    axes[1].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[1].set_xlabel("Mean tau in bin")
    axes[1].set_ylabel("R0 90% coverage rate")
    axes[1].set_title("E3: tau bin vs R0 90% coverage")

    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E3_truth_group_panels(gamma_group_summary, r0_group_summary, beta_group_summary, out_dir=None):
    figs = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(gamma_group_summary["true_gamma_group"].astype(str), gamma_group_summary["gamma_covered90_rate"])
    axes[0].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[0].set_title("E3: gamma coverage by true_gamma group")
    axes[0].set_xlabel("true_gamma group")
    axes[0].set_ylabel("gamma 90% coverage rate")

    axes[1].bar(gamma_group_summary["true_gamma_group"].astype(str), gamma_group_summary["R0_covered90_rate"])
    axes[1].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[1].set_title("E3: R0 coverage by true_gamma group")
    axes[1].set_xlabel("true_gamma group")
    axes[1].set_ylabel("R0 90% coverage rate")

    axes[2].bar(gamma_group_summary["true_gamma_group"].astype(str), gamma_group_summary["gamma_err_median_mean"])
    axes[2].axhline(0.0, color="gray", linestyle="--", lw=1.0)
    axes[2].set_title("E3: gamma median error by true_gamma group")
    axes[2].set_xlabel("true_gamma group")
    axes[2].set_ylabel("mean gamma median error")
    plt.tight_layout()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{out_dir}/E3_true_gamma_group_panels.png", dpi=200, bbox_inches="tight")
    plt.show()
    figs.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(r0_group_summary["true_R0_group"].astype(str), r0_group_summary["R0_covered90_rate"])
    axes[0].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[0].set_title("E3: R0 coverage by true_R0 group")
    axes[0].set_xlabel("true_R0 group")
    axes[0].set_ylabel("R0 90% coverage rate")

    axes[1].bar(r0_group_summary["true_R0_group"].astype(str), r0_group_summary["R0_err_median_mean"])
    axes[1].axhline(0.0, color="gray", linestyle="--", lw=1.0)
    axes[1].set_title("E3: R0 median error by true_R0 group")
    axes[1].set_xlabel("true_R0 group")
    axes[1].set_ylabel("mean R0 median error")
    plt.tight_layout()
    if out_dir is not None:
        fig.savefig(f"{out_dir}/E3_true_R0_group_panels.png", dpi=200, bbox_inches="tight")
    plt.show()
    figs.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(beta_group_summary["true_beta_group"].astype(str), beta_group_summary["beta_covered90_rate"])
    axes[0].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[0].set_title("E3: beta coverage by true_beta group")
    axes[0].set_xlabel("true_beta group")
    axes[0].set_ylabel("beta 90% coverage rate")

    axes[1].bar(beta_group_summary["true_beta_group"].astype(str), beta_group_summary["R0_covered90_rate"])
    axes[1].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[1].set_title("E3: R0 coverage by true_beta group")
    axes[1].set_xlabel("true_beta group")
    axes[1].set_ylabel("R0 90% coverage rate")
    plt.tight_layout()
    if out_dir is not None:
        fig.savefig(f"{out_dir}/E3_true_beta_group_panels.png", dpi=200, bbox_inches="tight")
    plt.show()
    figs.append(fig)

    return figs


def plot_E3_true_gamma_group_panels(gamma_group_summary, out_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(gamma_group_summary["true_gamma_group"].astype(str), gamma_group_summary["gamma_covered90_rate"])
    axes[0].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[0].set_title("E3: gamma coverage by true_gamma group")
    axes[0].set_xlabel("true_gamma group")
    axes[0].set_ylabel("gamma 90% coverage rate")

    axes[1].bar(gamma_group_summary["true_gamma_group"].astype(str), gamma_group_summary["R0_covered90_rate"])
    axes[1].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[1].set_title("E3: R0 coverage by true_gamma group")
    axes[1].set_xlabel("true_gamma group")
    axes[1].set_ylabel("R0 90% coverage rate")

    axes[2].bar(gamma_group_summary["true_gamma_group"].astype(str), gamma_group_summary["gamma_err_median_mean"])
    axes[2].axhline(0.0, color="gray", linestyle="--", lw=1.0)
    axes[2].set_title("E3: gamma median error by true_gamma group")
    axes[2].set_xlabel("true_gamma group")
    axes[2].set_ylabel("mean gamma median error")
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E3_true_R0_group_panels(r0_group_summary, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(r0_group_summary["true_R0_group"].astype(str), r0_group_summary["R0_covered90_rate"])
    axes[0].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[0].set_title("E3: R0 coverage by true_R0 group")
    axes[0].set_xlabel("true_R0 group")
    axes[0].set_ylabel("R0 90% coverage rate")

    axes[1].bar(r0_group_summary["true_R0_group"].astype(str), r0_group_summary["R0_err_median_mean"])
    axes[1].axhline(0.0, color="gray", linestyle="--", lw=1.0)
    axes[1].set_title("E3: R0 median error by true_R0 group")
    axes[1].set_xlabel("true_R0 group")
    axes[1].set_ylabel("mean R0 median error")
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_E3_true_beta_group_panels(beta_group_summary, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(beta_group_summary["true_beta_group"].astype(str), beta_group_summary["beta_covered90_rate"])
    axes[0].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[0].set_title("E3: beta coverage by true_beta group")
    axes[0].set_xlabel("true_beta group")
    axes[0].set_ylabel("beta 90% coverage rate")

    axes[1].bar(beta_group_summary["true_beta_group"].astype(str), beta_group_summary["R0_covered90_rate"])
    axes[1].axhline(0.90, color="gray", linestyle="--", lw=1.0)
    axes[1].set_title("E3: R0 coverage by true_beta group")
    axes[1].set_xlabel("true_beta group")
    axes[1].set_ylabel("R0 90% coverage rate")
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig
