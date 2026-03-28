import os

import matplotlib.pyplot as plt
import seaborn as sns

from .config import get_e2_severity_summary_output_dir


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


def plot_feature_residuals(feature_case_df, feature_order=None):
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
        ax.set_title(f"Residuals: {feat}")
        ax.set_xlabel("")
        ax.set_ylabel("Predictive median - observed")
    plt.tight_layout()
    plt.show()


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


def plot_E2_severity_lines(summary_df, out_dir=None):
    if out_dir is None:
        out_dir = get_e2_severity_summary_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(summary_df["severity"].astype(str), summary_df["beta_empirical_90"], marker="o", lw=2)
    plt.axhline(0.90, color="gray", linestyle="--", lw=1.2)
    plt.xlabel("Severity")
    plt.ylabel("Empirical 90% coverage")
    plt.title("E2: beta empirical 90% coverage vs severity")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/E2_beta_empirical90_vs_severity.png", dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(summary_df["severity"].astype(str), summary_df["R0_empirical_90"], marker="o", lw=2)
    plt.axhline(0.90, color="gray", linestyle="--", lw=1.2)
    plt.xlabel("Severity")
    plt.ylabel("Empirical 90% coverage")
    plt.title("E2: R0 empirical 90% coverage vs severity")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/E2_R0_empirical90_vs_severity.png", dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(summary_df["severity"].astype(str), summary_df["beta_bias_median"], marker="o", lw=2)
    plt.axhline(0.0, color="gray", linestyle="--", lw=1.2)
    plt.xlabel("Severity")
    plt.ylabel("Bias of posterior median beta")
    plt.title("E2: beta bias (median estimator) vs severity")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/E2_beta_bias_median_vs_severity.png", dpi=200, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(summary_df["severity"].astype(str), summary_df["beta_empirical_90"], marker="o", lw=2)
    axes[0].axhline(0.90, color="gray", linestyle="--", lw=1.2)
    axes[0].set_xlabel("Severity")
    axes[0].set_ylabel("Empirical 90% coverage")
    axes[0].set_title("beta empirical 90%")

    axes[1].plot(summary_df["severity"].astype(str), summary_df["R0_empirical_90"], marker="o", lw=2)
    axes[1].axhline(0.90, color="gray", linestyle="--", lw=1.2)
    axes[1].set_xlabel("Severity")
    axes[1].set_ylabel("Empirical 90% coverage")
    axes[1].set_title("R0 empirical 90%")

    axes[2].plot(summary_df["severity"].astype(str), summary_df["beta_bias_median"], marker="o", lw=2)
    axes[2].axhline(0.0, color="gray", linestyle="--", lw=1.2)
    axes[2].set_xlabel("Severity")
    axes[2].set_ylabel("Bias of posterior median beta")
    axes[2].set_title("beta bias (median)")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/E2_severity_3panel_summary.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_curve_level_ppc_summary(curve_case_df, severity_label="medium", out_dir=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    sns.boxplot(data=curve_case_df, y="curve_rmse_full", ax=axes[0])
    axes[0].set_title(f"E2 {severity_label}: curve RMSE (full)")
    axes[0].set_ylabel("RMSE")

    rmse_long = curve_case_df.melt(
        value_vars=["curve_rmse_early", "curve_rmse_late"],
        var_name="window",
        value_name="rmse",
    )
    sns.boxplot(data=rmse_long, x="window", y="rmse", ax=axes[1])
    axes[1].set_title(f"E2 {severity_label}: early vs late RMSE")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMSE")

    sns.boxplot(data=curve_case_df, y="late_minus_early_rmse", ax=axes[2])
    axes[2].axhline(0.0, color="black", linestyle="--", lw=1.2)
    axes[2].set_title(f"E2 {severity_label}: late minus early RMSE")
    axes[2].set_ylabel("Late RMSE - Early RMSE")

    cov_long = curve_case_df.melt(
        value_vars=["pointwise_empirical_50", "pointwise_empirical_90"],
        var_name="interval",
        value_name="coverage",
    )
    sns.boxplot(data=cov_long, x="interval", y="coverage", ax=axes[3])
    axes[3].axhline(0.50, color="gray", linestyle="--", lw=1.0)
    axes[3].axhline(0.90, color="gray", linestyle=":", lw=1.0)
    axes[3].set_title(f"E2 {severity_label}: pointwise curve coverage")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("Coverage")

    plt.tight_layout()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{out_dir}/E2_curve_level_ppc_{severity_label}.png", dpi=200, bbox_inches="tight")
    plt.show()
