import os
import warnings
from importlib import metadata

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import seaborn as sns

EXPECTED_VERSIONS = {
    "bayesflow": "2.0.7",
    "tensorflow": "2.17.0",
    "keras": "3.9.0",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "matplotlib": "3.8.4",
}

N_DAYS = 50
TIME_GRID = np.arange(N_DAYS)
I0 = 0.01
R0_INIT = 0.0
S0 = 1.0 - I0 - R0_INIT
OBS_NOISE = 0.02

PRIOR_BOUNDS = {
    "beta": (0.10, 1.00),
    "gamma": (0.05, 0.50),
}
E1_SIGMA_BOUNDS = (0.10, 0.80)
E1_SEVERITY_CONFIG = {
    "mild": {"sigma_low": 0.35, "sigma_high": 0.60},
    "medium": {"sigma_low": 0.20, "sigma_high": 0.35},
    "strong": {"sigma_low": 0.08, "sigma_high": 0.20},
}

EPS = 1e-6

E2_CHANGE_DAY_RANGE = (10, 30)
E2_SEVERITY_CONFIG = {
    "mild": {"kappa_low": 0.85, "kappa_high": 1.00},
    "medium": {"kappa_low": 0.60, "kappa_high": 0.85},
    "strong": {"kappa_low": 0.30, "kappa_high": 0.60},
}
E3_CHANGE_DAY_RANGE = (10, 30)
E3_SEVERITY_CONFIG = {
    "mild": {"eta_low": 0.85, "eta_high": 1.00},
    "medium": {"eta_low": 0.60, "eta_high": 0.85},
    "strong": {"eta_low": 0.30, "eta_high": 0.60},
}

OUTPUTS_ROOT = "outputs"
E0_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "E0")
E1_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "E1")
E2_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "E2")
E3_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "E3")


def get_e1_curve_level_output_dir():
    return os.path.join(E1_OUTPUT_DIR, "curve_level")


def get_e1_severity_output_dir(severity):
    return os.path.join(E1_OUTPUT_DIR, severity)


def get_e1_curve_level_output_dir_for_severity(severity):
    return os.path.join(E1_OUTPUT_DIR, "curve_level", severity)


def get_e1_severity_summary_output_dir():
    return os.path.join(E1_OUTPUT_DIR, "severity_summary")


def get_e1_reference_ablation_output_dir(reference_mode=None):
    base_dir = os.path.join(E1_OUTPUT_DIR, "ablation_reference")
    if reference_mode is None:
        return base_dir
    return os.path.join(base_dir, reference_mode)


def get_e1_posterior_family_ablation_output_dir(posterior_family=None):
    base_dir = os.path.join(E1_OUTPUT_DIR, "ablation_posterior_family")
    if posterior_family is None:
        return base_dir
    return os.path.join(base_dir, posterior_family)


def get_e2_severity_output_dir(severity):
    return os.path.join(E2_OUTPUT_DIR, severity)


def get_e2_curve_level_output_dir(severity):
    return os.path.join(E2_OUTPUT_DIR, "curve_level", severity)


def get_e2_curve_level_tau_output_dir(severity):
    return os.path.join(E2_OUTPUT_DIR, "curve_level_tau", severity)


def get_e2_severity_summary_output_dir():
    return os.path.join(E2_OUTPUT_DIR, "severity_summary")


def get_e2_reference_sensitivity_output_dir():
    return os.path.join(E2_OUTPUT_DIR, "reference_sensitivity")


def get_e3_severity_output_dir(severity):
    return os.path.join(E3_OUTPUT_DIR, severity)


def get_e3_curve_level_output_dir(severity):
    return os.path.join(E3_OUTPUT_DIR, "curve_level", severity)


def get_e3_curve_level_tau_output_dir(severity):
    return os.path.join(E3_OUTPUT_DIR, "curve_level_tau", severity)


def get_e3_severity_summary_output_dir():
    return os.path.join(E3_OUTPUT_DIR, "severity_summary")


def get_e3_caselevel_analysis_output_dir():
    return os.path.join(E3_OUTPUT_DIR, "caselevel_analysis")


def get_e3_truth_region_output_dir():
    return os.path.join(E3_OUTPUT_DIR, "truth_region_analysis")


def setup_environment(seed=42, check_versions=True, set_theme=True):
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import keras

    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    if set_theme:
        sns.set_theme(style="whitegrid", context="notebook")

    if check_versions:
        for pkg, expected in EXPECTED_VERSIONS.items():
            current = metadata.version(pkg)
            print(f"{pkg}: {current}")
            if current != expected:
                print(f"  warning: expected {expected}, but found {current}")
