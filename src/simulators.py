import numpy as np
from scipy.integrate import solve_ivp

from .config import (
    E1_SIGMA_BOUNDS,
    E2_CHANGE_DAY_RANGE,
    E2_SEVERITY_CONFIG,
    N_DAYS,
    OBS_NOISE,
    PRIOR_BOUNDS,
    R0_INIT,
    S0,
)


def sir_rhs(t, y, beta, gamma):
    s, i, r = y
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i
    return [ds, di, dr]


def solve_sir(beta, gamma, n_days=N_DAYS):
    sol = solve_ivp(
        lambda t, y: sir_rhs(t, y, beta, gamma),
        (0, n_days - 1),
        [S0, 0.01, R0_INIT],
        t_eval=np.arange(n_days),
        rtol=1e-6,
        atol=1e-8,
    )
    return sol.y.T


def simulate_curve(beta, gamma, rng=None):
    infected = solve_sir(beta, gamma)[:, 1]
    if rng is None:
        noise = np.random.normal(0.0, OBS_NOISE, size=infected.shape)
    else:
        noise = rng.normal(0.0, OBS_NOISE, size=infected.shape)
    obs = infected + noise
    obs = np.clip(obs, 0.0, 1.0)
    return obs[:, None].astype("float32")


def simulate_case_matched(rng=None, beta=None, gamma=None):
    if rng is None:
        rng = np.random.default_rng()
    if beta is None:
        beta = rng.uniform(*PRIOR_BOUNDS["beta"])
    if gamma is None:
        gamma = rng.uniform(*PRIOR_BOUNDS["gamma"])
    obs = simulate_curve(beta, gamma, rng=rng)
    return {
        "true_beta": float(beta),
        "true_gamma": float(gamma),
        "true_R0": float(beta / gamma),
        "obs": obs,
    }


def seir_rhs(t, y, beta, gamma, sigma):
    s, e, i, r = y
    ds = -beta * s * i
    de = beta * s * i - sigma * e
    di = sigma * e - gamma * i
    dr = gamma * i
    return [ds, de, di, dr]


def solve_seir(beta, gamma, sigma, n_days=N_DAYS, e0=0.005, i0=0.01, r0_init=0.0):
    s0 = 1.0 - e0 - i0 - r0_init
    sol = solve_ivp(
        lambda t, y: seir_rhs(t, y, beta, gamma, sigma),
        (0, n_days - 1),
        [s0, e0, i0, r0_init],
        t_eval=np.arange(n_days),
        rtol=1e-6,
        atol=1e-8,
    )
    return sol.y.T


def simulate_curve_seir(beta, gamma, sigma, rng=None):
    infected = solve_seir(beta, gamma, sigma)[:, 2]
    if rng is None:
        noise = np.random.normal(0.0, OBS_NOISE, size=infected.shape)
    else:
        noise = rng.normal(0.0, OBS_NOISE, size=infected.shape)
    obs = infected + noise
    obs = np.clip(obs, 0.0, 1.0)
    return obs[:, None].astype("float32")


def simulate_case_E1(rng=None, beta=None, gamma=None, sigma=None):
    if rng is None:
        rng = np.random.default_rng()
    if beta is None:
        beta = rng.uniform(*PRIOR_BOUNDS["beta"])
    if gamma is None:
        gamma = rng.uniform(*PRIOR_BOUNDS["gamma"])
    if sigma is None:
        sigma = rng.uniform(*E1_SIGMA_BOUNDS)

    obs = simulate_curve_seir(beta, gamma, sigma, rng=rng)
    return {
        "true_beta": float(beta),
        "true_gamma": float(gamma),
        "true_R0": float(beta / gamma),
        "sigma": float(sigma),
        "obs": obs,
    }


def beta_t_piecewise(time_grid, beta0, tau, kappa):
    t = np.asarray(time_grid)
    return np.where(t < tau, beta0, kappa * beta0).astype(float)


def sir_rhs_timevarying(t, y, beta0, gamma, tau, kappa):
    s, i, r = y
    beta_t = beta0 if t < tau else kappa * beta0
    ds = -beta_t * s * i
    di = beta_t * s * i - gamma * i
    dr = gamma * i
    return [ds, di, dr]


def solve_sir_timevarying_beta(beta0, gamma, tau, kappa, n_days=N_DAYS):
    sol = solve_ivp(
        lambda t, y: sir_rhs_timevarying(t, y, beta0, gamma, tau, kappa),
        (0, n_days - 1),
        [S0, 0.01, R0_INIT],
        t_eval=np.arange(n_days),
        rtol=1e-6,
        atol=1e-8,
    )
    return sol.y.T


def simulate_curve_timevarying_beta(beta0, gamma, tau, kappa, rng=None):
    infected = solve_sir_timevarying_beta(beta0, gamma, tau, kappa)[:, 1]
    if rng is None:
        noise = np.random.normal(0.0, OBS_NOISE, size=infected.shape)
    else:
        noise = rng.normal(0.0, OBS_NOISE, size=infected.shape)
    obs = infected + noise
    obs = np.clip(obs, 0.0, 1.0)
    return obs[:, None].astype("float32")


def compute_beta_ref_mean(beta0, tau, kappa, n_days=N_DAYS):
    beta_path = beta_t_piecewise(np.arange(n_days), beta0, tau, kappa)
    return float(beta_path.mean())


def simulate_case_E2(rng=None, severity="strong", beta0=None, gamma=None, tau=None, kappa=None):
    if rng is None:
        rng = np.random.default_rng()
    if severity not in E2_SEVERITY_CONFIG:
        raise ValueError(f"Unknown severity='{severity}'. Choose from {list(E2_SEVERITY_CONFIG.keys())}")

    cfg = E2_SEVERITY_CONFIG[severity]
    if beta0 is None:
        beta0 = rng.uniform(*PRIOR_BOUNDS["beta"])
    if gamma is None:
        gamma = rng.uniform(*PRIOR_BOUNDS["gamma"])
    if tau is None:
        tau = int(rng.integers(E2_CHANGE_DAY_RANGE[0], E2_CHANGE_DAY_RANGE[1] + 1))
    if kappa is None:
        kappa = rng.uniform(cfg["kappa_low"], cfg["kappa_high"])

    obs = simulate_curve_timevarying_beta(beta0, gamma, tau, kappa, rng=rng)
    beta_ref = compute_beta_ref_mean(beta0, tau, kappa, n_days=N_DAYS)

    return {
        "true_beta": float(beta_ref),
        "true_gamma": float(gamma),
        "true_R0": float(beta_ref / gamma),
        "beta0": float(beta0),
        "gamma": float(gamma),
        "tau": int(tau),
        "kappa": float(kappa),
        "severity": severity,
        "obs": obs,
    }
