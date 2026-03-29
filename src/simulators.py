import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .config import (
    E1_SIGMA_BOUNDS,
    E1_SEVERITY_CONFIG,
    E2_CHANGE_DAY_RANGE,
    E2_SEVERITY_CONFIG,
    E3_CHANGE_DAY_RANGE,
    E3_SEVERITY_CONFIG,
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


def _clip_to_prior_bounds(value, key):
    low, high = PRIOR_BOUNDS[key]
    return float(np.clip(value, low, high))


def fit_pseudotrue_sir_reference(
    beta_seir,
    gamma_seir,
    sigma_seir,
    reference_mode="full_curve",
    early_phase_days=12,
    n_days=N_DAYS,
):
    if reference_mode not in {"full_curve", "early_phase"}:
        raise ValueError("reference_mode must be one of {'full_curve', 'early_phase'}.")

    seir_infected = solve_seir(beta_seir, gamma_seir, sigma_seir, n_days=n_days)[:, 2]
    if reference_mode == "full_curve":
        window = slice(None)
    else:
        k = min(max(int(early_phase_days), 2), n_days)
        window = slice(0, k)

    target_curve = seir_infected[window]

    def objective(theta):
        beta_candidate = float(theta[0])
        gamma_candidate = float(theta[1])
        sir_infected = solve_sir(beta_candidate, gamma_candidate, n_days=n_days)[:, 1]
        residual = sir_infected[window] - target_curve
        return float(np.sum(residual ** 2))

    initial_guesses = [
        np.array(
            [
                _clip_to_prior_bounds(beta_seir, "beta"),
                _clip_to_prior_bounds(gamma_seir, "gamma"),
            ],
            dtype=float,
        ),
        np.array(
            [
                _clip_to_prior_bounds(beta_seir * 0.9, "beta"),
                _clip_to_prior_bounds(max(gamma_seir, sigma_seir), "gamma"),
            ],
            dtype=float,
        ),
        np.array(
            [
                np.mean(PRIOR_BOUNDS["beta"]),
                np.mean(PRIOR_BOUNDS["gamma"]),
            ],
            dtype=float,
        ),
    ]
    bounds = [PRIOR_BOUNDS["beta"], PRIOR_BOUNDS["gamma"]]

    best_result = None
    for x0 in initial_guesses:
        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    beta_ref, gamma_ref = best_result.x
    beta_ref = _clip_to_prior_bounds(beta_ref, "beta")
    gamma_ref = _clip_to_prior_bounds(gamma_ref, "gamma")
    return {
        "beta_ref": beta_ref,
        "gamma_ref": gamma_ref,
        "R0_ref": float(beta_ref / gamma_ref),
        "reference_mode": reference_mode,
        "reference_window_days": n_days if reference_mode == "full_curve" else min(max(int(early_phase_days), 2), n_days),
        "projection_loss": float(best_result.fun),
    }


def simulate_case_E1(
    rng=None,
    beta=None,
    gamma=None,
    sigma=None,
    severity=None,
    reference_mode="full_curve",
    early_phase_days=12,
):
    if rng is None:
        rng = np.random.default_rng()
    if severity is not None and severity not in E1_SEVERITY_CONFIG:
        raise ValueError(f"Unknown severity='{severity}'. Choose from {list(E1_SEVERITY_CONFIG.keys())}.")
    if beta is None:
        beta = rng.uniform(*PRIOR_BOUNDS["beta"])
    if gamma is None:
        gamma = rng.uniform(*PRIOR_BOUNDS["gamma"])
    if sigma is None:
        if severity is None:
            sigma = rng.uniform(*E1_SIGMA_BOUNDS)
        else:
            cfg = E1_SEVERITY_CONFIG[severity]
            sigma = rng.uniform(cfg["sigma_low"], cfg["sigma_high"])

    obs = simulate_curve_seir(beta, gamma, sigma, rng=rng)
    pseudo_true = fit_pseudotrue_sir_reference(
        beta_seir=beta,
        gamma_seir=gamma,
        sigma_seir=sigma,
        reference_mode=reference_mode,
        early_phase_days=early_phase_days,
    )
    return {
        "true_beta": float(pseudo_true["beta_ref"]),
        "true_gamma": float(pseudo_true["gamma_ref"]),
        "true_R0": float(pseudo_true["R0_ref"]),
        "beta_ref": float(pseudo_true["beta_ref"]),
        "gamma_ref": float(pseudo_true["gamma_ref"]),
        "R0_ref": float(pseudo_true["R0_ref"]),
        "beta_seir": float(beta),
        "gamma_seir": float(gamma),
        "sigma_seir": float(sigma),
        "sigma": float(sigma),
        "severity": severity if severity is not None else "pooled",
        "reference_mode": pseudo_true["reference_mode"],
        "reference_window_days": int(pseudo_true["reference_window_days"]),
        "projection_loss": float(pseudo_true["projection_loss"]),
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


def compute_beta_ref_infection_weighted(beta0, gamma, tau, kappa, n_days=N_DAYS, eps=1e-8):
    latent = solve_sir_timevarying_beta(beta0, gamma, tau, kappa, n_days=n_days)
    infected = latent[:, 1]
    beta_path = beta_t_piecewise(np.arange(n_days), beta0, tau, kappa)
    weight_sum = float(np.sum(infected))
    if weight_sum <= eps:
        return float(beta_path.mean())
    return float(np.sum(beta_path * infected) / weight_sum)


def simulate_case_E2(
    rng=None,
    severity="strong",
    beta0=None,
    gamma=None,
    tau=None,
    kappa=None,
    beta_ref_mode="time_avg",
):
    if rng is None:
        rng = np.random.default_rng()
    if severity not in E2_SEVERITY_CONFIG:
        raise ValueError(f"Unknown severity='{severity}'. Choose from {list(E2_SEVERITY_CONFIG.keys())}")
    if beta_ref_mode not in {"time_avg", "infection_weighted"}:
        raise ValueError("beta_ref_mode must be one of {'time_avg', 'infection_weighted'}.")

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
    beta_ref_timeavg = compute_beta_ref_mean(beta0, tau, kappa, n_days=N_DAYS)
    beta_ref_iw = compute_beta_ref_infection_weighted(
        beta0=beta0,
        gamma=gamma,
        tau=tau,
        kappa=kappa,
        n_days=N_DAYS,
    )
    beta_ref = beta_ref_timeavg if beta_ref_mode == "time_avg" else beta_ref_iw

    return {
        "true_beta": float(beta_ref),
        "true_gamma": float(gamma),
        "true_R0": float(beta_ref / gamma),
        "true_beta_timeavg": float(beta_ref_timeavg),
        "true_beta_infweighted": float(beta_ref_iw),
        "true_R0_timeavg": float(beta_ref_timeavg / gamma),
        "true_R0_infweighted": float(beta_ref_iw / gamma),
        "beta_ref_mode": beta_ref_mode,
        "beta0": float(beta0),
        "gamma": float(gamma),
        "tau": int(tau),
        "kappa": float(kappa),
        "severity": severity,
        "obs": obs,
    }


def rho_t_piecewise(time_grid, rho0, tau, eta):
    t = np.asarray(time_grid)
    return np.where(t < tau, rho0, eta * rho0).astype(float)


def simulate_curve_timevarying_reporting(beta, gamma, rho0, tau, eta, rng=None):
    infected = solve_sir(beta, gamma)[:, 1]
    rho_path = rho_t_piecewise(np.arange(len(infected)), rho0, tau, eta)
    observed_mean = rho_path * infected
    if rng is None:
        noise = np.random.normal(0.0, OBS_NOISE, size=observed_mean.shape)
    else:
        noise = rng.normal(0.0, OBS_NOISE, size=observed_mean.shape)
    obs = observed_mean + noise
    obs = np.clip(obs, 0.0, 1.0)
    return obs[:, None].astype("float32")


def simulate_case_E3(
    rng=None,
    severity="strong",
    beta=None,
    gamma=None,
    rho0=1.0,
    tau=None,
    eta=None,
):
    if rng is None:
        rng = np.random.default_rng()
    if severity not in E3_SEVERITY_CONFIG:
        raise ValueError(f"Unknown severity='{severity}'. Choose from {list(E3_SEVERITY_CONFIG.keys())}")

    cfg = E3_SEVERITY_CONFIG[severity]
    if beta is None:
        beta = rng.uniform(*PRIOR_BOUNDS["beta"])
    if gamma is None:
        gamma = rng.uniform(*PRIOR_BOUNDS["gamma"])
    if tau is None:
        tau = int(rng.integers(E3_CHANGE_DAY_RANGE[0], E3_CHANGE_DAY_RANGE[1] + 1))
    if eta is None:
        eta = rng.uniform(cfg["eta_low"], cfg["eta_high"])

    obs = simulate_curve_timevarying_reporting(beta, gamma, rho0, tau, eta, rng=rng)
    return {
        "true_beta": float(beta),
        "true_gamma": float(gamma),
        "true_R0": float(beta / gamma),
        "rho0": float(rho0),
        "tau": int(tau),
        "eta": float(eta),
        "severity": severity,
        "obs": obs,
    }
