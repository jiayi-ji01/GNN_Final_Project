import math
import importlib

import numpy as np

from .config import EPS, PRIOR_BOUNDS
from .simulators import simulate_curve

def _require_workflow_backend():
    try:
        import bayesflow as bf
        import keras
    except Exception as e:
        raise RuntimeError(
            "BayesFlow/TensorFlow backend is unavailable. "
            "Install the required packages and restart the kernel/session. "
            + repr(e)
        )
    return bf, keras


def to_unconstrained(x, low, high):
    u = (x - low) / (high - low)
    u = np.clip(u, EPS, 1.0 - EPS)
    return np.log(u / (1.0 - u))


def from_unconstrained(z, low, high):
    s = 1.0 / (1.0 + np.exp(-z))
    return low + (high - low) * s


def make_training_simulator(train_rng):
    def sim_one():
        beta = train_rng.uniform(*PRIOR_BOUNDS["beta"])
        gamma = train_rng.uniform(*PRIOR_BOUNDS["gamma"])

        z_beta = to_unconstrained(beta, *PRIOR_BOUNDS["beta"])
        z_gamma = to_unconstrained(gamma, *PRIOR_BOUNDS["gamma"])

        return {
            "z_beta": np.array([z_beta], dtype="float32"),
            "z_gamma": np.array([z_gamma], dtype="float32"),
            "obs": simulate_curve(beta, gamma, rng=train_rng),
        }

    return sim_one


def _resolve_bayesflow_scoring_class(bf, class_name):
    if hasattr(bf, "networks") and hasattr(bf.networks, "ScoringRuleNetwork"):
        globals_dict = getattr(getattr(bf.networks.ScoringRuleNetwork, "__init__", None), "__globals__", {})
        resolved = globals_dict.get(class_name)
        if resolved is not None:
            return resolved

    candidate_modules = []

    if hasattr(bf, "scoring_rules"):
        candidate_modules.append("bayesflow.scoring_rules")

    if hasattr(bf, "networks") and hasattr(bf.networks, "ScoringRuleNetwork"):
        network_module = getattr(bf.networks.ScoringRuleNetwork, "__module__", "")
        if network_module:
            scoring_module = network_module.rsplit(".", 1)[0] + ".scoring_rules"
            candidate_modules.append(scoring_module)

    candidate_modules.extend(
        [
            "bayesflow.networks.inference.scoring.scoring_rules",
            "bayesflow.networks.scoring.scoring_rules",
        ]
    )

    seen = set()
    for module_name in candidate_modules:
        if module_name in seen:
            continue
        seen.add(module_name)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        resolved = getattr(module, class_name, None)
        if resolved is not None:
            return resolved

    return None


def _build_diag_gaussian_inference_network(bf, keras):
    ParametricDistributionScore = _resolve_bayesflow_scoring_class(bf, "ParametricDistributionScore")
    ScoringRule = _resolve_bayesflow_scoring_class(bf, "ScoringRule")

    if ParametricDistributionScore is not None:
        score_base = ParametricDistributionScore
        use_custom_score = False
    elif ScoringRule is not None:
        score_base = ScoringRule
        use_custom_score = True
    else:
        raise RuntimeError(
            "Could not resolve a BayesFlow scoring-rule base class for "
            "`posterior_family='diag_gaussian'` in this environment."
        )

    class DiagonalNormalScore(score_base):
        def __init__(self, dim=None):
            super().__init__(links={"std": "softplus"})
            self.dim = dim

        def get_config(self):
            return super().get_config() | {"dim": self.dim}

        def get_head_shapes_from_target_shape(self, target_shape):
            self.dim = target_shape[-1]
            return {"mean": (self.dim,), "std": (self.dim,)}

        def log_prob(self, x, mean, std):
            log_two_pi = keras.ops.convert_to_tensor(math.log(2.0 * math.pi), dtype=keras.ops.dtype(mean))
            z = (x - mean) / std
            return -0.5 * keras.ops.sum((z ** 2) + log_two_pi + 2.0 * keras.ops.log(std), axis=-1)

        def sample(self, batch_shape, mean, std):
            eps = keras.random.normal(keras.ops.shape(mean))
            return mean + std * eps

        if use_custom_score:
            def score(self, estimates, targets, weights=None):
                scores = -self.log_prob(x=targets, **estimates)
                if weights is None:
                    return keras.ops.mean(scores)

                weights = keras.ops.cast(weights, keras.ops.dtype(scores))
                return keras.ops.sum(scores * weights) / keras.ops.sum(weights)

    return bf.networks.ScoringRuleNetwork(
        scoring_rules={"diag_gaussian": DiagonalNormalScore()},
        subnet="mlp",
    )


def build_workflow(simulator, posterior_family="flow"):
    bf, keras = _require_workflow_backend()
    if posterior_family == "flow":
        inference_network = "coupling_flow"
    elif posterior_family == "diag_gaussian":
        inference_network = _build_diag_gaussian_inference_network(bf, keras)
    else:
        raise ValueError("posterior_family must be one of {'flow', 'diag_gaussian'}.")

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        inference_variables=["z_beta", "z_gamma"],
        summary_variables=["obs"],
        summary_network="time_series_network",
        inference_network=inference_network,
    )
    workflow.approximator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4)
    )
    return workflow


def train_workflow(epochs=12, batch_size=32, num_batches=60, seed=42, posterior_family="flow"):
    bf, keras = _require_workflow_backend()
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    train_rng = np.random.default_rng(seed)
    sim_one = make_training_simulator(train_rng)
    simulator = bf.make_simulator(sim_one)
    workflow = build_workflow(simulator, posterior_family=posterior_family)
    history = workflow.approximator.fit(
        simulator=simulator,
        epochs=epochs,
        batch_size=batch_size,
        num_batches=num_batches,
        workers=1,
        verbose=2,
    )
    return workflow, history


def sample_posterior(workflow, obs, num_samples=2000):
    try:
        samples = workflow.sample(
            num_samples=num_samples,
            conditions={"obs": obs[None, ...]},
            split=True,
        )
    except NotImplementedError:
        samples = workflow.sample(
            num_samples=num_samples,
            conditions={"obs": obs[None, ...]},
            split=False,
        )
    z_beta_samps = np.asarray(samples["z_beta"]).reshape(-1)
    z_gamma_samps = np.asarray(samples["z_gamma"]).reshape(-1)
    beta_samps = from_unconstrained(z_beta_samps, *PRIOR_BOUNDS["beta"])
    gamma_samps = from_unconstrained(z_gamma_samps, *PRIOR_BOUNDS["gamma"])
    return {
        "z_beta": z_beta_samps,
        "z_gamma": z_gamma_samps,
        "beta": beta_samps,
        "gamma": gamma_samps,
    }
