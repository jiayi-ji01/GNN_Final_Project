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


def build_workflow(simulator):
    bf, keras = _require_workflow_backend()
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        inference_variables=["z_beta", "z_gamma"],
        summary_variables=["obs"],
        summary_network="time_series_network",
        inference_network="coupling_flow",
    )
    workflow.approximator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4)
    )
    return workflow


def train_workflow(epochs=12, batch_size=32, num_batches=60, seed=42):
    bf, keras = _require_workflow_backend()
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    train_rng = np.random.default_rng(seed)
    sim_one = make_training_simulator(train_rng)
    simulator = bf.make_simulator(sim_one)
    workflow = build_workflow(simulator)
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
    samples = workflow.sample(
        num_samples=num_samples,
        conditions={"obs": obs[None, ...]},
        split=True,
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
