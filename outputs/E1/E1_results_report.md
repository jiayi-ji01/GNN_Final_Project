# E1 Structural Misspecification Report

## Setup

- Truth model: SEIR
- Inference model: SIR-trained BayesFlow amortizer
- Purpose: evaluate structural misspecification caused by ignoring the exposed compartment

## Key Findings

- Posterior median bias:
  - beta: 0.0216
  - gamma: 0.0390
  - R0: 0.0000
- Empirical 90% coverage:
  - beta: 0.7900
  - gamma: 0.8500
  - R0: 0.9500
- Feature-level PPC:
  - peak_time empirical 90% coverage: 0.9500
  - early_growth_slope empirical 50% coverage: 0.4500
- Curve-level PPC:
  - mean full-curve RMSE: 0.0214
  - mean pointwise empirical 90% coverage: 0.9414

## Interpretation

Under E1, the inference model is forced to explain SEIR-generated observations with a simpler SIR family. This primarily tests whether the missing latent exposure stage distorts parameter recovery, narrows posterior intervals too aggressively, or creates predictive mismatches that are more visible in feature-level or curve-level diagnostics than in matched settings.

In practice, the report should be interpreted by comparing transmission-related quantities, recovery-related quantities, and predictive diagnostics together rather than relying on a single metric alone.
