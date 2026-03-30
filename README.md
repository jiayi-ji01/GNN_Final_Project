# Posterior Reliability of BayesFlow in Epidemic Inverse Problems under Model Misspecification

Simulation-based inference experiments for studying how posterior quality changes when epidemic models are misspecified.

## Overview

This project uses **BayesFlow** with classical epidemic simulators to study a simple question:

> When predictive curves still look reasonable, is the posterior still trustworthy?

The repository compares a matched baseline against three misspecification settings:

- **E0**: matched baseline (`SIR -> SIR`)
- **E1**: structural misspecification (`SEIR truth -> SIR inference`)
- **E2**: dynamical misspecification (time-varying transmission)
- **E3**: observation-process misspecification (time-varying reporting distortion)

The main diagnostics are:

- parameter recovery
- empirical coverage
- feature-level PPC
- curve-level PPC

## Highlights

- BayesFlow-based amortized posterior inference
- Controlled simulator-generated epidemic time series
- Severity analyses for E1, E2, and E3
- Pseudo-truth projection for structural mismatch
- Reproducible scripts and notebook reports

## Repository Structure

```text
.
├── notebooks/
│   ├── 00_E0_matched_baseline.ipynb
│   ├── 01_E1_structural_misspecification.ipynb
│   ├── 02_E2_dynamical_misspecification.ipynb
│   └── 03_E3_observation_process_misspecification.ipynb
├── outputs/
│   ├── E0/
│   ├── E1/
│   ├── E2/
│   └── E3/
├── scripts/
│   ├── train_e0_workflow.py
│   ├── run_e0_evaluation.py
│   ├── run_e1_evaluation.py
│   ├── run_e2_evaluation.py
│   └── run_e3_evaluation.py
└── src/
    ├── config.py
    ├── simulators.py
    ├── workflow.py
    ├── diagnostics.py
    ├── experiments.py
    └── plotting.py
```

## Installation

```bash
git clone https://github.com/jiayi-ji01/GNN_Final_Project.git
cd GNN_Final_Project

python -m venv .venv
source .venv/bin/activate

pip install bayesflow tensorflow keras numpy pandas scipy matplotlib seaborn jupyter
```

## Usage

Train the matched amortizer:

```bash
python scripts/train_e0_workflow.py
```

Run each experiment:

```bash
python scripts/run_e0_evaluation.py
python scripts/run_e1_evaluation.py
python scripts/run_e2_evaluation.py
python scripts/run_e3_evaluation.py
```

Open the notebooks:

```bash
jupyter notebook
```

Recommended order:

1. `00_E0_matched_baseline.ipynb`
2. `01_E1_structural_misspecification.ipynb`
3. `02_E2_dynamical_misspecification.ipynb`
4. `03_E3_observation_process_misspecification.ipynb`

## Data

This project uses **simulator-generated epidemic time series** rather than an external dataset.

Data are generated from:

- SIR
- SEIR
- time-varying transmission SIR
- time-varying observation-process distortion

Simulation is necessary because the project studies **posterior recovery and calibration under misspecification**, which requires known ground truth.

## Results

Representative outputs:

- [E0 baseline figure](./outputs/E0/E0_matched_overall_baseline_figure.png)
- [E1 summary figure](./outputs/E1/E1_summary_4panel.png)
- [E2 severity summary](./outputs/E2/severity_summary/E2_severity_3panel_summary.png)
- [E3 controlled severity grid](./outputs/E3/E3_controlled_severity_grid.png)

Main conclusion:

> Predictive adequacy does not guarantee posterior reliability.

## Configuration

Important settings are defined in [`src/config.py`](./src/config.py), including:

- prior bounds
- observation noise
- number of days
- severity grids for E1, E2, and E3
- output paths

## Reproducibility

To reproduce the main results:

1. use fixed seeds
2. run the scripts in `scripts/`
3. inspect the saved outputs in `outputs/`
4. use the notebooks for presentation and interpretation

## Dependencies

- Python
- BayesFlow
- TensorFlow
- Keras
- NumPy
- pandas
- SciPy
- matplotlib
- seaborn
- Jupyter

## Future Work

- add stronger calibration diagnostics
- compare against non-amortized Bayesian baselines
- test more robust misspecification-aware inference methods
- extend to semi-synthetic or real epidemic datasets

## Citation

```bibtex
@misc{epidemic_misspecification_project,
  title        = {Posterior Reliability of BayesFlow in Epidemic Inverse Problems under Model Misspecification},
  author       = {jiayi-ji01},
  year         = {2026},
  howpublished = {\url{https://github.com/jiayi-ji01/GNN_Final_Project}},
  note         = {GitHub repository}
}
```

## License

Add your preferred open-source license here, for example `MIT`.
