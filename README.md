# Posterior Reliability of BayesFlow under Model Misspecification in Epidemic Inverse Problems

## Overview

This project studies **posterior reliability** in epidemic inverse problems using **BayesFlow**.  
The main question is whether a posterior remains trustworthy when the simulator used for inference does not match the true data-generating process.

We focus on three aspects of reliability:

- posterior center
- posterior uncertainty
- posterior predictive adequacy

## Research Questions

This project is organized around three questions:

1. Does BayesFlow produce reliable posteriors in the **matched setting**?
2. How does reliability degrade under **model misspecification**?
3. Do posterior/model design choices affect the conclusion?

## Project Structure

```text
gnn_project/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── simulators.py
│   ├── bayesflow_model.py
│   ├── diagnostics.py
│   └── plotting.py
│
├── scripts/
│   ├── run_matched.py
│   ├── run_misspec.py
│   └── run_ablation.py
│
├── outputs/
│   ├── models/
│   ├── results/
│   └── figures/
│
└── notebooks/
    ├── 01_setup.ipynb
    ├── 02_baseline.ipynb
    ├── 03_misspecification.ipynb
    ├── 04_ablation.ipynb
    └── 05_conclusion.ipynb
```

### Main Files

- `src/simulators.py`: epidemic simulators
- `src/bayesflow_model.py`: BayesFlow training and inference
- `src/diagnostics.py`: reliability diagnostics
- `src/plotting.py`: figures and visual summaries
- `scripts/`: runnable experiment scripts
- `outputs/`: saved models, results, and figures
- `notebooks/`: experiment notebooks and analysis
