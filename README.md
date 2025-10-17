# Predictive Maintenance (RUL) — AI Project

This repository contains the code and a Colab notebook used to develop and evaluate machine‑learning models for predictive maintenance of mechanical components, specifically targeting Remaining Useful Life (RUL) prediction.

Important: the code cannot be executed end‑to‑end out of the box because the datasets are proprietary to the University of Bologna. 

## Repository Structure

- `colab.ipynb` — Main notebook that orchestrates the workflow:
  - Imports, dataset loading, and preprocessing hooks (expects pre‑generated datasets).
  - Model definitions: a TimeDistributed Conv2D + LSTM architecture and a Conv2D‑only baseline.
  - Metric/score definitions (nMAE, MAPE, relative overestimation penalty, normalized std, and a weighted aggregate score).
  - Two‑phase hyperparameter search (global randomized search followed by local refinement with time budgets).
  - Training loops, best‑model selection, statistics/plots, and utilities to extract the best model’s layer parameters.

- `constants.py` — Points to the expected dataset files:
  - `dataset_lstm_name = "repository_lstm_12-14-15-16-19_17-18.dat"`
  - `dataset_conv_name = "repository_conv_15-16_17.dat"`

- `debug/` — Utilities and experimental model variants:
  - `debug/debug_models.py`: alternative Conv+LSTM architectures used during prototyping and ablations.

- `thesis/` — Reference material:
  - `thesis/Studio e applicazione di modelli di Artificial Intellligence per la manutenzione predittiva di componenti meccanici.pdf`
  (My thesis document, Italian language) — consult for methodology details, assumptions, and results.

## Data Availability

- The datasets used in this project are owned by the University of Bologna and are not publicly available in this repository.

## Methodology & Contributions

This work focuses on modeling RUL as a supervised learning problem and contributes the following pieces:

- Conv+LSTM architecture for RUL
  - TimeDistributed Conv2D feature extraction across temporal slices feeding an LSTM layer, followed by TimeDistributed dense heads for per‑timestep regression.
  - Parameterized design (filter scaling, strides, activations, dropout) to enable systematic search.

- Conv2D baseline
  - A compact 2D‑only model for comparison and ablation studies.

- Metrics and scoring
  - Normalized MAE, MAPE, a relative overestimation penalty (heavier near end‑of‑life), normalized error standard deviation, and a weighted aggregate score that prioritizes MAPE while incorporating precision and safety considerations.

- Two‑phase hyperparameter search
  - Global randomized search to explore broad configurations.
  - Local refinement around the current best within fixed time budgets, tracking the minimum aggregate score on the test split.

- Training workflow and best‑model handling
  - Model compilation (Adam + MSE), training loops, model selection and saving, prediction post‑processing (e.g., non‑negative clipping for RUL), and visual summaries.

- Introspection and retraining utilities
  - Tools to extract layer configurations (activations, filters, strides, dropout, LSTM units) from the best model.
  - Optional retraining of the best model to inspect loss curves and convergence dynamics.

For a comprehensive exposition (motivation, industrial setting, and results), consult the thesis document in `thesis/` (Italian).

## Limitations & Notes

- Without the University of Bologna datasets (and any pretrained `.h5` models referenced in the notebook), the pipeline cannot be executed end‑to‑end.
- Results depend on random sampling in the search phase; set seeds and/or persist checkpoints for reproducibility if needed.
- Paths and filenames in `constants.py` are placeholders and must match your local data placement.

## Acknowledgements

This work was carried out in the context of academic research at the University of Bologna. Data used in experiments are the property of the University and are subject to their usage policies.
