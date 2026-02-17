---
title: "Turbofan Engine RUL Prediction"
date: 2026-01-21
draft: false
categories: ["Projects"]
summary: "End-to-end predictive maintenance system for aircraft turbofan engines. RUL prediction on NASA CMAPSS dataset — RMSE 17.34 · R² 0.806 on official NASA benchmark."
cover:
  image: ""
  alt: "Turbofan RUL Prediction"
  relative: false
showToc: true
TocOpen: false
hidemeta: false
comments: false
disableShare: true
ShowReadingTime: false
ShowBreadCrumbs: false
math: true
---

## Overview

Predictive maintenance system for aircraft turbofan engines using the **NASA CMAPSS dataset**.  
The model predicts the **Remaining Useful Life (RUL)** — the number of operational cycles remaining before failure — enabling condition-based maintenance decisions.

<a href="https://github.com/elouanXP/turbofan-rul-prediction" target="_blank">**View on GitHub →**</a>

---

## Results

| Model | RMSE (test) | MAE (test) | R² (test) |
|-------|-------------|------------|-----------|
| Baseline | 39.92 | 35.26 | -0.000 |
| Linear Regression | 19.00 | 15.39 | 0.773 |
| Random Forest | 15.90 | 10.92 | 0.841 |
| RF Tuned | 15.51 | 10.71 | 0.849 |
| XGBoost | 16.98 | 11.48 | 0.819 |
| XGBoost Tuned | 15.53 | 11.18 | 0.849 |

**NASA Benchmark (full train → test_FD001.txt)** : RMSE = 17.34 · MAE = 12.10 · R² = 0.806

---

## Dataset

**CMAPSS** (Commercial Modular Aero-Propulsion System Simulation) — NASA Ames Research Center.  
Simulates the degradation of a turbofan engine high-pressure compressor from healthy state to failure.

- FD001 : 100 training engines · 100 test engines
- Single operating condition · single fault mode
- 21 sensor measurements per cycle
- Objective : predict remaining cycles at last observed measurement

---

## Methodology

### 1. Feature Engineering
- RUL computation with clipping at 120 cycles
- Low-variance feature removal
- Temporal features per engine : rolling mean (window=5), rolling std, first difference
- Correlation filtering against RUL target

### 2. Modelling
- Unit-based train/test split to prevent data leakage
- GroupKFold cross-validation by engine unit
- sklearn Pipeline (scaler + model) for consistent preprocessing
- MLflow experiment tracking — all runs logged with parameters, metrics and artifacts
- Feature selection by model importance (top 15 features)
- GridSearchCV hyperparameter tuning

### 3. Final Evaluation
- Model retrained on full training set
- Evaluated on official NASA test_FD001.txt benchmark
- NASA asymmetric scoring function 

![Model comparison](/images/03_model_comparison_FD001.png)
![NASA benchmark](/images/03_nasa_benchmark_FD001.png)

---

## NASA Scoring Function

Standard regression metrics treat all errors equally. The NASA scoring function captures the **asymmetry of prediction errors** in a safety-critical context:

$$s_i = \begin{cases} e^{-d/13} - 1 & d < 0 \text{ (early — unnecessary maintenance)} \\\ e^{d/10} - 1 & d \geq 0 \text{ (late — potential failure)} \end{cases}$$

Late predictions are penalized more severely than early ones — an undetected failure is far more costly than a preventive maintenance action.

---

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `MLflow` · `sklearn Pipeline` · `joblib` · `pandas` · `numpy` · `matplotlib`
