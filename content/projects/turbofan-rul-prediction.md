---
title: "Turbofan Engine RUL Prediction"
date: 2026-01-21
draft: false
categories: ["Projects"]
summary: "An end-to-end predictive maintenance system for aircraft turbofan engines. RUL prediction on NASA CMAPSS dataset using Random Forest, XGBoost, sklearn Pipelines & MLflow."
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
featured: true
---
<a href="https://github.com/elouanXP/turbofan-rul-prediction" target="_blank">**View on GitHub →**</a>

## 1- Overview

**An end-to-end predictive maintenance system for aircraft turbofan engines. RUL prediction on NASA CMAPSS dataset using Random Forest, XGBoost, sklearn Pipelines & MLflow.**

![turbofan_cmapss.png](/images/turbofan_cmapss.png)

---

## 2- Context & Motivation

### Problem Statement

Aviation maintenance operates on two modes: **scheduled** (replace every N hours regardless of condition) and **unscheduled** (repair after failure). The first wastes money by replacing healthy components while the second risks catastrophic in-flight failure. Predictive maintenance is the third way: maintenance operations are scheduled when and only when a component is approaching failure.

The business case is substantial: a single unplanned engine removal can rapidly cost hundreds of thousands of dollars in AOG (Aircraft on Ground)fees and emergency parts, before even accounting for safety consequences.

### Technical challenges

An aircraft turbofan engine degrades continuously through thousands of operational cycles. The challenge is to:

- Extract a clean degradation signal from noisy, high-dimensional time-series data
- Predict the number of remaining cycles with sufficient accuracy to schedule maintenance in advance
- Generalize to engines the model has never observed

To do so, **RUL (Remaining Useful Life)** is introduced. Predicting it accurately is the key challenge in maintenance prediction problems, and can be applied to a broad range of scientific and industrial applications.

![rul.png](/images/rul.png)

### Project Goal & Methodology

Build an end-to-end ML pipeline from raw sensor data to operational risk assessment, using the NASA CMAPSS benchmark dataset, validated against the official NASA test set and the PHM 2008 competition scoring function.

{{< rawhtml >}}
<!-- Project pipeline diagram -->
<div style="margin: 1.5rem 0;">
<svg viewBox="0 0 860 90" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:860px;display:block;margin:auto;font-family:monospace;">
  <!-- Boxes -->
  <rect x="5"   y="20" width="120" height="50" rx="6" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <rect x="160" y="20" width="120" height="50" rx="6" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <rect x="315" y="20" width="150" height="50" rx="6" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <rect x="500" y="20" width="140" height="50" rx="6" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.5"/>
  <rect x="675" y="20" width="180" height="50" rx="6" fill="#ede7f6" stroke="#ce93d8" stroke-width="1.5"/>
  <!-- Arrows -->
  <path d="M125 45 L158 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M280 45 L313 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M465 45 L498 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M640 45 L673 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- Arrow marker -->
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>
  <!-- Labels -->
  <text x="65"  y="42" text-anchor="middle" font-size="11" fill="#1565c0">Raw CMAPSS</text>
  <text x="65"  y="57" text-anchor="middle" font-size="11" fill="#1565c0">data</text>
  <text x="220" y="42" text-anchor="middle" font-size="11" fill="#2e7d32">EDA &amp;</text>
  <text x="220" y="57" text-anchor="middle" font-size="11" fill="#2e7d32">exploration</text>
  <text x="390" y="42" text-anchor="middle" font-size="11" fill="#e65100">Feature</text>
  <text x="390" y="57" text-anchor="middle" font-size="11" fill="#e65100">engineering</text>
  <text x="570" y="42" text-anchor="middle" font-size="11" fill="#880e4f">Model training</text>
  <text x="570" y="57" text-anchor="middle" font-size="11" fill="#880e4f">&amp; tuning</text>
  <text x="765" y="42" text-anchor="middle" font-size="11" fill="#4a148c">NASA benchmark</text>
  <text x="765" y="57" text-anchor="middle" font-size="11" fill="#4a148c">&amp; risk analysis</text>
</svg>
</div>
{{< /rawhtml >}}

## 3- Dataset

CMAPSS (Commercial Modular Aero-Propulsion System Simulation) was released by NASA Ames Research Center for the 2008 PHM challenge and remains the standard benchmark for RUL prediction. It simulates run-to-failure degradation of a turbofan engine's high-pressure compressor.

This project uses the FD001 subset: 100 training engines and 100 test engines, all operating under a single condition with a single fault mode.

Each row in the dataset represents one engine at one operational cycle, with 21 sensor readings and 3 operating condition settings:

| # | Symbol | Description | Unit |
|---|--------|-------------|------|
| **Engine** | | | |
| **Cycle** | | | |
| **Setting 1** | | Altitude | ft |
| **Setting 2** | | Mach Number | M |
| **Setting 3** | | TRA (Throttle-Resolver Angle) | deg |
| **Sensor 1** | T2 | Total temperature at fan inlet | °R |
| **Sensor 2** | T24 | Total temperature at LPC outlet | °R |
| **Sensor 3** | T30 | Total temperature at HPC outlet | °R |
| **Sensor 4** | T50 | Total temperature at LPT outlet | °R |
| **Sensor 5** | P2 | Pressure at fan inlet | psia |
| **Sensor 6** | P15 | Total pressure in bypass-duct | psia |
| **Sensor 7** | P30 | Total pressure at HPC outlet | psia |
| **Sensor 8** | Nf | Physical fan speed | rpm |
| **Sensor 9** | Nc | Physical core speed | rpm |
| **Sensor 10** | epr | Engine pressure ratio | – |
| **Sensor 11** | Ps30 | Static pressure at HPC outlet | psia |
| **Sensor 12** | phi | Ratio of fuel flow to Ps30 | pps/psi |
| **Sensor 13** | NRf | Corrected fan speed | rpm |
| **Sensor 14** | NRc | Corrected core speed | rpm |
| **Sensor 15** | BPR | Bypass ratio | – |
| **Sensor 16** | farB | Burner fuel-air ratio | – |
| **Sensor 17** | htBleed | Bleed enthalpy | – |
| **Sensor 18** | Nf_dmd | Demanded fan speed | rpm |
| **Sensor 19** | PCNfR_dmd | Demanded corrected fan speed | rpm |
| **Sensor 20** | W31 | HPT coolant bleed | lbm/s |
| **Sensor 21** | W32 | LPT coolant bleed | lbm/s |

- LPC/HPC: Low/High Pressure Compressor
- LPT/HPT: Low/High Pressure Turbine

Engine 1 sensors show a clear monotonic degradation while others are flat or pure noise:

![Sensor evolution over cycles for engine 1](/images/01_settings_and_sensors_unit_1_FD001.png)


Engine lifetimes in the training set range from 128 to 362 cycles, giving the following distribution:

![Engine lifetime distribution](/images/01_max_time_cycles_distribution_FD001.png)


---

## 4- Feature Engineering
Raw sensor readings require significant transformation before they can be used as model inputs. Four operations are applied in sequence:
{{< rawhtml >}}
<svg viewBox="0 0 820 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:820px;display:block;margin:1.2rem auto;font-family:monospace;">
  <defs>
    <marker id="a2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#888"/>
    </marker>
  </defs>
  <!-- Step boxes -->
  <rect x="10"  y="75" width="155" height="50" rx="6" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <rect x="210" y="75" width="155" height="50" rx="6" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <rect x="410" y="75" width="155" height="50" rx="6" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <rect x="610" y="75" width="175" height="50" rx="6" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.5"/>
  <!-- Arrows -->
  <path d="M165 100 L208 100" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>
  <path d="M365 100 L408 100" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>
  <path d="M565 100 L608 100" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>
  <!-- Box labels -->
  <text x="87"  y="97"  text-anchor="middle" font-size="11" fill="#1565c0" font-weight="bold">RUL target</text>
  <text x="87"  y="113" text-anchor="middle" font-size="11" fill="#1565c0">clip=120</text>
  <text x="287" y="97"  text-anchor="middle" font-size="11" fill="#2e7d32" font-weight="bold">Low variance removal</text>
  <text x="287" y="113" text-anchor="middle" font-size="11" fill="#2e7d32">var&lt;1e-5</text>
  <text x="487" y="97"  text-anchor="middle" font-size="11" fill="#e65100" font-weight="bold"> Temporal features</text>
  <text x="487" y="113" text-anchor="middle" font-size="11" fill="#e65100">window=5</text>
  <text x="697" y="97"  text-anchor="middle" font-size="11" fill="#880e4f" font-weight="bold">Low RUL correlation removal</text>
  <text x="697" y="113" text-anchor="middle" font-size="11" fill="#880e4f">|corr|&lt;0.1</text>
  <!-- Sub-labels below boxes -->
  <text x="87"  y="145" text-anchor="middle" font-size="9.5" fill="#666">max_cycles − t</text>
  <text x="287" y="145" text-anchor="middle" font-size="9.5" fill="#666">7 sensors + 3 settings removed</text>
  <text x="487" y="145" text-anchor="middle" font-size="9.5" fill="#666">rolling mean/std, diff</text>
  <text x="697" y="145" text-anchor="middle" font-size="9.5" fill="#666">28 features retained</text>
</svg>
{{< /rawhtml >}}

### RUL computation and clipping

For each engine:
{{< rawhtml >}}
$$\text{RUL}(t) = \text{max\_cycles} - t$$
{{< /rawhtml >}}

A clip of 120 cycles is applied. Beyond this value, the engine is healthy and predicting a precise RUL has less operational relevance. Indeed, maintenance is not scheduled 300 cycles in advance. This clipping is standard in the CMAPSS literature and also speeds up convergence so that the model focuses on the region where prediction matters.

### Low-variance feature removal

Features whose variance falls below $10^{-5}$ are constant (or near-constant) and carry no useful information.

### Temporal features

Mechanical degradation is a slow process. In order to capture underlying trends while smoothing the noise, a 5-cycle rolling mean and standard deviation are implemented, as well as an instantaneous difference to capture rapid changes. For each retained sensor we then get:

- `rolling_mean_5` for local trend
- `rolling_std_5` for local variability
- `diff_1` (first-order difference) for the rate of change

### Correlation filtering

Features with $|\text{corr}(\text{RUL})| \leq 0.1$ are removed. The final feature set contains 28 features.

![Feature correlation with RUL](/images/02_rul_corr_heatmap_FD001.png)

---

## 5- Model Selection & Training

The split is performed by engine unit (not randomly across rows) to prevent data leakage: the same engine can't appear in both training and test sets. A standard progression from simple to more complex models is then followed.

### Baseline

The baseline model systematically predicts the mean RUL of the training set for every observation. It is the minimum benchmark to outperform.

### Linear Regression

First supervised model, scale-sensitive so wrapped in a `StandardScaler`, to discover if the degradation signal is partially linearly exploitable.

### Random Forest

As a reminder, Random Forest is an ensemble learning model of decision trees, each trained on a random subset of features and data (bagging). The final prediction is the average across all trees. Each tree sees a random bootstrap sample of the data and at each split considers only a random subset of features. This double randomness reduces variance (overfitting) while keeping each tree individually interpretable. 

![Random Forest](/images/rf.png)

Before hyperparameter tuning, a vanilla Random Forest is trained with default parameters, and the top 15 features by importance are selected. Feature importance is measured by average impurity decrease (Gini) across all splits — providing interpretable feature rankings. Reducing dimensionality at this stage lowers overfitting and speeds up the grid search.

### XGBoost

Gradient-boosted trees: rather than building trees in parallel (like RF), XGBoost builds them sequentially, each tree correcting the residual errors of the previous. This makes it more data-efficient, but also more prone to overfitting without regularization. 

![XGBoost](/images/xgboost.png)

Before hyperparameter tuning, a vanilla XGBoost is trained with default parameters, and the top 15 features by importance are selected. XGBoost uses gain-based feature importance (improvement in loss from each split), which ranks features differently than RF's Gini impurity.

### Hyperparameter tuning (Random Forest and XGBoost)

 `GridSearchCV()` over the key regularization parameters:

| Model         | Parameters tuned               |
|---------------|--------------------------------|
| Random Forest |`n_estimators` ∈ {100, 200, 300}|
|               |`max_depth` ∈ {None, 10, 20}    |
|               |`min_samples_leaf` ∈ {1, 3, 5}  |
| XGBoost       |`n_estimators` ∈ {100, 200, 300}|
|               | `max_depth` ∈ {3, 5, 7}        |
|               | `learning_rate` ∈ {0.05, 0.1, 0.2}|

**Preventing data leakage with GroupKFold for cross-validation**

A random train/test split would place different cycles of the same engine in both train and test folds. GroupKFold partitions by engine unit, ensuring each fold contains engines unseen during training, therefore avoiding any data leakage.

{{< rawhtml >}}
<svg viewBox="0 0 700 125" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;display:block;margin:1.2rem auto;font-family:monospace;">
  <text x="10" y="20" font-size="11" fill="#333" font-weight="bold">
    GroupKFold cross-validation (k=3, split by engine unit)
  </text>

  <!-- Row labels -->
  <text x="10" y="52"  font-size="10" fill="#555">Fold 1</text>
  <text x="10" y="82"  font-size="10" fill="#555">Fold 2</text>
  <text x="10" y="112" font-size="10" fill="#555">Fold 3</text>

  <!-- Fold 1 -->
  <rect x="55" y="37" width="200" height="22" rx="3" fill="#ef9a9a" stroke="#e53935" stroke-width="1"/>
  <rect x="255" y="37" width="400" height="22" rx="3" fill="#a5d6a7" stroke="#43a047" stroke-width="1"/>
  <text x="155" y="52" text-anchor="middle" font-size="9" fill="#b71c1c">
    Val (engines 1–33)
  </text>
  <text x="455" y="52" text-anchor="middle" font-size="9" fill="#1b5e20">
    Train (engines 34–100)
  </text>

  <!-- Fold 2 -->
  <rect x="55"  y="67" width="200" height="22" rx="3" fill="#a5d6a7" stroke="#43a047" stroke-width="1"/>
  <rect x="255" y="67" width="200" height="22" rx="3" fill="#ef9a9a" stroke="#e53935" stroke-width="1"/>
  <rect x="455" y="67" width="200" height="22" rx="3" fill="#a5d6a7" stroke="#43a047" stroke-width="1"/>
  <text x="155" y="82" text-anchor="middle" font-size="9" fill="#1b5e20">Train</text>
  <text x="355" y="82" text-anchor="middle" font-size="9" fill="#b71c1c">
    Val (34–66)
  </text>
  <text x="555" y="82" text-anchor="middle" font-size="9" fill="#1b5e20">Train</text>

  <!-- Fold 3 -->
  <rect x="55"  y="97" width="400" height="22" rx="3" fill="#a5d6a7" stroke="#43a047" stroke-width="1"/>
  <rect x="455" y="97" width="200" height="22" rx="3" fill="#ef9a9a" stroke="#e53935" stroke-width="1"/>
  <text x="255" y="112" text-anchor="middle" font-size="9" fill="#1b5e20">
    Train (engines 1–66)
  </text>
  <text x="555" y="112" text-anchor="middle" font-size="9" fill="#b71c1c">
    Val (67–100)
  </text>

</svg>
{{< /rawhtml >}}

---

## 6- Results

Each model is wrapped in a **sklearn Pipeline** (scaler → model). **MLflow** experiment tracking is implemented and enables direct run comparison in the MLflow UI.

### Metrics (RMSE, MAE, R², NASA Score)

Standard regression metrics for model selection:
{{< rawhtml >}}
$$
\mathbf{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$
$$
\mathbf{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$
$$
\mathbf{R^2} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
{{< /rawhtml >}}

In addition, the **NASA asymmetric scoring function** for operational validation is introduced.

For each engine, the individual score is:

{{< rawhtml >}}
$$s_i = \begin{cases} e^{-d_i/13} - 1 & \text{if } d_i < 0 \text{ (early prediction)} \\ e^{d_i/10} - 1 & \text{if } d_i \geq 0 \text{ (late prediction)} \end{cases}$$
{{< /rawhtml >}}

where $d_i = \hat{y}_i - y_i$ is the prediction error.

The total score is $S = \sum_i s_i$. Lower is better, an undetected failure is far more costly than a preventive maintenance action.

![NASA scoring function asymmetry](/images/score.png)

### Model comparison

![Model comparison — RMSE, MAE, R²](/images/03_model_comparison_FD001.png)

| Model | RMSE (test) | MAE (test) | R² (test) |
|-------|:-----------:|:----------:|:---------:|
| Baseline | 39.92 | 35.26 | 0.000 |
| Linear Regression | 19.00 | 15.39 | 0.773 |
| Random Forest | 15.90 | 10.92 | 0.841 |
| **RF Tuned** | **15.51** | **10.71** | **0.849** |
| XGBoost | 16.98 | 11.48 | 0.819 |
| XGBoost Tuned | 15.53 | 11.18 | 0.849 |

Tuning narrows the train/test gap substantially in both RF and XGBoost, confirming the initial vanilla models were overfitting. RF Tuned and XGBoost Tuned reach essentially identical performance. Random Forest Tuned model is selected as the final model for its robustness and interpretability.

### NASA benchmark

The final model is retrained on the full training set and evaluated on the official NASA test file, a completely held-out set never seen during development.

| Metric | Value |
|--------|-------|
| RMSE | 17.34 cycles |
| MAE | 12.10 cycles |
| R² | 0.806 |
| NASA Score | 902.7 |

![NASA benchmark — true vs predicted RUL](/images/03_nasa_benchmark_FD001.png)

![True vs predicted RUL](/images/result1.png)

A NASA score of 902.7 is an encouraging result, ranking our model among the top 6 (based on the PHM 2008 competition results). However, this should be interpreted with caution, as we only used the FD001 dataset, which is the easiest subset.

### Operational risk analysis

In a real maintenance context, maintenance is triggered when the predicted RUL falls below an operational threshold, determined by a human operator. For aerospace applications, the threshold should be set conservatively low given the catastrophic cost of an in-flight failure.

To translate our model performance into business value, we categorize predictions into three operational zones:

| Zone | Condition | Operational consequence |
|------|-----------|------------------------|
| **Safe** | $ abs(d) <= 13 $ cycles | Maintenance planned correctly |
| **Early warning** | $d < -13$ cycles | Unnecessary early intervention |
| **Danger** | $d > 13$ cycles | Engine may fail before maintenance, critical safety risk |

The threshold of 13 cycles is derived from the NASA scoring function scale parameter for early predictions.

![Operational risk per engine](/images/04_operational_risk_FD001.png)

---

## 7- Conclusion & Next Steps

### Physical Interpretation

T50 (LPT outlet temperature) emerges as the dominant predictor of RUL. This result aligns with FD001 dataset single fault mode. Because T50 sits at the very end of the gas path, it acts as a natural integrator of all upstream degradation mechanisms, making it the most informative single signal for RUL estimation.

The use of a 5-cycle rolling mean across all selected features is physically motivated: mechanical degradation is a slow, cumulative process whose signature is more reliably captured over several consecutive cycles than in any instantaneous measurement.

### Limitations and next steps

- FD001 is the simplest CMAPSS subset: single operating condition, single fault mode. Extending to FD002/FD003/FD004 (multiple conditions, multiple fault modes) would better reflect real-world complexity.

- The RUL clipping at 120 cycles is a modeling assumption. In production, a two-stage model could first classify whether the engine is in its degradation phase before predicting RUL.

- LSTM / Transformer architectures are known to outperform classical ML models on this benchmark by explicitly modeling temporal dependencies, this is what I will focus on next.

---
`Python` · `pandas` · `numpy` · `scikit-learn` · `sklearn Pipeline` · `XGBoost` · `MLFlow` · `matplotlib` · `seaborn` · `joblib`