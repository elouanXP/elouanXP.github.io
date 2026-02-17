---
title: "EV Smart Charge Scheduling"
date: 2023-05-07
draft: false
categories: ["Projects"]
summary: "Optimized charging policy for electric vehicles using LSTM forecasting and convex optimization to minimize CO2 emissions — 57% emission reduction vs baseline charging."
cover:
  image: ""
  alt: "EV Smart Charge Scheduling"
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

As electric vehicle adoption accelerates, home charging during peak evening hours creates a double problem: high grid load and high emissions (solar energy unavailable at night). This research project developed an **optimized charging policy** that schedules EV charging to minimize CO2 emissions while ensuring sufficient battery charge for daily use.

<a href="https://github.com/elouanXP/ev-smart-charge-scheduling" target="_blank">**View on GitHub →**</a>

---

## Key Results

| Strategy | Average CO2 Emission (lbs) | Reduction vs Baseline |
|----------|---------------------------|----------------------|
| Baseline Charging | 7.5 | — |
| Shift Charging | 3.2 | **57%** |
| Vehicle-to-Grid | -0.8 | **111%** (net negative) |

- **Shift charging** reduces emissions by 57% by charging during low-emission windows
- **Vehicle-to-grid** achieves net-negative emissions in 68% of sessions
- **65% of peak-hour demand** can be shifted to idle periods

---

## Problem Statement

Electric vehicles charged overnight coincide with the **highest emission periods** — when renewable energy (especially solar) is unavailable. The standard approach of plugging in and charging to 100% is both:

- **Environmentally inefficient** — charging when the grid is dirtiest
- **Operationally wasteful** — most users don't need a full charge every day

We hypothesized that a dynamic charging policy could:
1. Charge opportunistically during low-emission windows
2. Charge only to the SOC needed for the next trip (plus buffer)
3. Reduce grid load during peak hours

---

## Methodology

### 1. Emission Curve Forecasting

**Objective**: Predict the Marginal Operating Emission Rate (MOER) — CO2 lbs/MWh at 5-minute intervals — for the next 24 hours.

**Data**: CAISO North MOER API (WattTime) — one full year of emission data for Northern California.

**Challenge**: Unpredictable daily spikes in the data. During summer, renewable energy creates consistent low-emission windows. During winter, spikes occur randomly due to inconsistent weather patterns.

**Models developed**:

**Feed-Forward Neural Network (FFNN)**  
8 hidden layers · ReLU activation · predicts next 288 values (24h) from previous 1000 points  
→ Huber loss (summer) = 77.8 · (winter) = 320.6  
→ Accurate for summer, poor winter spike timing

**LSTM Recurrent Neural Network**  
Trained on 25 days to predict 1 full day · window = 20 values · 10 epochs  
→ Huber loss (summer) = 78.6 · (winter) = 231.9  
→ **Accurately predicts spike timing in winter** — critical for optimization

The LSTM model successfully captures temporal dependencies in emission patterns, enabling precise identification of low-emission charging windows.

### 2. Battery Modeling

**Model subject**: Tesla Model 3 Long Range (4416 cells · 96 series groups × 46 parallel)

**Approach**: PyBamm Single Particle Model calibrated to Tesla/Panasonic 2170 cell parameters.

**Power-SOC relationship** derived through convex fitting:

Charging (0–95% SOC):  
$$P(SOC) = 142.8 \log(SOC) + 8755.4$$

Charging (95–100% SOC):  
$$P(SOC) = -175107.3 \cdot SOC + 175107.3$$

Discharging:  
$$P(SOC) = 306.3 \log(SOC) + 8838.8$$

These expressions define the power limits used as constraints in the optimization problem.

### 3. Charging Session Data

**Dataset**: 6,844 real residential charging sessions · 96 users · Norway housing cooperative (Dec 2018 – Jan 2020)

**Session characteristics**:
- Average plug-in duration: 11.85 hours
- Average energy charged: 15.58 kWh
- Most frequent plug-in: 4pm · plug-out: 7am

This real-world data avoids the pitfalls of simulated driving behavior and provides actual constraints for optimization.

### 4. Optimization Formulation

**Objective**: Minimize total CO2 emissions from charging

$$\min_{P,I,SOC} \sum Emission(t) \times P(t)$$

**Constraints** (shift charging):

$$SOC(t+1) = SOC(t) + \frac{I(t)}{Q} \Delta t$$

$$0 \leq P(t) \leq P_{charging}(SOC) \leq P_{limit}$$

$$SOC(t=0) = SOC_{start} \quad \text{(from data)}$$

$$SOC(t=t_f) \geq SOC_{target} \quad \text{(from data)}$$

**Vehicle-to-Grid** extends the constraints to allow negative power (discharge):

$$P_{discharging}(SOC) \leq P(t) \leq P_{charging}(SOC)$$

Solved using **CVXPY** in Python.

---

## Results & Discussion

**Emission reduction**: Shift charging achieves 57% emission reduction. Vehicle-to-Grid goes further — 68% of sessions achieve net-negative emissions by discharging during high-emission periods and charging during clean windows.

**Grid load balancing**: 65% of peak-hour demand (5–10pm) can be shifted to idle periods during the day, significantly flattening the load curve.

**LSTM forecasting accuracy**: The LSTM model successfully predicts spike timing in winter — the most critical period. While spike magnitude prediction could be improved, timing accuracy is sufficient for effective charging policy.

**Limitations**:
- Norway charging data + California emission curve = geographic mismatch. Results are conceptually valid but numerical values are theoretical.
- Battery model excludes temperature and aging effects. Future work should incorporate these parameters for higher fidelity power limits.

---

## My Contribution

I developed and implemented the **LSTM forecasting model** for emission curves — the core component enabling optimized charging decisions.

**Technical work**:
- Data preprocessing and feature engineering (scaling with DART library, co-variate series for day/month)
- Implemented LSTM architecture from scratch
- Hyperparameter tuning and model evaluation
- Comparative analysis of FFNN vs LSTM performance

**Team collaboration**:
- Explained RNN/LSTM concepts to teammates unfamiliar with sequential models
- Documented model methodology and results for final report
- Coordinated integration of forecasting output with optimization pipeline

---

## Tech Stack

`Python` · `LSTM` · `PyTorch` · `CVXPY` · `PyBamm` · `WattTime API` · `DART` · `pandas` · `numpy` · `matplotlib`