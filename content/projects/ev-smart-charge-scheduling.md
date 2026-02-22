---
title: "EV Smart Charge Scheduling"
date: 2023-05-11
draft: false
categories: ["Projects"]
summary: "Intelligent electric vehicle charging optimization using LSTM emission forecasting and convex optimization to minimize CO2 emissions while ensuring user mobility needs."
cover:
  image: "/images/ev.png"
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

<a href="/reports/capstone_ep.pdf" target="_blank">**View full report (PDF) →**</a></br>
<a href="https://github.com/elouanXP/ev-smart-charge-scheduling" target="_blank">**View on GitHub →**</a>  

## 1- Overview

**Intelligent electric vehicle charging optimization using LSTM emission forecasting and convex optimization to minimize CO2 emissions while ensuring user mobility needs.**

![EV charging optimization](/images/ev_charging_header.png)

---

## 2- Context & Motivation

### Problem Statement

The rapid adoption of electric vehicles represents a critical strategy for addressing energy dependence and mitigating climate change. As EV adoption accelerates, a new challenge emerges: the growing strain on the energy grid. The current infrastructure is already stressed by rising energy demand, and this demand will only intensify as EV adoption becomes widespread.

The challenge manifests in two distinct charging infrastructure needs: public charging stations and residential home chargers. This project focuses exclusively on **home charging**, as public stations operate on a fast-charging model where drivers top up quickly, leaving minimal room for optimizing the power delivery curve.

**The core inefficiency:** Home charging typically occurs overnight when users plug in after returning home. This creates two critical problems:

1. **Peak grid demand** - simultaneous evening charging (5-10pm) creates a second daily peak, risking grid overload
2. **Maximum emissions** - nighttime charging coincides with the absence of solar energy, forcing reliance on high-carbon generation sources (800-1000 lbs CO2/MWh vs. 50 lbs during solar windows)

Not every vehicle requires 100% charge daily. The median commute requires only 20-30% SOC replenishment, yet standard practice charges to full capacity regardless of actual need.

### Technical Challenge

The need for dynamic charging policy stems from a fundamental mismatch between **user behavior** (plug in at night, charge to 100%) and **grid reality** (electricity is dirtiest at night, most users don't need full charge).

To prescribe an optimal charging policy, three interdependent challenges must be solved:

**1. Emission forecasting**  
Predict the Marginal Operating Emission Rate (MOER) - CO2 lbs/MWh at 5-minute intervals - for the next 24 hours. Challenge: unpredictable daily spikes in emission data. Summer patterns are consistent (reliable solar windows), but winter spikes occur randomly due to inconsistent weather (cloud cover, rainfall, snow).

**2. Battery modeling**  
Derive accurate power-SOC relationships that capture the battery's non-linear voltage behavior across the full charge range. The model must be convex (solvable by optimization algorithms) while remaining physically realistic.

**3. Real charging constraints**  
Honor actual user charging patterns from empirical data: plug-in/plug-out times, required final SOC, session duration. Simulated driving behavior fails to capture real-world variability.

### Project Goal

Create an optimized charging policy that:
- Charges opportunistically during low-emission windows
- Charges only to the SOC needed for the next trip (plus safety buffer)
- Reduces grid load during peak hours

This approach addresses both environmental impact (minimize emissions) and grid stability (flatten demand curve), while ensuring user mobility needs are met.

{{< rawhtml >}}
<!-- Project pipeline diagram -->
<div style="margin: 1.5rem 0;">
<svg viewBox="0 0 960 90" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:960px;display:block;margin:auto;font-family:monospace;">
  <!-- Boxes -->
  <rect x="5"   y="20" width="160" height="50" rx="6" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <rect x="200" y="20" width="160" height="50" rx="6" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <rect x="395" y="20" width="160" height="50" rx="6" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <rect x="590" y="20" width="160" height="50" rx="6" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.5"/>
  <rect x="785" y="20" width="170" height="50" rx="6" fill="#ede7f6" stroke="#ce93d8" stroke-width="1.5"/>
  <!-- Arrows -->
  <path d="M165 45 L198 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M360 45 L393 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M555 45 L588 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M750 45 L783 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- Arrow marker -->
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>
  <!-- Labels -->
  <text x="85"  y="42" text-anchor="middle" font-size="11" fill="#1565c0">MOER emission</text>
  <text x="85"  y="57" text-anchor="middle" font-size="11" fill="#1565c0">data (WattTime)</text>
  <text x="280" y="42" text-anchor="middle" font-size="11" fill="#2e7d32">LSTM emission</text>
  <text x="280" y="57" text-anchor="middle" font-size="11" fill="#2e7d32">forecasting</text>
  <text x="475" y="42" text-anchor="middle" font-size="11" fill="#e65100">Battery</text>
  <text x="475" y="57" text-anchor="middle" font-size="11" fill="#e65100">modeling (PyBamm)</text>
  <text x="670" y="42" text-anchor="middle" font-size="11" fill="#880e4f">Convex</text>
  <text x="670" y="57" text-anchor="middle" font-size="11" fill="#880e4f">optimization (CVXPY)</text>
  <text x="870" y="42" text-anchor="middle" font-size="11" fill="#4a148c">Charging policy</text>
  <text x="870" y="57" text-anchor="middle" font-size="11" fill="#4a148c">& validation</text>
</svg>
</div>
{{< /rawhtml >}}

---

## 3- Data Sources

This project integrates three datasets: emission time-series, battery electrochemistry simulations, and real residential charging sessions.

### Emission Curve Data

CAISO North Marginal Operating Emission Rate (MOER) from WattTime API

Time-series of CO2 lbs/MWh at 5-minute intervals for Northern California (1 full year). MOER represents the marginal emissions from the next unit of electricity generation.

### Battery Data

PyBamm Single Particle Model is used to simulate a Tesla Model 3 Long Range battery pack of 4,416 cells (96 series × 46 parallel):

| Parameter | Tesla cell | PyBamm Model |
|-----------|-----------|--------------|
| Capacity | 4.8 Ah | 4.73 Ah |
| Lower cutoff | 2.5 V | 2.6 V |
| Upper cutoff | 4.2 V | 4.1 V |

### Charging Session Data

Simulated driving surveys (NHTS) assume uniform behavior and neglect the high variability captured by real-world data (weekend vs weekday, irregular plug-in times). Real residential charging dataset from a Norway housing cooperative is therefore used:

6,844 sessions from 96 users (Dec 2018 – Jan 2020). Each session records plug-in/plug-out timestamp, energy charged (kWh) and user ID.

| Statistic | Value |
|-----------|-------|
| Avg plug-in duration | 11.85 hours |
| Avg energy charged | 15.58 kWh |
| Most frequent plug-in | 4pm |
| Most frequent plug-out | 7am |

![Energy charged distribution](/images/energy_charged_distribution.png)

---

## 4- Emission Forecasting

**Objective:** Predict 24-hour MOER curve (288 values at 5-min intervals) to identify low-emission charging windows.

The core challenge is to accurately forecast the timing of the spikes more than their magnitude. The optimizer is robust to magnitude errors but fragile to timing errors: a 4-hour shift misses the entire low-emission window.

{{< rawhtml >}}
<svg viewBox="0 0 820 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:820px;display:block;margin:1.2rem auto;font-family:monospace;">
  <defs>
    <marker id="a2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#888"/>
    </marker>
  </defs>
  <!-- Step boxes -->
  <rect x="10"  y="75" width="180" height="50" rx="6" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <rect x="235" y="75" width="180" height="50" rx="6" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <rect x="460" y="75" width="180" height="50" rx="6" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <rect x="685" y="75" width="125" height="50" rx="6" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.5"/>
  <!-- Arrows -->
  <path d="M190 100 L233 100" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>
  <path d="M415 100 L458 100" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>
  <path d="M640 100 L683 100" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>
  <!-- Box labels -->
  <text x="100" y="97"  text-anchor="middle" font-size="11" fill="#1565c0" font-weight="bold">1 year MOER data</text>
  <text x="100" y="113" text-anchor="middle" font-size="11" fill="#1565c0">5-min intervals</text>
  <text x="325" y="97"  text-anchor="middle" font-size="11" fill="#2e7d32" font-weight="bold">Data preprocessing</text>
  <text x="325" y="113" text-anchor="middle" font-size="11" fill="#2e7d32">DART scaling, covariates</text>
  <text x="550" y="97"  text-anchor="middle" font-size="11" fill="#e65100" font-weight="bold">Model training</text>
  <text x="550" y="113" text-anchor="middle" font-size="11" fill="#e65100">FFNN → LSTM</text>
  <text x="747" y="97"  text-anchor="middle" font-size="11" fill="#880e4f" font-weight="bold">24h forecast</text>
  <text x="747" y="113" text-anchor="middle" font-size="11" fill="#880e4f">288 values</text>
  <!-- Sub-labels below boxes -->
  <text x="100" y="145" text-anchor="middle" font-size="9.5" fill="#666">WattTime API</text>
  <text x="325" y="145" text-anchor="middle" font-size="9.5" fill="#666">day/month features</text>
  <text x="550" y="145" text-anchor="middle" font-size="9.5" fill="#666">25 days → 1 day</text>
  <text x="747" y="145" text-anchor="middle" font-size="9.5" fill="#666">spike timing</text>
</svg>
{{< /rawhtml >}}

### Loss Function: Huber Loss

Unlike other loss functions, the Huber loss function applies a smaller penalty to minor errors while imposing a linear penalty on substantial
errors. This unique characteristic makes it less prone to outliers and more capable of managing data with extreme values or sudden spikes.

{{< rawhtml >}}
$$
L_{\delta}(y, \hat{y}) = 
\begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$
{{< /rawhtml >}}

We used δ=1 in our two models.

### Feedforward Neural Network (FFNN)

As a baseline for our prediction, we use a classic FFNN with the following architecture:

- Input window: 1000 points (83 hours) captures weekly patterns
- Output: 288 points (24 hours) matches optimization horizon
- Depth: 8 hidden layers approximates complex non-linear emission dynamics, ReLU activation

![FFNN winter prediction — 4 hour timing error](/images/ffnn_winter_prediction.png)


**Summer Huber loss = 77.8** (consistent solar pattern) and **Winter Huber loss = 320.6** (spike timing error ±4 hours)

FFNN treats each 1000-point window independently, but winter weather evolves sequentially (cloud system persists 2-3 days for instance). We need a model with a  temporal memory.

### Long Short-Term Memory (LSTM)

Recurrent architecture maintains hidden state across timesteps, capturing sequential dependencies. A typical LSTM architecture is composed of 4 FFNN (3 gates, 1 candidate) and process the cell state and hidden state (long and short-term memories) from the previous LSTM cell:

![LSTM architecture](/images/lstm.png)

{{< rawhtml >}}
$$
\begin{align*}
F_t &= \sigma(W_f \cdot [H_{t-1}, X_t] + b_f) \quad \text{(forget gate)} \\
I_t &= \sigma(W_i \cdot [H_{t-1}, X_t] + b_i) \quad \text{(input gate)} \\
O_t &= \sigma(W_o \cdot [h_{t-1}, X_t] + b_o) \quad \text{(output gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [H_{t-1}, X_t] + b_C) \quad \text{(candidate values)} \\
C_t &= F_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(cell state, long-term memory)} \\
H_t &= O_t \odot \tanh(C_t) \quad \text{(hidden state, short-term output)}
\end{align*}
$$
{{< /rawhtml >}}

- Training data: 25 days to predict 1 full day
- Window size: 20 timesteps (100 minutes)
- Epochs: 10 (convergence validated on held-out validation set)
- Preprocessing: DART library scaling + day/month co-variates

![LSTM winter prediction — accurate spike timing](/images/Picture28.png)

**Summer: Huber loss = 78.6** (same as FFNN) and **Winter: Huber loss = 231.9** (**28% improvement**, spike timing ±30 min)

### Model Comparison

| Season | FFNN Huber Loss | LSTM Huber Loss | Spike Timing Error |
|--------|----------------|----------------|-------------------|
| Summer | 77.8 | 78.6 | <15 min (both) |
| Winter | 320.6 | 231.9 | FFNN ±4h / LSTM ±30min |

LSTM better captures when spikes occur even if magnitude is 2-3× off.

---

## 5- Battery Modeling

Battery terminal voltage is non-linear with SOC and differs between charging/discharging due to electrochemistry:

![battery](/images/soc_vs_voltage.png)

We generate high-granularity charging/discharging curves across 0-100% SOC to derive convex power-SOC relationships.

### Charging Power Limit

Battery voltage rises slowly 0-95% SOC (constant current), then rapidly 95-100% (constant voltage taper). Two-piece convex model:

{{< rawhtml >}}
$$
P_{\text{charge}}(SOC) = 
\begin{cases} 
142.8 \log(SOC) + 8755.4 & \text{for } SOC \in [0, 0.95] \\
-175107.3 \cdot SOC + 175107.3 & \text{for } SOC \in (0.95, 1]
\end{cases}
$$
{{< /rawhtml >}}

![Battery charging and discharging power limits](/images/charge_model.png)

### Discharging Power Limit

{{< rawhtml >}}
$$
P_{\text{discharge}}(SOC) = 306.3 \log(SOC) + 8838.8
$$
{{< /rawhtml >}}

![Battery charging and discharging power limits](/images/discharge_model.png)

---

## 6- Optimization 

Three charging strategies are compared across 6,844 real residential sessions:

1. Baseline: Plug in → charge to 100% → stop
2. Shift Charging: Charge opportunistically during low-emission windows → reach target SOC
3. Vehicle-to-Grid (V2G): Discharge during high-emission peaks + charge during clean windows

**Objective:** Minimize total emissions over the charging session

{{< rawhtml >}}
$$
\min_{P(t), I(t), SOC(t)} \sum_{t=0}^{T} \text{MOER}(t) \times P(t) \times \Delta t
$$
{{< /rawhtml >}}

where MOER(t) is the forecasted emission curve, P(t) charging power, Δt = 5 min.

### Constraints

SOC dynamics (state equation):
{{< rawhtml >}}
$$
SOC(t+1) = SOC(t) + \frac{I(t)}{Q} \Delta t
$$
{{< /rawhtml >}}

Power-current relationship (resistive losses):
{{< rawhtml >}}
$$
P(t) \geq V \cdot I(t) + C \cdot |I(t)|
$$
{{< /rawhtml >}}

Power limits (battery physics + charger rating):
{{< rawhtml >}}
$$
\begin{cases}
0 \leq P(t) \leq \min(P_{\text{charge}}(SOC), P_{\text{limit}}) & \text{(shift charging)} \\
P_{\text{discharge}}(SOC) \leq P(t) \leq P_{\text{charge}}(SOC) & \text{(V2G)}
\end{cases}
$$
{{< /rawhtml >}}

Boundary conditions (from charging session data):
{{< rawhtml >}}
$$
\begin{align*}
SOC(t=0) &= SOC_{\text{start}} \quad \text{(initial SOC from previous trip)} \\
SOC(t=T) &\geq SOC_{\text{target}} \quad \text{(required SOC for next trip)}
\end{align*}
$$
{{< /rawhtml >}}

### Parameters

| Symbol | Description | Value |
|--------|-------------|-------|
| $P_{\text{charge}}(SOC)$ | Max charging power | From battery model |
| $P_{\text{discharge}}(SOC)$ | Max discharging power | From battery model |
| $P_{\text{limit}}$ | Charger power rating | 9.6 kW (Level 2: 240V × 40A) |
| $V$ | Charging voltage | 240 V |
| $Q$ | Pack capacity | 250 Ah |
| $C$ | Loss coefficient | 36 (~15% round-trip efficiency) |
| $\Delta t$ | Time resolution | 5 min |
| $SOC_{\text{start}}, SOC_{\text{target}}$ | Initial/final SOC | From charging data |

This optimization problem is solved using CVXPY.

---

## 7- Results & Discussion

### Results

![results](/images/ave_co2_3.png)

| Strategy | Avg CO2 (lbs) | Reduction vs Baseline |
|----------|--------------|----------------------|
| **Baseline** | 7.5 | — |
| **Shift Charging** | 3.2 | **-57%** |
| **Vehicle-to-Grid** | -0.8 | **-111%** (net negative) |

The net-negative V2G result stems from avoided emissions, which are recorded as negative in the carbon balance.

### Limitations

- **LSTM winter predictions:** Huber Loss >100, accurately predicts spike timing but struggles with magnitude
- **Simplified battery model:** Limited variables, logarithmic SOC relationship only, excludes temperature and aging effects
- **Geographic mismatch:** Norway charging data substituted for unavailable California data so numerical CO2 values are theoretical, though the framework is conceptually valid

### Next Steps

- Explore additional variables to improve emission prediction accuracy
- Incorporate battery temperature and electrochemistry parameters for more accurate power limits
- Explore additional scenarios: electricity price curves, battery protection from low SOC
---
`Python` · `PyTorch` · `Darts` · `LSTM` · `CVXPY` · `PyBamm` · `pandas` · `numpy` · `matplotlib` · `WattTime API`