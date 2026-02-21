---
title: "COVID-19 Impact on Mobility in California"
date: 2023-05-09
draft: false
categories: ["Data Science", "Transportation", "Public Policy"]
summary: "A comprehensive research study analyzing 2.4M+ users' mobility patterns to inform California's 2030 GHG reduction policy — using machine learning, geospatial big data analysis and network science."
cover:
  image: ""
  alt: "COVID-19 and Mobility Analysis"
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

<a href="https://humnetlab.berkeley.edu/wp-content/uploads/2024/05/Final_Draft_CARB.pdf" target="_blank">**View full report (PDF) →**</a>

## 1. Overview

**A large-scale research study analyzing 2.4M+ California mobile phone users' mobility patterns to inform the state's 2030 greenhouse gas reduction policy — combining machine learning, geospatial big data analysis, and network science.**

![carb-overview.png](/images/carb-overview.png)

---

## 2. Context & Motivation

### Problem Statement

California has committed to reducing greenhouse gas emissions 40% below 1990 levels by 2030. Transportation is the single largest contributor to those emissions, and **Vehicle Miles Traveled (VMT)** is the primary lever. The COVID-19 pandemic created an unprecedented natural experiment: a sudden, statewide disruption to mobility that revealed which travel behaviors are habitual (and reversible) and which are structural (and permanent).

The **California Air Resources Board (CARB)**, in partnership with **UC Berkeley** and **Lawrence Berkeley National Laboratory**, commissioned this study to quantify those shifts and translate them into actionable policy guidance.

### Technical challenge

Understanding mobility at this scale requires solving three distinct problems simultaneously: inferring travel mode (car vs. walking) from raw GPS pings without any labeled training data, detecting residential relocations from nighttime location patterns, and measuring network-level changes in commute structure across ~8,000 census tracts over four years. Each requires a different algorithmic approach — unsupervised, semi-supervised, and graph-theoretic respectively.

### Project goal

Measure COVID-19's impact on VMT, commute patterns, and residential relocation across California at census-tract granularity using four years of anonymized LBS data, and derive targeted VMT reduction strategies by region and demographic group to inform CARB's 2030 climate roadmap.

{{< rawhtml >}}
<div style="margin: 1.5rem 0;">
<svg viewBox="0 0 860 90" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:860px;display:block;margin:auto;font-family:monospace;">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>
  <rect x="5"   y="20" width="145" height="50" rx="6" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <rect x="185" y="20" width="145" height="50" rx="6" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <rect x="365" y="20" width="145" height="50" rx="6" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <rect x="545" y="20" width="145" height="50" rx="6" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.5"/>
  <rect x="725" y="20" width="130" height="50" rx="6" fill="#ede7f6" stroke="#ce93d8" stroke-width="1.5"/>
  <path d="M150 45 L183 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M330 45 L363 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M510 45 L543 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M690 45 L723 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="77"  y="42" text-anchor="middle" font-size="11" fill="#1565c0">LBS data</text>
  <text x="77"  y="57" text-anchor="middle" font-size="11" fill="#1565c0">2.4M+ users</text>
  <text x="257" y="42" text-anchor="middle" font-size="11" fill="#2e7d32">Mode detection</text>
  <text x="257" y="57" text-anchor="middle" font-size="11" fill="#2e7d32">GMM</text>
  <text x="437" y="42" text-anchor="middle" font-size="11" fill="#e65100">Relocation</text>
  <text x="437" y="57" text-anchor="middle" font-size="11" fill="#e65100">KMeans-SVM</text>
  <text x="617" y="42" text-anchor="middle" font-size="11" fill="#880e4f">Commute</text>
  <text x="617" y="57" text-anchor="middle" font-size="11" fill="#880e4f">network analysis</text>
  <text x="790" y="42" text-anchor="middle" font-size="11" fill="#4a148c">Policy</text>
  <text x="790" y="57" text-anchor="middle" font-size="11" fill="#4a148c">recommendations</text>
</svg>
</div>
{{< /rawhtml >}}

---

## 3. Dataset

**Location-Based Service (LBS) data** sourced from Spectus — anonymized, opt-in, CCPA/GDPR-compliant mobile phone GPS pings collected across California from 2019 to 2022.

| Year | Total users | High-quality users |
|------|------------:|-------------------:|
| 2019 | 9.4M | 3,482,574 |
| 2020 | 5.9M | 2,396,990 |
| 2021 | 5.2M | 2,036,110 |
| 2022 | 5.6M | 2,582,405 |

**Spatial resolution:** Census block group (~600–3,000 people). **Quality filters** applied to retain only users with sufficient coverage: >316 pings, >60 active days, >10 visits to both home and work locations.

**Home and work location detection** uses a heuristic based on the most frequently visited location during the appropriate time window:

```python
home = most_frequent_location(user, time_range="7pm-7am", min_visits=10)
work = most_frequent_location(user, time_range="7am-7pm", weekdays_only=True, min_visits=10)
```

Validation against 2019 Census tract population yields a Pearson correlation of r=0.5 — satisfactory for anonymized opt-in data.

---

## 4. Methodology

Three distinct algorithmic problems, each requiring a different approach.

### 4.1 Algorithm 1 — Mode Detection via GMM

**Challenge.** No labeled trip data is available for supervised learning — trips must be classified as motorized, non-motorized, or noise without any ground truth.

**Solution: Gaussian Mixture Model (unsupervised).** Each trip is represented by two features: `log(max_speed_kph)` and `log(trajectory_length_km)`. A GMM with k=3 components (selected via the elbow method on AIC/BIC) fits three Gaussian distributions in this 2D feature space, producing soft cluster assignments:

$$\text{GMM}(X) = \sum_{k=1}^{3} \pi_k \,\mathcal{N}(X \mid \mu_k, \Sigma_k)$$

{{< rawhtml >}}
<svg viewBox="0 0 680 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;display:block;margin:1.2rem auto;font-family:monospace;">
  <defs>
    <marker id="gm" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#888"/>
    </marker>
  </defs>
  <!-- Input -->
  <rect x="5" y="40" width="130" height="50" rx="5" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <text x="70" y="62" text-anchor="middle" font-size="10" fill="#1565c0" font-weight="bold">Trip features</text>
  <text x="70" y="76" text-anchor="middle" font-size="9.5" fill="#1565c0">log(max_speed)</text>
  <text x="70" y="88" text-anchor="middle" font-size="9.5" fill="#1565c0">log(length_km)</text>
  <!-- GMM -->
  <rect x="175" y="30" width="130" height="70" rx="5" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <text x="240" y="58" text-anchor="middle" font-size="11" fill="#e65100" font-weight="bold">GMM k=3</text>
  <text x="240" y="73" text-anchor="middle" font-size="9.5" fill="#bf360c">AIC/BIC elbow</text>
  <text x="240" y="87" text-anchor="middle" font-size="9.5" fill="#bf360c">soft assignment</text>
  <!-- Arrow -->
  <path d="M135 65 L173 65" stroke="#888" stroke-width="1.2" marker-end="url(#gm)"/>
  <path d="M305 50 L343 35" stroke="#888" stroke-width="1.2" marker-end="url(#gm)"/>
  <path d="M305 65 L343 65" stroke="#888" stroke-width="1.2" marker-end="url(#gm)"/>
  <path d="M305 80 L343 95" stroke="#888" stroke-width="1.2" marker-end="url(#gm)"/>
  <!-- Clusters -->
  <rect x="345" y="15" width="145" height="35" rx="4" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.2"/>
  <rect x="345" y="55" width="145" height="35" rx="4" fill="#fff9c4" stroke="#f9a825" stroke-width="1.2"/>
  <rect x="345" y="80" width="145" height="30" rx="4" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.2"/>
  <text x="418" y="36" text-anchor="middle" font-size="10" fill="#2e7d32" font-weight="bold">Motorized  &gt;40 kph</text>
  <text x="418" y="75" text-anchor="middle" font-size="10" fill="#f57f17" font-weight="bold">Non-motorized  &lt;6.3 kph</text>
  <text x="418" y="97" text-anchor="middle" font-size="10" fill="#880e4f" font-weight="bold">Noise (ambiguous)</text>
  <!-- Advantage -->
  <text x="345" y="125" font-size="9" fill="#555">→ Scales to billions of trips · no manual labeling</text>
</svg>
{{< /rawhtml >}}

---

### 4.2 Algorithm 2 — Home Change Detection via KMeans-SVM

**Challenge.** Detect intra-state residential moves (≥5 miles) from nighttime GPS patterns, without any labeled relocation data.

**Two-step semi-supervised approach:**

{{< rawhtml >}}
<svg viewBox="0 0 700 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;display:block;margin:1.2rem auto;font-family:monospace;">
  <defs>
    <marker id="km" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#888"/>
    </marker>
  </defs>
  <!-- Step 1 -->
  <rect x="5" y="20" width="200" height="70" rx="5" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <text x="105" y="42" text-anchor="middle" font-size="11" fill="#1565c0" font-weight="bold">Step 1 — K-Means</text>
  <text x="105" y="57" text-anchor="middle" font-size="9.5" fill="#1565c0">(unsupervised)</text>
  <text x="105" y="72" text-anchor="middle" font-size="9.5" fill="#1565c0">lat, lng, days_since_jan1</text>
  <text x="105" y="84" text-anchor="middle" font-size="9.5" fill="#1565c0">→ 2 clusters: before / after</text>
  <!-- Arrow -->
  <path d="M205 55 L248 55" stroke="#888" stroke-width="1.2" marker-end="url(#km)"/>
  <text x="226" y="49" text-anchor="middle" font-size="8.5" fill="#888">pseudo-labels</text>
  <!-- Step 2 -->
  <rect x="250" y="20" width="200" height="70" rx="5" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <text x="350" y="42" text-anchor="middle" font-size="11" fill="#2e7d32" font-weight="bold">Step 2 — SVM</text>
  <text x="350" y="57" text-anchor="middle" font-size="9.5" fill="#2e7d32">(supervised, c=0.01)</text>
  <text x="350" y="72" text-anchor="middle" font-size="9.5" fill="#2e7d32">wide margin — prevents</text>
  <text x="350" y="84" text-anchor="middle" font-size="9.5" fill="#2e7d32">geographic overfitting</text>
  <!-- Arrow -->
  <path d="M450 55 L493 55" stroke="#888" stroke-width="1.2" marker-end="url(#km)"/>
  <!-- Output -->
  <rect x="495" y="20" width="200" height="70" rx="5" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <text x="595" y="42" text-anchor="middle" font-size="11" fill="#e65100" font-weight="bold">Move detected</text>
  <text x="595" y="57" text-anchor="middle" font-size="9.5" fill="#e65100">dist ≥ 5 miles</text>
  <text x="595" y="72" text-anchor="middle" font-size="9.5" fill="#e65100">≥ 20 obs / cluster</text>
  <text x="595" y="84" text-anchor="middle" font-size="9.5" fill="#e65100">MoveDate computed</text>
  <!-- Performance -->
  <rect x="5" y="110" width="690" height="40" rx="4" fill="#f5f5f5" stroke="#e0e0e0" stroke-width="1"/>
  <text x="15" y="127" font-size="9.5" fill="#333" font-weight="bold">Performance (synthetic dataset):</text>
  <text x="15" y="142" font-size="9.5" fill="#555">Accuracy: 86.4% (4,321 / 5,000 true positives) · Type I Error: 1.6% (447 / 28,260 false positives)</text>
</svg>
{{< /rawhtml >}}

The move date is estimated as:
$$\text{MoveDate} = \min\bigl(\max(C_{\text{before}}),\ \max(C_{\text{after}})\bigr)$$

---

### 4.3 Algorithm 3 — Radius of Gyration

The **radius of gyration** $r_g$ measures the spatial spread of a user's activity — a single scalar summary of how far from home a person typically travels:

$$r_g(u) = \sqrt{\frac{1}{n_u} \sum_{i=1}^{n_u} \text{dist}\!\left(r_i(u) - r_{cm}(u)\right)^2}$$

where $r_{cm}(u)$ is the center of mass of all visited locations. High $r_g$ signals long-distance, vehicle-dependent travel; low $r_g$ signals localized, walkable mobility. Tracking $r_g$ before and after lockdown quantifies not just whether people traveled less, but whether they traveled shorter distances.

---

### 4.4 Commute Flow Network Analysis

Commute patterns are modeled as a directed weighted graph: **nodes** are census tracts (~8,000), **edges** are weighted by observed home→work trip flows. **Louvain community detection** is applied to identify natural commute catchment areas by maximizing modularity:

$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Higher modularity means the network splits more easily into self-contained regions — a signature of reduced inter-regional commuting. Tracking $Q$ and the number of communities year-over-year reveals whether the pandemic fragmented California's commute structure permanently.

---

## 5. Results

### VMT reduction — regional disparities

Urban counties (LA, Orange, Santa Barbara) saw **38–55% VMT reductions** in April 2020; rural counties (Imperial, Kern) only **20–30%**. The 20-percentage-point gap reflects fundamentally different vehicle dependency: urban residents could substitute car trips for walking or remote work, rural residents largely could not. A positive correlation (r=0.376) between county area and VMT persistence confirms this vehicle dependency is structural, not behavioral.

Weekend VMT dropped more sharply than weekday VMT — consistent with non-essential travel (leisure, errands) being more elastic than work commutes.

---

### Trip purpose — commute recovery lagged permanently

| Metric | Commute trips | Non-commute trips |
|--------|:-------------:|:-----------------:|
| Recovery by 2022 | Below 2019 baseline | Near 2019 baseline |
| Work-to-non-work ratio | Never recovered | — |

Non-commute travel (shopping, leisure) recovered to near-baseline levels by 2022. Commute trips did not — the work-to-non-work ratio never returned to its pre-pandemic level. Remote and hybrid work restructured not just where people work, but the fundamental shape of California's daily travel demand.

---

### Commute network fragmentation

| Year | Edges | Communities | Modularity |
|------|------:|:-----------:|:----------:|
| 2019 | 145,838 | 6 | 0.628 |
| 2020 | 89,382 | 8 ↑ | 0.653 ↑ |
| 2021 | 73,401 | 8 | 0.648 |
| 2022 | 111,652 | 8 | **0.666** ↑ |

The network lost **38.7% of its edges** in 2020. Two new isolated communities emerged in remote regions (El Centro, eastern Sierras). Crucially, **modularity remained elevated through 2022** — even as some edges recovered, the network never returned to its pre-pandemic integrated structure. Major urban centers (SF, LA, San Diego) continue to act as commute gravity wells over their surrounding regions.

---

### Residential relocation

A sharp spike in moves occurred in the 15-day window between the State of Emergency (March 4) and the Shelter-in-Place order (March 19, 2020). The distance distribution during this crisis window was **bimodal**: one mode at ~50 miles (local reshuffling within metro areas) and one at ~350 miles (inter-regional, roughly Bay Area ↔ Southern California).

Lower-income census tracts showed longer-distance moves on average — consistent with housing instability forcing displacement rather than voluntary relocation. Net outflows from SF and LA; stable populations in the Central Valley (Fresno, Bakersfield).

---

### Radius of gyration

Urban cores (Bay Area, LA, San Diego) consistently show $r_g$ ≈ 20 km — compact travel patterns even pre-pandemic. Post-lockdown, urban $r_g$ dropped sharply. Rural areas showed less reduction: residents maintained long-distance travel patterns because alternatives (transit, active mobility) were unavailable.

---

### Mobility intervention assessment (Sacramento 2019)

| Intervention | Date | Result |
|-------------|------|--------|
| JUMP e-scooter launch | Feb 2019 | No significant change |
| GIG Car Share launch | Mar 2019 | No significant change |
| **JUMP fleet expansion** | **Jun 2019** | **✓ Significant vehicle reduction** |
| SacRT Forward transit | Sep 2019 | No significant change |

Bootstrap sampling (100 iterations, n=50,000 trips) on month-over-month vehicle trip percentages. The key finding: **program launch has no measurable effect; scaling does**. Reaching a critical mass of deployed vehicles is required before modal substitution occurs.

---

## 6. Impact & Policy Recommendations

The results translate into four concrete policy levers for CARB's 2030 strategy:

**Urban areas** can leverage the structural reduction in vehicle dependency post-COVID — zoning reforms near transit, reduced parking minimums, and suburban infill development would lock in these gains rather than allowing car use to creep back.

**Rural areas** present the harder problem: vehicle dependency persisted precisely where alternatives are absent. EV charging infrastructure and carpool lanes address the emissions dimension without requiring behavioral change.

**Remote work** is now structurally embedded in California's labor market. Land use policy should adapt — not resist — by encouraging residential development near suburban employment centers and reducing commute-driven VMT.

**Micro-mobility at scale** (not at pilot) is the only intervention category shown to move the needle on modal shift. Policy should prioritize rapid deployment density over geographic coverage breadth.

**Validated outcomes:** LBS data confirmed as a viable, cost-effective alternative to expensive travel surveys; reusable open-source algorithms for mode detection and relocation analysis without labeled training data; CARB 2030 roadmap informed with census-tract-level VMT reduction targets.

---
`Python` · `pandas` · `numpy` · `scikit-learn` · `networkx` · `geopandas` · `matplotlib`  