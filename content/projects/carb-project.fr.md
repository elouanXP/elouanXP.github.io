---
title: "COVID-19 Mobility Analysis: VMT by Mode and Purpose"
date: 2023-05-15
draft: false
categories: ["Data Science", "Transportation", "Public Policy"]
tags: ["Python", "Machine Learning", "Network Analysis", "Geospatial Analysis", "Unsupervised Learning"]
summary: "Developed machine learning algorithms to analyze 4 years of mobile phone location data (2.4-3.5M users), measuring how COVID-19 transformed travel behavior across California. Built novel unsupervised mode detection and home change detection models to track VMT changes and inform state climate policy."
cover:
  image: ""
  alt: "COVID-19 and Mobility Analysis"
  relative: false
showToc: true
TocOpen: false
hidemeta: false
comments: false
disableShare: true
ShowReadingTime: true
ShowBreadCrumbs: true
math: true
---

## Project Overview

This research project, conducted for the **California Air Resources Board (CARB)** in partnership with **UC Berkeley** and **Lawrence Berkeley National Laboratory**, investigates how the COVID-19 pandemic fundamentally altered mobility patterns across California. Using large-scale anonymized location data from 2.4-3.5 million mobile phone users, we measured changes in Vehicle Miles Traveled (VMT) by mode and purpose to inform California's 2030 greenhouse gas emission reduction goals (40% below 1990 levels).

**Organization:** UC Berkeley & Lawrence Berkeley National Laboratory  
**Client:** California Air Resources Board (CARB)  
**Grant:** CARB Agreement No. 20RD005  
**Duration:** 2019-2023  
**Team Size:** 7 researchers  
**Principal Investigator:** Prof. Marta C. González

## My Role & Contributions

As a core research team member, I developed novel machine learning algorithms and conducted large-scale data analysis on 4 years of Location-Based Service (LBS) data. My primary contributions included:

### 1. **Data Pipeline & Quality Control**
- Designed and implemented user selection criteria for 2.4-3.5M active users per year
- Processed billions of GPS pings into trajectory format at census block group resolution
- Developed home/work detection algorithms with 0.5 correlation to census population data

### 2. **Novel Algorithm Development**
- **Unsupervised Mode Detection (GMM)**: Created Gaussian Mixture Model to classify trips as motorized, non-motorized, or noise using only speed and distance features—no labeled data required
- **Semi-Supervised Home Change Detection (KMeans-SVM)**: Built two-step algorithm achieving 86.4% accuracy on synthetic data (1.6% Type I error) for detecting residential relocations

### 3. **Mobility Analysis**
- Analyzed statewide VMT changes across 58 California counties
- Computed Radius of Gyration (r_g) metrics to measure travel distance changes
- Built commute flow networks and applied Louvain community detection to identify regional fragmentation

### 4. **Policy Impact Assessment**
- Evaluated 4 transportation interventions in Sacramento using bootstrap sampling
- Detected significant vehicle usage reduction from e-scooter fleet expansion

## Technical Approach

### Data Sources & Processing

**Location-Based Service (LBS) Data (Spectus)**
- **Volume:** 4 years (2019-2022), 9.4M → 5.9M → 5.2M → 5.6M total users
- **Spatial Resolution:** Census block group (~600-3,000 people)
- **Privacy:** Anonymized, opt-in, CCPA/GDPR-compliant
- **Quality Filters:** 
  - \> $10^{2.5}$ (316) pings per user
  - \> 60 days active timespan
  - \> 10 visits to both home and work locations

**Home & Work Detection**
```python
# Heuristic-based detection
home = most_frequent_location(user, time_range="7pm-7am", min_visits=10)
work = most_frequent_location(user, time_range="7am-7pm", weekdays_only=True, min_visits=10)
```
**Validation:** Pearson correlation r=0.5 with 2019 Census tract population

### Algorithm 1: Mode Detection via GMM

**Challenge:** No labeled trip data available for supervised learning

**Solution:** Gaussian Mixture Model (unsupervised clustering)
- **Features:** 
  - `log(max_speed_kph)` 
  - `log(trajectory_length_km)`
- **Clusters:** 3 (motorized, non-motorized, noise)
- **Parameter Selection:** Elbow method on AIC/BIC → optimal k=3

**Logic:**
- Motorized: max speed > $10^{1.6}$ (40 kph / 25 mph)
- Non-motorized: max speed < $10^{0.8}$ (6.3 kph)
- Noise: Ambiguous trips

$$
\text{GMM}(X) = \sum_{k=1}^{3} \pi_k \mathcal{N}(X | \mu_k, \Sigma_k)
$$

**Advantage:** Scales to billions of trips without manual labeling

---

### Algorithm 2: Home Change Detection via KMeans-SVM

**Challenge:** Detect intra-state moves (≥5 miles) at census tract level

**Two-Step Semi-Supervised Approach:**

**Step 1: K-Means Clustering (Unsupervised)**
- **Input:** All nighttime trajectories (7pm-7am) with features:
  - `latitude`, `longitude`, `days_since_jan1`
- **Output:** 2 clusters representing "before move" and "after move" behaviors

**Step 2: SVM Classification (Supervised with Pseudo-Labels)**
- **Input:** K-means cluster labels as pseudo-labels
- **Purpose:** Refine boundaries to prevent geographic overfitting
- **Parameter:** Low penalty c=0.01 (wide margin)

**Move Date Calculation:**
$$
\text{MoveDate} = \min(\max(C_{\text{before}}), \max(C_{\text{after}}))
$$

**Validation Criteria:**
- Distance between old/new homes ≥ 5 miles (8 km)
- ≥ 20 observations in each cluster

**Performance (Synthetic Dataset):**
- Accuracy: 86.4% (4,321/5,000 true positives)
- Type I Error: 1.6% (447/28,260 false positives)

---

### Algorithm 3: Radius of Gyration (r_g)

Measures the spatial spread of user activity:

$$
r_g(u) = \sqrt{\frac{1}{n_u} \sum_{i=1}^{n_u} \text{dist}(r_i(u) - r_{cm}(u))^2}
$$

where $r_{cm}(u)$ is the center of mass of all user locations.

- **Higher r_g:** More long-distance travel (rural areas, high vehicle use)
- **Lower r_g:** Localized travel (urban areas, walkable neighborhoods)

---

### Network Analysis: Commute Flow Networks

**Graph Construction:**
- **Nodes:** Census tracts (n ≈ 8,000)
- **Edges:** Weighted by commute flow (home → work trips)
- **Method:** Louvain community detection (maximizes modularity)

**Modularity Formula:**
$$
Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

**Interpretation:**
- **Higher modularity:** Network easily splits into isolated regions
- **More communities:** Reduced inter-regional commuting

## Key Results

### 1. Regional Disparities in VMT Reduction

| Region Type | VMT Reduction (April 2020) |
|-------------|---------------------------|
| **Urban counties** (LA, Orange, Santa Barbara) | **38-55%** ↓ |
| **Rural counties** (Imperial, Kern) | **20-30%** ↓ |

- Urban areas showed 20% greater VMT reduction than rural areas
- Positive correlation (r=0.376) between county area and persistence of high VMT
- Weekend VMT dropped more than weekdays (non-essential travel)

---

### 2. Differential Recovery by Trip Purpose

| Metric | Commute Trips | Non-Commute Trips |
|--------|--------------|------------------|
| Recovery rate | **Slower** | **Faster** |
| 2022 status | Below 2019 baseline | Near 2019 baseline |
| Work-to-non-work ratio | **Never recovered** to pre-pandemic levels | — |

**Interpretation:** Remote work had a **lasting structural impact** on travel behavior

---

### 3. Commute Network Fragmentation

| Year | Nodes | Edges | Communities | Modularity |
|------|-------|-------|-------------|-----------|
| **2019** | 8,033 | **145,838** | **6** | 0.628 |
| **2020** | 8,026 | **89,382** | **8** ↑ | 0.653 ↑ |
| **2021** | 8,009 | **73,401** | **8** | 0.648 |
| **2022** | 8,019 | **111,652** | **8** | **0.666** ↑ |

**Key Findings:**
- **38.7% reduction** in network edges (2019 → 2020)
- **2 new isolated communities** emerged in remote regions (El Centro, eastern Sierras)
- **Modularity remained elevated in 2022:** Commute network did not return to pre-pandemic structure

**Spatial Patterns:**
- Bay Area dominance over surrounding regions (even Sac<ramento/Bakersfield catchment areas)
- Major urban centers (SF, LA, San Diego) act as commute "gravity wells"

---

### 4. Residential Relocation Spike

**Temporal Pattern:**
- **Sharp spike:** March 4-19, 2020 (State of Emergency → SIP order)
- Overall 2020 moves: **5.25%** of users (vs. 5.72% in 2019)

**Move Distance Distribution:**
- **Bimodal distribution** during crisis period:
  - Mode 1: Local moves (~50 miles)
  - Mode 2: Inter-regional moves (~313-394 miles / 500-630 km)
    - Equivalent to **Bay Area ↔ Southern California**

**Socioeconomic Patterns:**
- **Lower-income census tracts:** Longer-distance moves during crisis
- **Urban exodus:** Net outflow from SF and LA; stable populations in Central Valley (Fresno, Bakersfield)

**Annual Comparison:**

| Year | # High-Quality Users | # Home Changes | % Moved |
|------|---------------------|----------------|---------|
| 2019 | 3,482,574 | 199,286 | 5.72% |
| **2020** | 2,396,990 | 125,929 | **5.25%** |
| 2021 | 2,036,110 | 118,446 | 5.82% |
| 2022 | 2,582,405 | 164,927 | 6.39% |

---

### 5. Radius of Gyration Changes

**Statewide Patterns:**
- **Urban cores** (Bay Area, LA, San Diego): r_g ≈ 20 km (consistent)
- **Rural areas:** Higher variation, less reduction in travel distance
- **Correlation:** Larger counties (rural) → smaller r_g reduction

**Post-SIP Impact:**
- People traveled **shorter distances**, not just less frequently
- Rural residents maintained long-distance travel patterns (vehicle dependency)

---

### 6. Mobility Intervention Assessment (Sacramento 2019)

**Tested 4 Transportation Initiatives:**

| Event | Date | Hypothesis | Result |
|-------|------|-----------|--------|
| JUMP e-scooter launch | Feb 2019 | ↓ Vehicle usage | No significant change |
| GIG Car Share launch | Mar 2019 | ↑ Vehicle usage | No significant change |
| **JUMP fleet expansion** | **Jun 2019** | **↓ Vehicle usage** | **✓ Significant decrease** |
| SacRT Forward transit | Sep 2019 | ↑ Vehicle usage | No significant change |

**Method:** Bootstrap sampling (100 iterations, n=50,000 trips) comparing vehicle trip percentage month-over-month

**Key Finding:** Only **scaling** of micro-mobility (not launch) reduced vehicle usage

## Technical Stack & Skills

**Languages & Libraries:**
- **Python:** pandas, scikit-learn, networkx, geopandas, matplotlib, numpy
- **Machine Learning:** Gaussian Mixture Models, Support Vector Machines, K-Means clustering
- **Geospatial:** Census API, GeoPandas, spatial joins, coordinate systems
- **Network Analysis:** Louvain community detection, modularity optimization, directed weighted graphs
- **Statistical Methods:** Bootstrap sampling, panel regression, correlation analysis

**Domain Expertise:**
- Transportation modeling & travel behavior analysis
- Human mobility science & spatial-temporal data
- Privacy-preserving data analysis (CCPA/GDPR compliance)
- Policy impact evaluation & causal inference

**Data Scale:**
- **Volume:** Billions of GPS pings, millions of users, 8,000 census tracts
- **Duration:** 4-year longitudinal study
- **Processing:** Distributed computing for trajectory construction

## Impact & Policy Recommendations

### For California Policymakers

1. **Targeted VMT Reduction Strategies**
   - **Urban areas:** Leverage reduced vehicle dependency post-COVID through zoning reforms and transit investment
   - **Rural areas:** Vehicle dependency persisted—require infrastructure investment (EV charging, carpool lanes)

2. **Remote Work Integration**
   - Lasting commute pattern changes suggest **hybrid work is permanent**
   - Recommendation: Update land use policies, reduce parking minimums near transit, incentivize suburban infill development

3. **Transit & Active Mobility**
   - E-scooter/bike programs reduce vehicle usage **only when scaled** (not pilot launches)
   - Focus on "last-mile" connectivity in suburban areas

4. **Demographic Considerations**
   - Lower-income populations more likely to relocate during crises (housing instability)
   - Need affordable housing policy near stable employment centers

### Academic Contributions

- **Novel Algorithms:** Unsupervised mode detection and semi-supervised home change detection overcome labeled data scarcity
- **Validation Framework:** Synthetic data generation for algorithm testing
- **Methodological Innovation:** Combining network science with mobility data for policy insights

## Deliverables

- **Final Research Report:** 67-page technical document to CARB
- **Interactive Web Application:** Radius of Gyration visualization tool (county/week level)
- **White Paper:** "Human Mobility Data in the 21st Century" (Appendix I, 47 pages)
- **Presentations:** Research seminars at UC Berkeley and CARB stakeholder meetings

## Project Outcomes

✅ **Validated LBS data** as viable alternative to expensive travel surveys  
✅ **Quantified regional disparities** in COVID-19 mobility impacts for targeted policy  
✅ **Developed reusable algorithms** for mode/relocation detection without labeled data  
✅ **Informed CARB's 2030 climate strategy** with data-driven VMT reduction pathways  
✅ **Demonstrated micro-mobility scaling** as effective VMT reduction intervention

---

**Full Report:** [California Air Resources Board Website](https://ww2.arb.ca.gov/our-work/programs/research-division)  
**Related Publications:** In preparation for *Transportation Research Part D: Transport and Environment*

---

*This project demonstrates advanced machine learning, geospatial analysis, and network science applied to real-world transportation policy during an unprecedented global crisis. By developing novel unsupervised algorithms, we overcame data limitations to provide actionable insights for California's climate goals.*