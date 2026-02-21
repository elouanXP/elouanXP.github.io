---
title: "Optimisation de la recharge de véhicules électriques"
date: 2023-05-11
draft: false
categories: ["Projects"]
summary: "Optimisation de la recharge de véhicules électriques à l'aide d'un réseau LSTM de prédiction des émissions et optimisation convexe de l'empreinte CO2, tout en garantissant les besoins de mobilité."
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

<a href="/reports/capstone_ep.pdf" target="_blank">**Voir le rapport complet (PDF) →**</a></br>
<a href="https://github.com/elouanXP/ev-smart-charge-scheduling" target="_blank">**Voir sur GitHub →**</a>

## 1- Vue d'ensemble

**Optimisation intelligente de la recharge de véhicules électriques par prévision LSTM des émissions et optimisation convexe pour minimiser l'empreinte CO2 tout en garantissant les besoins de mobilité.**

![EV charging optimization](/images/ev_charging_header.png)

---

## 2- Contexte & Motivation

### Problématique

L'adoption rapide des véhicules électriques constitue une stratégie essentielle pour réduire la dépendance énergétique et lutter contre le changement climatique. Mais à mesure que cette adoption s'accélère, un nouveau défi émerge : la pression croissante sur le réseau électrique. L'infrastructure actuelle est déjà sollicitée par une demande en hausse, et cette pression ne fera que s'intensifier avec la généralisation des VE.

Ce projet se concentre exclusivement sur la **recharge à domicile**, car les bornes publiques fonctionnent sur un modèle de recharge rapide où les conducteurs font le plein en peu de temps, laissant peu de marge pour optimiser la courbe de puissance.

**La problématique :** la recharge à domicile se produit typiquement en soirée, au retour du travail. Cela crée deux problèmes critiques :

1. **Pic de demande réseau**: la recharge simultanée en soirée (17h–22h) crée un second pic journalier, risquant la surcharge du réseau
2. **Émissions maximales**: la recharge nocturne coïncide avec l'absence d'énergie solaire, contraignant à recourir aux sources de production les plus carbonées (800-1 000 lbs CO2/MWh contre 50 lbs en journée)

Par ailleurs, tous les véhicules n'ont pas besoin d'une charge à 100% chaque jour. Le trajet médian ne nécessite qu'un rechargement de 20 à 30% de SOC (State Of Charge), pourtant la pratique standard consiste à systématiquement à pleine capacité.

### Défi technique

La nécessité d'une politique de recharge dynamique découle d'une inadéquation fondamentale entre le comportement des utilisateurs (brancher le soir, charger à 100%) et la réalité du réseau (l'électricité est la plus carbonée la nuit, et la plupart des utilisateurs n'ont pas besoin d'une charge complète).

Pour définir une politique de recharge optimale, trois défis interdépendants doivent être résolus :

**1. Prévision des émissions**
Prédire le Marginal Operating Emission Rate (MOER) - CO2 lbs/MWh à intervalles de 5 minutes - pour les 24 heures suivantes. Défi : pics journaliers imprévisibles dans les données. Les patterns estivaux sont réguliers (fenêtres solaires fiables), mais les pics hivernaux surviennent aléatoirement en raison de conditions météo instables (couverture nuageuse, pluie, neige).

**2. Modélisation de la batterie**
La modélisation dérive des relations puissance-SOC précises qui capturent le comportement non linéaire de la tension de la batterie sur toute la plage de charge. Le modèle doit être convexe (résolvable par les algorithmes d'optimisation) tout en restant physiquement réaliste.

**3. Contraintes de recharge réelles**
Respecter les patterns de recharge réels des utilisateurs issus de données empiriques : horaires de branchement/débranchement, SOC final requis, durée de session. Les simulations des usages de véhicules ne parviennent pas à capturer la variabilité du monde réel.

### Objectif du projet

Créer une politique de recharge optimisée qui :
- Recharge de manière opportuniste pendant les fenêtres à faibles émissions
- Recharge uniquement jusqu'au SOC nécessaire pour le prochain trajet (plus une marge de sécurité)
- Réduit la charge réseau aux heures de pointe

Cette approche répond à la fois à l'impact environnemental (minimiser les émissions) et à la stabilité du réseau (aplatir la courbe de demande), tout en garantissant la mobilité des utilisateurs.

{{< rawhtml >}}
<div style="margin: 1.5rem 0;">
<svg viewBox="0 0 960 90" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:960px;display:block;margin:auto;font-family:monospace;">
  <rect x="5"   y="20" width="160" height="50" rx="6" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <rect x="200" y="20" width="160" height="50" rx="6" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <rect x="395" y="20" width="160" height="50" rx="6" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <rect x="590" y="20" width="160" height="50" rx="6" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.5"/>
  <rect x="785" y="20" width="170" height="50" rx="6" fill="#ede7f6" stroke="#ce93d8" stroke-width="1.5"/>
  <path d="M165 45 L198 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M360 45 L393 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M555 45 L588 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M750 45 L783 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>
  <text x="85"  y="42" text-anchor="middle" font-size="11" fill="#1565c0">Données MOER</text>
  <text x="85"  y="57" text-anchor="middle" font-size="11" fill="#1565c0">(WattTime)</text>
  <text x="280" y="42" text-anchor="middle" font-size="11" fill="#2e7d32">Prévision LSTM</text>
  <text x="280" y="57" text-anchor="middle" font-size="11" fill="#2e7d32">des émissions</text>
  <text x="475" y="42" text-anchor="middle" font-size="11" fill="#e65100">Modélisation</text>
  <text x="475" y="57" text-anchor="middle" font-size="11" fill="#e65100">batterie (PyBamm)</text>
  <text x="670" y="42" text-anchor="middle" font-size="11" fill="#880e4f">Optimisation</text>
  <text x="670" y="57" text-anchor="middle" font-size="11" fill="#880e4f">convexe (CVXPY)</text>
  <text x="870" y="42" text-anchor="middle" font-size="11" fill="#4a148c">Politique de recharge</text>
  <text x="870" y="57" text-anchor="middle" font-size="11" fill="#4a148c">&amp; validation</text>
</svg>
</div>
{{< /rawhtml >}}

---

## 3- Sources de données

Ce projet intègre trois datasets : séries temporelles d'émissions, simulation de batterie, et sessions de recharge résidentielles réelles.

### Données d'émission

MOER (Marginal Operating Emission Rate) CAISO North via l'API WattTime.

Série temporelle de CO2 lbs/MWh à intervalles de 5 minutes pour le nord de la Californie (1 année complète). Le MOER représente les émissions marginales de la prochaine unité d'électricité produite.

### Données de batterie

Le Single Particle Model de PyBaMM est utilisé pour simuler le pack batterie d'une Tesla Model 3 Long Range composé de 4 416 cellules (96 en série × 46 en parallèle) :

| Paramètre | Cellule Tesla | Modèle PyBaMM |
|-----------|--------------|---------------|
| Capacité | 4,8 Ah | 4,73 Ah |
| Tension minimale | 2,5 V | 2,6 V |
| Tension maximale | 4,2 V | 4,1 V |

### Données de sessions de recharge

Les enquêtes de conduite simulée (NHTS) supposent un comportement uniforme et négligent la forte variabilité capturée par les données réelles (semaine vs week-end, horaires irréguliers de branchement). On utilise donc un dataset réel de recharge résidentielle issu d'une coopérative de logements norvégienne :

6 844 sessions de 96 utilisateurs (déc. 2018 - janv. 2020). Chaque session enregistre les horodatages de branchement/débranchement, l'énergie chargée (kWh) et l'identifiant utilisateur.

| Statistique | Valeur |
|-------------|--------|
| Durée moyenne de branchement | 11,85 heures |
| Énergie moyenne chargée | 15,58 kWh |
| Branchement le plus fréquent | 16h |
| Débranchement le plus fréquent | 7h |

![Distribution de l'énergie chargée](/images/energy_charged_distribution.png)

---

## 4- Prévision des émissions

**Objectif :** prédire la courbe MOER sur 24 heures (288 valeurs à intervalles de 5 min) pour identifier les fenêtres de recharge à faibles émissions.

Le défi central est de prédire précisément le timing des pics plutôt que leur magnitude. L'optimiseur est robuste aux erreurs de magnitude mais fragile face aux erreurs de timing : un décalage de 4 heures rate complètement la fenêtre à faibles émissions.

{{< rawhtml >}}
<svg viewBox="0 0 820 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:820px;display:block;margin:1.2rem auto;font-family:monospace;">
  <defs>
    <marker id="a2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#888"/>
    </marker>
  </defs>
  <rect x="10"  y="75" width="180" height="50" rx="6" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <rect x="235" y="75" width="180" height="50" rx="6" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <rect x="460" y="75" width="180" height="50" rx="6" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <rect x="685" y="75" width="125" height="50" rx="6" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.5"/>
  <path d="M190 100 L233 100" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>
  <path d="M415 100 L458 100" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>
  <path d="M640 100 L683 100" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>
  <text x="100" y="97"  text-anchor="middle" font-size="11" fill="#1565c0" font-weight="bold">1 an de données MOER</text>
  <text x="100" y="113" text-anchor="middle" font-size="11" fill="#1565c0">intervalles 5 min</text>
  <text x="325" y="97"  text-anchor="middle" font-size="11" fill="#2e7d32" font-weight="bold">Prétraitement</text>
  <text x="325" y="113" text-anchor="middle" font-size="11" fill="#2e7d32">DART, co-variables</text>
  <text x="550" y="97"  text-anchor="middle" font-size="11" fill="#e65100" font-weight="bold">Entraînement</text>
  <text x="550" y="113" text-anchor="middle" font-size="11" fill="#e65100">FFNN → LSTM</text>
  <text x="747" y="97"  text-anchor="middle" font-size="11" fill="#880e4f" font-weight="bold">Prévision 24h</text>
  <text x="747" y="113" text-anchor="middle" font-size="11" fill="#880e4f">288 valeurs</text>
  <text x="100" y="145" text-anchor="middle" font-size="9.5" fill="#666">API WattTime</text>
  <text x="325" y="145" text-anchor="middle" font-size="9.5" fill="#666">features jour/mois</text>
  <text x="550" y="145" text-anchor="middle" font-size="9.5" fill="#666">25 jours → 1 jour</text>
  <text x="747" y="145" text-anchor="middle" font-size="9.5" fill="#666">timing des pics</text>
</svg>
{{< /rawhtml >}}

### Fonction Huber Loss

Contrairement à d'autres loss function, la Huber loss applique une pénalité réduite aux erreurs mineures tout en imposant une pénalité linéaire aux erreurs importantes. Cette caractéristique la rend moins sensible aux outliers et mieux adaptée aux données présentant des valeurs extrêmes ou des pics soudains.

{{< rawhtml >}}
$$
L_{\delta}(y, \hat{y}) = 
\begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{si } |y - \hat{y}| \leq \delta \\
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{sinon}
\end{cases}
$$
{{< /rawhtml >}}

On utilise δ=1 dans nos deux modèles.

### Feedforward Neural Network (FFNN)

Comme baseline de prédiction, on utilise un FFNN classique avec l'architecture suivante :

- Input : 1 000 valeurs (83 heures) pour capturer les patterns hebdomadaires
- Output : 288 valeurs (24 heures) correspondant à l'horizon d'optimisation
- Hidden Layers : 8 couches cachées pour approximer les dynamiques d'émission non linéaires complexes, activation ReLU

![Prédiction FFNN en hiver — erreur de timing de 4 heures](/images/ffnn_winter_prediction.png)

**Huber loss été = 77,8** (pattern solaire régulier) et **Huber loss hiver = 320,6** (erreur de timing des pics ±4 heures)

Le FFNN traite chaque fenêtre de 1 000 valeurs indépendamment, mais les conditions météo hivernales évoluent séquentiellement (un ciel nuageux peut persister 2 à 3 jours par exemple). On a besoin d'un modèle avec une mémoire temporelle.

### Long Short-Term Memory (LSTM)

Les réseaux de neurones récurrents (RNN) possède un état caché à travers les pas de temps, capturant les dépendances séquentielles. En particulier, une architecture LSTM classique est composée de 4 FFNN (3 portes, 1 candidat) et traite le cell state et le hidden state (mémoires long terme et court terme) de la cellule précédente comme suit:

![Architecture LSTM](/images/lstm.png)

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

- Données d'entraînement : 25 jours pour prédire 1 jour complet
- Taille de fenêtre : 20 pas de temps (100 minutes)
- Epochs : 10 (convergence validée sur un jeu de validation)
- Prétraitement : normalisation DART + co-variables jour/mois

![Prédiction LSTM en hiver — timing des pics précis](/images/Picture28.png)

**Été : Huber loss = 78,6** (identique au FFNN) et **Hiver : Huber loss = 231,9** (**amélioration de 28%**, timing des pics ±30 min)

### Comparaison des modèles

| Saison | Huber Loss FFNN | Huber Loss LSTM | Erreur de timing |
|--------|----------------|----------------|-----------------|
| Été | 77,8 | 78,6 | <15 min (les deux) |
| Hiver | 320,6 | 231,9 | FFNN ±4h / LSTM ±30min |

Le LSTM prédit mieux quand les pics surviennent, même si leur magnitude peut être surestimée d'un facteur 2 à 3.

---

## 5- Modélisation de la batterie

La tension aux bornes de la batterie est non linéaire avec le SOC et diffère entre charge et décharge en raison de l'électrochimie :

![Batterie — tension en fonction du SOC](/images/soc_vs_voltage.png)

On génère des courbes de charge/décharge à haute granularité sur 0–100% de SOC pour correspondre aux relations puissance-SOC convexes.

### Limite de puissance en charge

La tension de la batterie monte lentement de 0 à 95% de SOC (courant constant), puis rapidement de 95 à 100% (réduction en tension constante). Modèle convexe en deux morceaux :

{{< rawhtml >}}
$$
P_{\text{charge}}(SOC) = 
\begin{cases} 
142.8 \log(SOC) + 8755.4 & \text{pour } SOC \in [0, 0.95] \\
-175107.3 \cdot SOC + 175107.3 & \text{pour } SOC \in (0.95, 1]
\end{cases}
$$
{{< /rawhtml >}}

![Limites de puissance en charge et décharge](/images/charge_model.png)

### Limite de puissance en décharge

{{< rawhtml >}}
$$
P_{	ext{décharge}}(SOC) = 306{,}3 \log(SOC) + 8838{,}8
$$
{{< /rawhtml >}}

![Limites de puissance en décharge](/images/discharge_model.png)

---

## 6- Optimisation

Trois stratégies de recharge sont comparées sur les 6 844 sessions résidentielles réelles :

1. **Baseline :** branchement → charge à 100% → arrêt
2. **Shift Charging :** recharge opportuniste pendant les fenêtres à faibles émissions → atteinte du SOC cible
3. **Vehicle-to-Grid (V2G) :** décharge pendant les pics de forte émission + recharge pendant les fenêtres propres

**Objectif :** minimiser les émissions totales sur la session de recharge

{{< rawhtml >}}
$$
\min_{P(t), I(t), SOC(t)} \sum_{t=0}^{T} \text{MOER}(t) \times P(t) \times \Delta t
$$
{{< /rawhtml >}}

où MOER(t) est la courbe d'émission prévue, P(t) la puissance de charge, Δt = 5 min.

### Contraintes

Dynamique du SOC (équation d'état) :
{{< rawhtml >}}
$$
SOC(t+1) = SOC(t) + \frac{I(t)}{Q} \Delta t
$$
{{< /rawhtml >}}

Relation puissance-courant (pertes résistives) :
{{< rawhtml >}}
$$
P(t) \geq V \cdot I(t) + C \cdot |I(t)|
$$
{{< /rawhtml >}}

Limites de puissance (physique de la batterie + puissance du chargeur) :
{{< rawhtml >}}
$$
\begin{cases}
0 \leq P(t) \leq \min(P_{\text{charge}}(SOC), P_{\text{limit}}) & \text{(shift charging)} \\
P_{\text{décharge}}(SOC) \leq P(t) \leq P_{\text{charge}}(SOC) & \text{(V2G)}
\end{cases}
$$
{{< /rawhtml >}}

Conditions aux limites (issues des données de session) :
{{< rawhtml >}}
$$
\begin{align*}
SOC(t=0) &= SOC_{\text{start}} \quad \text{(SOC initial, datant du précédent trajet)} \\
SOC(t=T) &\geq SOC_{\text{target}} \quad \text{(SOC final, nécessaire au prochain trajet)}
\end{align*}
$$
{{< /rawhtml >}}

### Paramètres

| Symbole | Description | Valeur |
|---------|-------------|--------|
| $P_{	ext{charge}}(SOC)$ | Puissance maximale de charge | Issu du modèle batterie |
| $P_{	ext{décharge}}(SOC)$ | Puissance maximale de décharge | Issu du modèle batterie |
| $P_{	ext{limite}}$ | Puissance nominale du chargeur | 9,6 kW (Niveau 2 : 240 V × 40 A) |
| $V$ | Tension de charge | 240 V |
| $Q$ | Capacité du pack | 250 Ah |
| $C$ | Coefficient de pertes | 36 (~15% de rendement aller-retour) |
| $\Delta t$ | Résolution temporelle | 5 min |
| $SOC_{	ext{initial}}/ SOC_{	ext{cible}}$ | SOC initial/final | Issus des données de recharge |

Ce problème d'optimisation est résolu avec CVXPY.

---

## 7- Résultats & Discussion

### Résultats

![Résultats — CO2 moyen par stratégie](/images/ave_co2_3.png)

| Stratégie | CO2 moyen (lbs) | Réduction vs Baseline |
|-----------|----------------|----------------------|
| **Baseline** | 7,5 | — |
| **Shift Charging** | 3,2 | **−57%** |
| **Vehicle-to-Grid** | −0,8 | **−111%** (net négatif) |

Le résultat net négatif du V2G provient des émissions évitées qui sont comptabilisées négativement dans le bilan.

### Limitations

- **Prédictions LSTM en hiver :** Huber Loss >100 ; le timing des pics est bien prédit mais la magnitude peut être sur-estimée ou sous-estimée d'un facteur 2 à 3.
- **Modèle de batterie simplifié :** variables limitées, relation SOC logarithmique uniquement, effets de température et de vieillissement non pris en compte.
- **Décalage géographique :** les données de recharge norvégiennes se substituent aux données californiennes indisponibles. Les valeurs numériques de CO2 sont donc théoriques, bien que le cadre méthodologique soit conceptuellement valide.

### Prochaines étapes

- Explorer des variables supplémentaires pour améliorer la précision de la prévision des émissions.
- Intégrer les paramètres de température et d'électrochimie de la batterie pour des limites de puissance plus précises.
- Explorer des scénarios additionnels : courbes de prix de l'électricité, protection de la batterie contre les SOC trop bas.

---
`Python` · `PyTorch` · `Darts` · `LSTM` · `CVXPY` · `PyBamm` · `pandas` · `numpy` · `matplotlib` · `WattTime API`