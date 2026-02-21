---
title: "Impact du COVID-19 sur la mobilité en Californie"
date: 2023-05-09
draft: false
categories: ["Data Science", "Transportation", "Public Policy"]
summary: "Étude de recherche analysant les mobilités de 2,4M+ d'utilisateurs pour orienter la politique de réduction des GES de la Californie en 2030 — machine learning, analyse géospatiale big data et science des réseaux."
cover:
  image: ""
  alt: "COVID-19 et analyse de mobilité"
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

<a href="https://humnetlab.berkeley.edu/wp-content/uploads/2024/05/Final_Draft_CARB.pdf" target="_blank">**Voir le rapport complet (PDF) →**</a>

## 1. Vue d'ensemble

**Étude de recherche à grande échelle analysant les mobilités de 2,4M+ utilisateurs californiens pour orienter la politique de réduction des émissions de gaz à effet de serre de l'État à l'horizon 2030 — machine learning, analyse géospatiale big data et science des réseaux.**

![carb-overview.png](/images/carb-overview.png)

---

## 2. Contexte & Motivation

### Problématique

La Californie s'est engagée à réduire ses émissions de gaz à effet de serre de 40% sous les niveaux de 1990 d'ici 2030. Le transport est le principal contributeur à ces émissions, et les **Vehicle Miles Traveled (VMT)** — le kilométrage parcouru en voiture — constituent le levier central. La pandémie de COVID-19 a créé une expérience naturelle sans précédent : une perturbation soudaine et à l'échelle de l'État qui a révélé quels comportements de déplacement sont habituels (et réversibles) et lesquels sont structurels (et permanents).

Le **California Air Resources Board (CARB)**, en partenariat avec l'**Université de Californie à Berkeley** et le **Lawrence Berkeley National Laboratory**, a commandé cette étude pour quantifier ces changements et les traduire en recommandations politiques concrètes.

### Défi technique

Comprendre la mobilité à cette échelle requiert de résoudre simultanément trois problèmes distincts : inférer le mode de transport (voiture vs. marche) depuis des pings GPS bruts sans aucune donnée d'entraînement labelisée, détecter les déménagements résidentiels depuis des patterns de localisation nocturne, et mesurer les changements structurels des flux de déplacement domicile-travail sur ~8 000 secteurs de recensement sur quatre ans. Chacun nécessite une approche algorithmique différente — non supervisée, semi-supervisée et par analyse de graphes respectivement.

### Objectif du projet

Mesurer l'impact du COVID-19 sur les VMT, les déplacements domicile-travail et les déménagements résidentiels à la granularité du secteur de recensement en Californie, à partir de quatre années de données LBS anonymisées, et dériver des stratégies de réduction des VMT ciblées par région et groupe démographique pour alimenter la feuille de route climatique 2030 du CARB.

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
  <text x="77"  y="42" text-anchor="middle" font-size="11" fill="#1565c0">Données LBS</text>
  <text x="77"  y="57" text-anchor="middle" font-size="11" fill="#1565c0">2,4M+ utilisateurs</text>
  <text x="257" y="42" text-anchor="middle" font-size="11" fill="#2e7d32">Détection</text>
  <text x="257" y="57" text-anchor="middle" font-size="11" fill="#2e7d32">de mode GMM</text>
  <text x="437" y="42" text-anchor="middle" font-size="11" fill="#e65100">Déménagements</text>
  <text x="437" y="57" text-anchor="middle" font-size="11" fill="#e65100">KMeans-SVM</text>
  <text x="617" y="42" text-anchor="middle" font-size="11" fill="#880e4f">Analyse de</text>
  <text x="617" y="57" text-anchor="middle" font-size="11" fill="#880e4f">réseau de trajet</text>
  <text x="790" y="42" text-anchor="middle" font-size="11" fill="#4a148c">Recommandations</text>
  <text x="790" y="57" text-anchor="middle" font-size="11" fill="#4a148c">politiques</text>
</svg>
</div>
{{< /rawhtml >}}

---

## 3. Dataset

Données **Location-Based Service (LBS)** fournies par Spectus — pings GPS de téléphones mobiles anonymisés, opt-in, conformes CCPA/RGPD, collectés en Californie de 2019 à 2022.

| Année | Utilisateurs totaux | Utilisateurs haute qualité |
|-------|--------------------:|---------------------------:|
| 2019 | 9,4M | 3 482 574 |
| 2020 | 5,9M | 2 396 990 |
| 2021 | 5,2M | 2 036 110 |
| 2022 | 5,6M | 2 582 405 |

**Résolution spatiale :** census block group (~600–3 000 personnes). **Filtres qualité** appliqués pour ne conserver que les utilisateurs avec une couverture suffisante : >316 pings, >60 jours actifs, >10 visites à la fois sur le lieu de domicile et de travail.

**Détection domicile et lieu de travail** par heuristique sur la localisation la plus fréquentée pendant la plage horaire appropriée :

```python
home = most_frequent_location(user, time_range="7pm-7am", min_visits=10)
work = most_frequent_location(user, time_range="7am-7pm", weekdays_only=True, min_visits=10)
```

Validation par corrélation de Pearson r=0,5 avec la population des census tracts 2019 — satisfaisant pour des données opt-in anonymisées.

---

## 4. Méthodologie

Trois problèmes algorithmiques distincts, chacun nécessitant une approche différente.

### 4.1 Algorithme 1 — Détection de mode par GMM

**Défi.** Aucune donnée de trajet labelisée n'est disponible pour l'apprentissage supervisé — les trajets doivent être classés en motorisé, non-motorisé ou bruit sans aucune vérité terrain.

**Solution : Gaussian Mixture Model (non supervisé).** Chaque trajet est représenté par deux features : `log(max_speed_kph)` et `log(trajectory_length_km)`. Un GMM à k=3 composantes (sélectionné par la méthode du coude sur AIC/BIC) ajuste trois distributions gaussiennes dans cet espace de features 2D, produisant des assignations souples :

$$\text{GMM}(X) = \sum_{k=1}^{3} \pi_k \,\mathcal{N}(X \mid \mu_k, \Sigma_k)$$

{{< rawhtml >}}
<svg viewBox="0 0 680 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;display:block;margin:1.2rem auto;font-family:monospace;">
  <defs>
    <marker id="gm" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#888"/>
    </marker>
  </defs>
  <rect x="5" y="40" width="130" height="50" rx="5" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <text x="70" y="62" text-anchor="middle" font-size="10" fill="#1565c0" font-weight="bold">Features du trajet</text>
  <text x="70" y="76" text-anchor="middle" font-size="9.5" fill="#1565c0">log(max_speed)</text>
  <text x="70" y="88" text-anchor="middle" font-size="9.5" fill="#1565c0">log(length_km)</text>
  <rect x="175" y="30" width="130" height="70" rx="5" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <text x="240" y="58" text-anchor="middle" font-size="11" fill="#e65100" font-weight="bold">GMM k=3</text>
  <text x="240" y="73" text-anchor="middle" font-size="9.5" fill="#bf360c">coude AIC/BIC</text>
  <text x="240" y="87" text-anchor="middle" font-size="9.5" fill="#bf360c">assignation souple</text>
  <path d="M135 65 L173 65" stroke="#888" stroke-width="1.2" marker-end="url(#gm)"/>
  <path d="M305 50 L343 35" stroke="#888" stroke-width="1.2" marker-end="url(#gm)"/>
  <path d="M305 65 L343 65" stroke="#888" stroke-width="1.2" marker-end="url(#gm)"/>
  <path d="M305 80 L343 95" stroke="#888" stroke-width="1.2" marker-end="url(#gm)"/>
  <rect x="345" y="15" width="145" height="35" rx="4" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.2"/>
  <rect x="345" y="55" width="145" height="35" rx="4" fill="#fff9c4" stroke="#f9a825" stroke-width="1.2"/>
  <rect x="345" y="80" width="145" height="30" rx="4" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.2"/>
  <text x="418" y="36" text-anchor="middle" font-size="10" fill="#2e7d32" font-weight="bold">Motorisé  &gt;40 km/h</text>
  <text x="418" y="75" text-anchor="middle" font-size="10" fill="#f57f17" font-weight="bold">Non-motorisé  &lt;6,3 km/h</text>
  <text x="418" y="97" text-anchor="middle" font-size="10" fill="#880e4f" font-weight="bold">Bruit (ambigu)</text>
  <text x="345" y="125" font-size="9" fill="#555">→ Passage à l'échelle de milliards de trajets · sans labelisation manuelle</text>
</svg>
{{< /rawhtml >}}

---

### 4.2 Algorithme 2 — Détection de déménagement par KMeans-SVM

**Défi.** Détecter les déménagements intra-étatiques (≥8 km) depuis des patterns GPS nocturnes, sans aucune donnée de déménagement labelisée.

**Approche semi-supervisée en deux étapes :**

{{< rawhtml >}}
<svg viewBox="0 0 700 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;display:block;margin:1.2rem auto;font-family:monospace;">
  <defs>
    <marker id="km" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#888"/>
    </marker>
  </defs>
  <rect x="5" y="20" width="200" height="70" rx="5" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <text x="105" y="42" text-anchor="middle" font-size="11" fill="#1565c0" font-weight="bold">Étape 1 — K-Means</text>
  <text x="105" y="57" text-anchor="middle" font-size="9.5" fill="#1565c0">(non supervisé)</text>
  <text x="105" y="72" text-anchor="middle" font-size="9.5" fill="#1565c0">lat, lng, days_since_jan1</text>
  <text x="105" y="84" text-anchor="middle" font-size="9.5" fill="#1565c0">→ 2 clusters : avant / après</text>
  <path d="M205 55 L248 55" stroke="#888" stroke-width="1.2" marker-end="url(#km)"/>
  <text x="226" y="49" text-anchor="middle" font-size="8.5" fill="#888">pseudo-labels</text>
  <rect x="250" y="20" width="200" height="70" rx="5" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <text x="350" y="42" text-anchor="middle" font-size="11" fill="#2e7d32" font-weight="bold">Étape 2 — SVM</text>
  <text x="350" y="57" text-anchor="middle" font-size="9.5" fill="#2e7d32">(supervisé, c=0,01)</text>
  <text x="350" y="72" text-anchor="middle" font-size="9.5" fill="#2e7d32">large marge — évite le</text>
  <text x="350" y="84" text-anchor="middle" font-size="9.5" fill="#2e7d32">surapprentissage géographique</text>
  <path d="M450 55 L493 55" stroke="#888" stroke-width="1.2" marker-end="url(#km)"/>
  <rect x="495" y="20" width="200" height="70" rx="5" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <text x="595" y="42" text-anchor="middle" font-size="11" fill="#e65100" font-weight="bold">Déménagement détecté</text>
  <text x="595" y="57" text-anchor="middle" font-size="9.5" fill="#e65100">dist ≥ 8 km</text>
  <text x="595" y="72" text-anchor="middle" font-size="9.5" fill="#e65100">≥ 20 obs / cluster</text>
  <text x="595" y="84" text-anchor="middle" font-size="9.5" fill="#e65100">date estimée</text>
  <rect x="5" y="110" width="690" height="40" rx="4" fill="#f5f5f5" stroke="#e0e0e0" stroke-width="1"/>
  <text x="15" y="127" font-size="9.5" fill="#333" font-weight="bold">Performance (dataset synthétique) :</text>
  <text x="15" y="142" font-size="9.5" fill="#555">Précision : 86,4% (4 321 / 5 000 vrais positifs) · Erreur de type I : 1,6% (447 / 28 260 faux positifs)</text>
</svg>
{{< /rawhtml >}}

La date du déménagement est estimée par :
$$\text{MoveDate} = \min\bigl(\max(C_{\text{avant}}),\ \max(C_{\text{après}})\bigr)$$

---

### 4.3 Algorithme 3 — Rayon de giration

Le **rayon de giration** $r_g$ mesure l'étendue spatiale de l'activité d'un utilisateur — un scalaire synthétisant à quelle distance de son domicile une personne se déplace typiquement :

$$r_g(u) = \sqrt{\frac{1}{n_u} \sum_{i=1}^{n_u} \text{dist}\!\left(r_i(u) - r_{cm}(u)\right)^2}$$

où $r_{cm}(u)$ est le centre de masse de tous les lieux visités. Un $r_g$ élevé signale des déplacements longue distance dépendants de la voiture ; un $r_g$ faible indique une mobilité locale et piétonne. Suivre $r_g$ avant et après le confinement permet de quantifier non seulement si les personnes se sont moins déplacées, mais si elles ont parcouru des distances plus courtes.

---

### 4.4 Analyse du réseau de flux domicile-travail

Les déplacements domicile-travail sont modélisés comme un graphe dirigé pondéré : les **nœuds** sont les census tracts (~8 000), les **arêtes** sont pondérées par les flux de trajets domicile→travail observés. La **détection de communautés par l'algorithme de Louvain** identifie les bassins de déplacement naturels en maximisant la modularité :

$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Une modularité plus élevée signifie que le réseau se fractionne plus facilement en régions autonomes — la signature d'une réduction des déplacements inter-régionaux. Suivre $Q$ et le nombre de communautés d'une année sur l'autre révèle si la pandémie a fragmenté durablement la structure des déplacements domicile-travail en Californie.

---

## 5. Résultats

### Réduction des VMT — disparités régionales

Les comtés urbains (LA, Orange, Santa Barbara) ont enregistré des **réductions des VMT de 38–55%** en avril 2020 ; les comtés ruraux (Imperial, Kern) seulement **20–30%**. L'écart de 20 points de pourcentage reflète une dépendance à la voiture fondamentalement différente : les résidents urbains pouvaient substituer les trajets en voiture par la marche ou le télétravail, les résidents ruraux en grande partie non. Une corrélation positive (r=0,376) entre la superficie du comté et la persistance des VMT confirme que cette dépendance est structurelle, non comportementale.

Les VMT du week-end ont chuté plus fortement que ceux des jours de semaine — cohérent avec les déplacements non essentiels (loisirs, courses) étant plus élastiques que les trajets domicile-travail.

---

### Récupération selon le motif de déplacement

| Métrique | Trajets domicile-travail | Trajets hors-travail |
|----------|:------------------------:|:--------------------:|
| Récupération en 2022 | En dessous du niveau 2019 | Proche du niveau 2019 |
| Ratio travail/hors-travail | Jamais revenu au niveau pré-pandémie | — |

Les déplacements hors-travail (shopping, loisirs) ont retrouvé leur niveau de base d'ici 2022. Les trajets domicile-travail, non — le ratio travail/hors-travail n'a jamais retrouvé son niveau pré-pandémique. Le télétravail et le travail hybride ont restructuré non seulement où les personnes travaillent, mais la forme fondamentale de la demande de déplacement quotidien en Californie.

---

### Fragmentation du réseau de déplacements domicile-travail

| Année | Arêtes | Communautés | Modularité |
|-------|-------:|:-----------:|:----------:|
| 2019 | 145 838 | 6 | 0,628 |
| 2020 | 89 382 | 8 ↑ | 0,653 ↑ |
| 2021 | 73 401 | 8 | 0,648 |
| 2022 | 111 652 | 8 | **0,666** ↑ |

Le réseau a perdu **38,7% de ses arêtes** en 2020. Deux nouvelles communautés isolées ont émergé dans des régions éloignées (El Centro, Sierras orientales). Surtout, **la modularité est restée élevée jusqu'en 2022** — même après une récupération partielle des arêtes, le réseau n'a jamais retrouvé sa structure intégrée d'avant-pandémie. Les grands centres urbains (SF, LA, San Diego) continuent d'exercer une influence gravitationnelle dominante sur leurs régions environnantes.

---

### Déménagements résidentiels

Un pic brutal de déménagements s'est produit dans la fenêtre de 15 jours entre la déclaration d'urgence (4 mars) et l'ordre de confinement (19 mars 2020). La distribution des distances pendant cette fenêtre de crise était **bimodale** : un mode autour de 80 km (reshuffling local au sein des zones métropolitaines) et un mode autour de 550 km (inter-régional, correspondant approximativement à Bay Area ↔ Sud de la Californie).

Les census tracts à faibles revenus ont montré des déménagements de plus longue distance en moyenne — cohérent avec l'instabilité du logement forçant des déplacements involontaires plutôt que des relocalisations volontaires. Sorties nettes depuis SF et LA ; populations stables dans la Central Valley (Fresno, Bakersfield).

---

### Rayon de giration

Les cœurs urbains (Bay Area, LA, San Diego) affichent systématiquement $r_g$ ≈ 20 km — des patterns de déplacement compacts même avant la pandémie. Post-confinement, le $r_g$ urbain a fortement chuté. Les zones rurales ont montré moins de réduction : les résidents ont maintenu leurs patterns de déplacements longue distance, les alternatives (transports en commun, mobilités actives) étant absentes.

---

### Évaluation des interventions de mobilité (Sacramento 2019)

| Intervention | Date | Résultat |
|-------------|------|----------|
| Lancement trottinettes JUMP | Fév. 2019 | Aucun changement significatif |
| Lancement GIG Car Share | Mars 2019 | Aucun changement significatif |
| **Expansion de la flotte JUMP** | **Juin 2019** | **✓ Réduction significative des trajets motorisés** |
| Réforme SacRT Forward transit | Sep. 2019 | Aucun changement significatif |

Bootstrap sampling (100 itérations, n=50 000 trajets) sur les pourcentages de trajets motorisés mois par mois. Le résultat clé : **le lancement d'un programme n'a aucun effet mesurable ; la mise à l'échelle, si**. Atteindre une masse critique de véhicules déployés est nécessaire avant que la substitution modale se produise.

---

## 6. Impact & Recommandations politiques

Les résultats se traduisent en quatre leviers politiques concrets pour la stratégie 2030 du CARB :

**Zones urbaines** — capitaliser sur la réduction structurelle de la dépendance automobile post-COVID : réformes de zonage autour des transports en commun, réduction des minimums de stationnement, développement infill en banlieue pour consolider ces gains plutôt que de laisser l'usage de la voiture reprendre progressivement.

**Zones rurales** — le problème est plus difficile : la dépendance automobile a persisté précisément là où les alternatives sont absentes. L'infrastructure de recharge pour véhicules électriques et les voies de covoiturage répondent à la dimension émissions sans exiger de changement comportemental.

**Télétravail** — désormais structurellement ancré dans le marché du travail californien. La politique d'utilisation des sols devrait s'adapter — et non résister — en encourageant le développement résidentiel près des pôles d'emploi suburbains et en réduisant les VMT liés aux trajets domicile-travail.

**Micromobilité à l'échelle** (et non en pilote) — la seule catégorie d'intervention ayant montré un effet mesurable sur le report modal. La politique devrait prioriser la densité de déploiement rapide plutôt que l'étendue géographique.

**Résultats validés :** les données LBS confirmées comme alternative viable et économique aux enquêtes de déplacement coûteuses ; algorithmes réutilisables open-source pour la détection de mode et de déménagement sans données d'entraînement labelisées ; feuille de route 2030 du CARB alimentée par des objectifs de réduction des VMT à la granularité du census tract.

---
`Python` · `pandas` · `numpy` · `scikit-learn` · `networkx` · `geopandas` · `matplotlib`