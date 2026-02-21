---
title: "Génération de photos à partir de portraits-robots"
date: 2023-08-31
draft: false
categories: ["Computer Vision", "Generative AI", "Deep Learning"]
summary: "Concepts et architectures de modèles sketch-to-face, de l'autoencoder au U-Net jusqu'au Pix2Pix (GAN conditionnel)."
cover:
  alt: "Portrait-robot vers photo réaliste"
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

## 1- Vue d'ensemble

> **Note :** Cette page est une note personnelle des concepts et architectures des modèles image-to-image explorés pendant mon passage à la Direction Générale de la Gendarmerie Nationale, illustrée par une petite application sur le dataset public CUHK de croquis de visages.

**Génération de photos à partir de portraits-robots**

![dggn-sketch-to-photo.png](/images/face.png)

---

## 2- Contexte & Motivation

### Problématique

Les portraits-robots sont l'outil principal des enquêteurs lorsqu'aucune photographie du suspect n'est disponible. Produits à partir des descriptions des témoins, ce sont des croquis qui capturent la géométrie du visage mais perdent toute information sur la texture, la couleur, la peau, l'éclairage et le réalisme photographique. Le défi est de générer une image photoréaliste à partir de toutes ces informations manquantes de manière à la fois :

- Structurellement fidèle : le visage généré doit correspondre à la géométrie du croquis
- Photoréaliste : il doit ressembler à une vraie photographie
- Généralisable : il doit fonctionner sur des croquis jamais vus lors de l'entraînement

Il s'agit d'un problème d'**image-to-image translation** : le modèle doit apprendre une correspondance complexe d'un domaine visuel (croquis) vers un autre (photographies), sans supervision intermédiaire sur ce que devrait être la texture manquante.

### Objectif de cette page

Explorer les algorithmes de génération de portraits à partir de croquis, en progressant d'un autoencoder de base vers un GAN conditionnel complet (Pix2Pix).

{{< rawhtml >}}
<div style="margin: 1.5rem 0;">
<svg viewBox="0 0 760 90" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:760px;display:block;margin:auto;font-family:monospace;">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>
  <rect x="5"   y="20" width="130" height="50" rx="6" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.5"/>
  <rect x="170" y="20" width="130" height="50" rx="6" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.5"/>
  <rect x="335" y="20" width="130" height="50" rx="6" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <rect x="500" y="20" width="130" height="50" rx="6" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.5"/>
  <rect x="665" y="20" width="90"  height="50" rx="6" fill="#ede7f6" stroke="#ce93d8" stroke-width="1.5"/>
  <path d="M135 45 L168 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M300 45 L333 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M465 45 L498 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M630 45 L663 45" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="70"  y="42" text-anchor="middle" font-size="11" fill="#1565c0">Dataset CUFS</text>
  <text x="70"  y="57" text-anchor="middle" font-size="11" fill="#1565c0">+ augmentation</text>
  <text x="235" y="42" text-anchor="middle" font-size="11" fill="#2e7d32">Autoencoder</text>
  <text x="235" y="57" text-anchor="middle" font-size="11" fill="#2e7d32">(baseline)</text>
  <text x="400" y="42" text-anchor="middle" font-size="11" fill="#e65100">U-Net</text>
  <text x="400" y="57" text-anchor="middle" font-size="11" fill="#e65100">+ skip connections</text>
  <text x="565" y="42" text-anchor="middle" font-size="11" fill="#880e4f">Pix2Pix</text>
  <text x="565" y="57" text-anchor="middle" font-size="11" fill="#880e4f">GAN conditionnel</text>
  <text x="710" y="42" text-anchor="middle" font-size="11" fill="#4a148c">Application</text>
  <text x="710" y="57" text-anchor="middle" font-size="11" fill="#4a148c">dataset CUHK</text>
</svg>
</div>
{{< /rawhtml >}}

---

## 3- Concepts fondamentaux

### Réseau de neurones convolutif (CNN)

Pour les images, les couches convolutionnelles appliquent un petit filtre qui glisse sur l'image et détecte des motifs locaux (contours, puis formes, puis features de haut niveau). C'est bien plus efficace que de connecter chaque pixel à chaque neurone.

### Encoder

Un encoder est un CNN qui sous-échantillonne progressivement l'image, la comprimant en un vecteur bottleneck compact. La résolution spatiale diminue tandis que le nombre de features abstraites augmente.

{{< rawhtml >}}
<svg viewBox="0 0 700 70" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;display:block;margin:1rem auto;font-family:monospace;">
  <defs>
    <marker id="ae" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#888"/>
    </marker>
  </defs>
  <rect x="5"   y="10" width="110" height="40" rx="4" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.2"/>
  <rect x="145" y="20" width="90"  height="30" rx="4" fill="#bbdefb" stroke="#90caf9" stroke-width="1.2"/>
  <rect x="265" y="25" width="70"  height="20" rx="4" fill="#90caf9" stroke="#64b5f6" stroke-width="1.2"/>
  <rect x="365" y="28" width="50"  height="14" rx="4" fill="#64b5f6" stroke="#42a5f5" stroke-width="1.2"/>
  <rect x="445" y="30" width="35"  height="10" rx="4" fill="#42a5f5" stroke="#1e88e5" stroke-width="1.2"/>
  <path d="M115 35 L143 35" stroke="#888" stroke-width="1.2" marker-end="url(#ae)"/>
  <path d="M235 35 L263 35" stroke="#888" stroke-width="1.2" marker-end="url(#ae)"/>
  <path d="M335 35 L363 35" stroke="#888" stroke-width="1.2" marker-end="url(#ae)"/>
  <path d="M415 35 L443 35" stroke="#888" stroke-width="1.2" marker-end="url(#ae)"/>
  <text x="60"  y="58" text-anchor="middle" font-size="9" fill="#555">256×256×3</text>
  <text x="190" y="58" text-anchor="middle" font-size="9" fill="#555">128×128×64</text>
  <text x="300" y="58" text-anchor="middle" font-size="9" fill="#555">64×64×128</text>
  <text x="390" y="58" text-anchor="middle" font-size="9" fill="#555">32×32×256</text>
  <text x="462" y="58" text-anchor="middle" font-size="9" fill="#555">4×4×512</text>
  <text x="490" y="35" font-size="9.5" fill="#888">← plus abstrait, moins de détail spatial</text>
</svg>
{{< /rawhtml >}}

### Decoder

C'est le miroir de l'encoder. Il prend le bottleneck compressé et le sur-échantillonne progressivement jusqu'à la résolution d'origine via des convolutions transposées.

### Autoencoder = Encoder + Decoder

Pour l'image-to-image, on fournit un croquis en entrée et on entraîne le decoder à produire une photo. Le réseau apprend à traduire les domaines via le bottleneck. Cependant, les détails spatiaux fins (forme exacte des yeux, positions des cheveux) sont perdus à la compression et difficiles à récupérer. C'est la limite fondamentale de l'autoencoder pur.

![Autoencoder](/images/autoencoder.png)

### U-Net

Le U-Net (Ronneberger et al., 2015) ajoute des raccourcis directs qui copient les feature maps de chaque couche de l'encoder vers la couche correspondante du decoder. Celui-ci peut alors utiliser à la fois le contexte sémantique de haut niveau (depuis le bottleneck) et les détails spatiaux de bas niveau (depuis les skip connections), produisant des sorties plus nettes et plus fidèles.

![U-Net](/images/unet.png)

Les skip connections permettent au decoder de recevoir à la fois la représentation sémantique abstraite du bottleneck et les informations spatiales précises préservées depuis les premières couches de l'encoder.

### GAN (Generative Adversarial Network)

Même avec un U-Net, entraîner avec une loss MAE tend à produire des sorties floues. En effet, la loss MAE fait la moyenne de toutes les sorties plausibles : si le réseau est incertain entre deux textures plausibles, il les moyenne, produisant du flou. Un GAN résout ce problème en ajoutant un second réseau, le discriminateur, entraîné en opposition contre le générateur :

{{< rawhtml >}}
<svg viewBox="0 0 680 150" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;display:block;margin:1.2rem auto;font-family:monospace;">
  <defs>
    <marker id="gan" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#888"/>
    </marker>
    <marker id="ganr" markerWidth="7" markerHeight="7" refX="2" refY="3" orient="auto">
      <path d="M7,0 L7,6 L0,3 z" fill="#e53935"/>
    </marker>
  </defs>
  <rect x="5"   y="55" width="90" height="40" rx="4" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.2"/>
  <text x="50" y="78" text-anchor="middle" font-size="10" fill="#1565c0">Croquis (x)</text>
  <rect x="135" y="45" width="110" height="60" rx="5" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <text x="190" y="79" text-anchor="middle" font-size="11" fill="#e65100" font-weight="bold">Générateur G</text>
  <rect x="290" y="10" width="100" height="35" rx="4" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.2"/>
  <text x="340" y="31" text-anchor="middle" font-size="10" fill="#880e4f">Fausse photo G(x)</text>
  <rect x="290" y="105" width="100" height="35" rx="4" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.2"/>
  <text x="340" y="126" text-anchor="middle" font-size="10" fill="#2e7d32">Vraie photo (y)</text>
  <rect x="435" y="45" width="120" height="60" rx="5" fill="#ede7f6" stroke="#ce93d8" stroke-width="1.5"/>
  <text x="495" y="79" text-anchor="middle" font-size="11" fill="#4a148c" font-weight="bold">Discriminateur D</text>
  <rect x="600" y="55" width="75" height="40" rx="4" fill="#f5f5f5" stroke="#bdbdbd" stroke-width="1.2"/>
  <text x="637" y="72" text-anchor="middle" font-size="10" fill="#424242">Vraie ou</text>
  <text x="637" y="86" text-anchor="middle" font-size="10" fill="#424242">fausse ?</text>
  <path d="M95 75  L133 75" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <path d="M245 65 L288 35" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <path d="M390 27 L433 62" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <path d="M390 122 L433 88" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <path d="M555 75 L598 75" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <path d="M650 55 L650 20 L540 20 L540 42" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#gan)"/>
  <path d="M665 54 L665 1 L190 1 L190 42" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#gan)"/>
  <text x="450" y="15" text-anchor="middle" font-size="10" fill="#888">G Loss</text>
  <text x="600" y="32" text-anchor="middle" font-size="10" fill="#888">D Loss</text>
</svg>
{{< /rawhtml >}}

$$\mathcal{L}_{	ext{GAN}} = \mathbb{E}[\log D(x, y)] + \mathbb{E}[\log(1 - D(x, G(x)))]$$

où $x$ est le croquis (condition), $y$ est la vraie photo, $G(x)$ est la photo générée. G est forcé à produire des détails nets et réalistes, car tout flou est immédiatement détecté par D comme fausse image.

### Pix2Pix

Pix2Pix (Isola et al., 2017) combine U-Net + PatchGAN pour la transformation d'image supervisée. Le **discriminateur PatchGAN** classe des patches de 70×70 pixels comme réels ou faux plutôt que l'image entière, forçant ainsi un réalisme de texture locale partout, pas seulement une plausibilité globale.

![Pix2Pix](/images/pix.png)

{{< rawhtml >}}
$$\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{GAN}}(G, D) + \lambda \cdot \mathcal{L}_{	ext{L1}}(G) \qquad$$
{{< /rawhtml >}}

MAE permet une sortie proche de la vraie photo, tandis que le réseau GAN force le réalisme local.

---

## 4- Exemple d'application sur le dataset CUHK

### CUHK Face Sketch Database

C'est le benchmark académique de référence pour la synthèse portrait-croquis/photo. Il contient des paires croquis/photo réalisées par des étudiants en art à partir de vraies photographies.

| Split | Paires |
|-------|--------|
| Entraînement | 550 |
| Test | 56 |

Le dataset est très petit au regard des standards du deep learning. Une augmentation de données est appliquée (flips et rotations, appliqués symétriquement aux paires croquis/photo pour conserver la cohérence).

### Résultats

Trois exemples du jeu de test, portrait-croquis → vraie photo → prédiction du modèle :

![Résultats — croquis, réel, prédit](/images/facial-composite-results.png)

### Métriques d'évaluation : SSIM et PSNR

L'évaluation visuelle est la première approche naturelle pour ce type de tâche, mais elle ne permet pas de comparer des modèles de manière reproductible. Deux métriques sont standard pour la qualité d'image générée.

**PSNR (Peak Signal-to-Noise Ratio)** mesure le rapport entre la valeur maximale possible d'un pixel et la puissance du bruit de reconstruction, exprimé en décibels :

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right) \qquad \text{avec} \quad \text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

où MAX est la valeur maximale d'un pixel (255 pour une image 8 bits), et MSE l'erreur quadratique moyenne entre l'image réelle $y$ et l'image générée $\hat{y}$. Plus le PSNR est élevé, plus la reconstruction est fidèle. Un PSNR > 20 dB est généralement considéré comme acceptable pour la synthèse de visages.

**SSIM (Structural Similarity Index)** évalue la similarité perceptuelle entre deux images en combinant trois composantes : luminance, contraste et structure :

$$\text{SSIM}(y, \hat{y}) = \frac{(2\mu_y\mu_{\hat{y}} + c_1)(2\sigma_{y\hat{y}} + c_2)}{(\mu_y^2 + \mu_{\hat{y}}^2 + c_1)(\sigma_y^2 + \sigma_{\hat{y}}^2 + c_2)}$$

où $\mu$ désigne la moyenne locale, $\sigma^2$ la variance locale, $\sigma_{y\hat{y}}$ la covariance, et $c_1, c_2$ des constantes de stabilisation. SSIM prend ses valeurs dans $[-1, 1]$, une valeur de 1 indiquant une similarité parfaite. Contrairement au PSNR, il est conçu pour mieux refléter la perception humaine de la qualité d'image.

Dans l'exemple ci-dessus, SSIM ≈ 0,64 et PSNR ≈ 17,3 dB, limité par les capacités de mon ordinateur. Pour Pix2Pix appliqué au même dataset, les valeurs de référence rapportées sont SSIM ≈ 0,70 et PSNR ≈ 18,4 dB en configuration baseline, améliorées à SSIM ≈ 0,81 et PSNR ≈ 23,0 dB avec un prétraitement d'inversion gamma des croquis. Ces valeurs sont cohérentes avec les miennes, les résultats dépendant de la durée d'entraînement, de l'augmentation de données/quantité de données initiales et du matériel disponible.

---
`Python` · `TensorFlow / Keras` · `NumPy` · `Matplotlib` · `PIL`