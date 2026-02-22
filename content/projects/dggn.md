---
title: "Sketch-to-Photo Generation for Law Enforcement"
date: 2023-08-31
draft: false
categories: ["Computer Vision", "Generative AI", "Deep Learning"]
summary: " Sketch-to-Face Translation, concepts & architectures. Iterating from autoencoders to U-Net to Pix2Pix (conditional GAN)."
cover:
  image: "/images/gn.png"
  alt: "Facial composite to realistic photo"
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

## 1- Overview

 >**Note:** This is a personal review and reminder of the image-to-image translation concepts and architectures explored during my time at the French National Gendarmerie, illustrated with a small application on the CUHK face-sketch dataset.

**Sketch-to-Face Translation - Concepts & Architectures**

![dggn-sketch-to-photo.png](/images/face.png)

---

## 2- Context & Motivation

### Problem Statement

Forensic facial composites are the primary tool investigators use when no photograph of a suspect is available. Produced from witness descriptions, they are sketch-like pictures that capture facial geometry but lose everything about texture, color, skin, lighting, and photographic realism. The challenge is to generate all that missing information in a way that is simultaneously:

- Structurally faithful: the generated face must match the sketch's geometry
- Photorealistic: it must look like an actual photograph, not a cartoon or a blur
- Generalisable: it must work on sketches it has never seen before

This is an **image-to-image translation** problem: the model must learn a complex mapping from one visual domain (sketches) to another (photographs), with no intermediate supervision on what the missing texture should look like.

### Goal of this page

Explore sketch-to-photo face generation algorithms, progressing from a simple autoencoder baseline to a full conditional GAN (Pix2Pix).

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
  <text x="70"  y="42" text-anchor="middle" font-size="11" fill="#1565c0">CUFS dataset</text>
  <text x="70"  y="57" text-anchor="middle" font-size="11" fill="#1565c0">+ augmentation</text>
  <text x="235" y="42" text-anchor="middle" font-size="11" fill="#2e7d32">Autoencoder</text>
  <text x="235" y="57" text-anchor="middle" font-size="11" fill="#2e7d32">(baseline)</text>
  <text x="400" y="42" text-anchor="middle" font-size="11" fill="#e65100">U-Net</text>
  <text x="400" y="57" text-anchor="middle" font-size="11" fill="#e65100">+ skip connections</text>
  <text x="565" y="42" text-anchor="middle" font-size="11" fill="#880e4f">Pix2Pix</text>
  <text x="565" y="57" text-anchor="middle" font-size="11" fill="#880e4f">conditional GAN</text>
  <text x="710" y="42" text-anchor="middle" font-size="11" fill="#4a148c">Application</text>
  <text x="710" y="57" text-anchor="middle" font-size="11" fill="#4a148c">CUHK dataset</text>
</svg>
</div>
{{< /rawhtml >}}

---

## 3- Core concepts

### Convolutional Neural Network (CNN)

For images, convolutional layers apply a small filter that slides across the image and detects local patterns (edges, then shapes, then high-level features). This is far more efficient than connecting every pixel to every neuron.

### Encoder

An encoder is a CNN that progressively downsamples the image, compressing it into a compact bottleneck vector. Spatial resolution decreases while the number of abstract feature channels increases.

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
  <text x="490" y="35" font-size="9.5" fill="#888">← more abstract, less spatial detail</text>
</svg>
{{< /rawhtml >}}

### Decoder 

It is the mirror of the encoder. It takes the compressed bottleneck and progressively upsamples it back to the original resolution using transposed convolutions.

### Autoencoder = Encoder + Decoder

For image translation, we feed a sketch as input and train the decoder to output a photo. The network learns to translate domains through the bottleneck. However, fine spatial details (exact eye shape, hair positions) are lost in compression and hard to recover. This is the fundamental limitation of the pure autoencoder.

![dggn-sketch-to-photo.png](/images/autoencoder.png)

### U-Net

The U-Net (Ronneberger et al., 2015) adds direct shortcuts that copy feature maps from each encoder layer to the corresponding decoder layer. The decoder can then use both high-level semantic context (from the bottleneck) and low-level spatial details (from the skip connections), producing sharper, more faithful outputs.

![dggn-sketch-to-photo.png](/images/unet.png)

Skip connections allow the decoder to receive both the abstract semantic representation from the bottleneck and the precise spatial information preserved from early encoder layers.

### GAN (Generative Adversarial Network)

Even with U-Net, training with MAE loss tends to produce blurry outputs. Indeed, MAE loss averages over all plausible outputs so if the network is uncertain between two plausible textures, it averages them, producing blur. A GAN solves this by adding a second network, the discriminator, trained adversarially against the generator:

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
  <!-- Sketch input -->
  <rect x="5"   y="55" width="90" height="40" rx="4" fill="#e3f2fd" stroke="#90caf9" stroke-width="1.2"/>
  <text x="50" y="78" text-anchor="middle" font-size="10" fill="#1565c0">Sketch (x)</text>
  <!-- Generator -->
  <rect x="135" y="45" width="110" height="60" rx="5" fill="#fff3e0" stroke="#ffcc80" stroke-width="1.5"/>
  <text x="190" y="79" text-anchor="middle" font-size="11" fill="#e65100" font-weight="bold">Generator G</text>
  <!-- Fake photo -->
  <rect x="290" y="10" width="100" height="35" rx="4" fill="#fce4ec" stroke="#f48fb1" stroke-width="1.2"/>
  <text x="340" y="31" text-anchor="middle" font-size="10" fill="#880e4f">Fake photo G(x)</text>
  <!-- Real photo -->
  <rect x="290" y="105" width="100" height="35" rx="4" fill="#e8f5e9" stroke="#a5d6a7" stroke-width="1.2"/>
  <text x="340" y="126" text-anchor="middle" font-size="10" fill="#2e7d32">Real photo (y)</text>
  <!-- Discriminator -->
  <rect x="435" y="45" width="120" height="60" rx="5" fill="#ede7f6" stroke="#ce93d8" stroke-width="1.5"/>
  <text x="495" y="79" text-anchor="middle" font-size="11" fill="#4a148c" font-weight="bold">Discriminator D</text>
  <!-- Output -->
  <rect x="600" y="55" width="75" height="40" rx="4" fill="#f5f5f5" stroke="#bdbdbd" stroke-width="1.2"/>
  <text x="637" y="72" text-anchor="middle" font-size="10" fill="#424242">Real</text>
  <text x="637" y="86" text-anchor="middle" font-size="10" fill="#424242">or Fake?</text>
  <!-- Arrows -->
  <path d="M95 75  L133 75" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <path d="M245 65 L288 35" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <path d="M390 27 L433 62" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <path d="M390 122 L433 88" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <path d="M555 75 L598 75" stroke="#888" stroke-width="1.2" marker-end="url(#gan)"/>
  <!-- Feedback arrows -->
  <path d="M650 55 L650 20 L540 20 L540 42" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#gan)"/>
  <path d="M665 54 L665 1 L190 1 L190 42" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#gan)"/>
  <text x="450" y="15" text-anchor="middle" font-size="10" fill="#888">G Loss</text>
  <text x="600" y="32" text-anchor="middle" font-size="10" fill="#888">D Loss</text>
</svg>
{{< /rawhtml >}}

$$\mathcal{L}_{\text{GAN}} = \mathbb{E}[\log D(x, y)] + \mathbb{E}[\log(1 - D(x, G(x)))]$$

where $x$ is the sketch (condition), $y$ is the real photo, $G(x)$ is the generated photo. The adversarial loss landscape forces G to produce sharp, realistic details, because any blurriness is immediately detected by D as fake.

### Pix2Pix

Pix2Pix (Isola et al., 2017) combines U-Net + PatchGAN for paired image-to-image translation. The **PatchGAN discriminator** classifies 70×70 overlapping patches as real/fake rather than the whole image, hence forcing local texture realism everywhere, not just global plausibility.

![dggn-sketch-to-photo.png](/images/pix.png)

{{< rawhtml >}}
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}}(G, D) + \lambda \cdot \mathcal{L}_{\text{L1}}(G) \qquad$$
{{< /rawhtml >}}

The MAE term anchors the output structurally close to the real photo while the GAN term forces local realism on top.

---

## 4- Example of potential application on the CUHK dataset


### CUHK Face Sketch Database

This the standard academic benchmark for face sketch-to-photo synthesis. It contains paired sketch/photo examples drawn by art students from real photographs.

| Split | Pairs |
|-------|-------|
| Training | ~550 |
| Test | ~56 |

The dataset is very small by deep learning standards. To increase diversity and reduce overfitting, data augmentation is applied on-the-fly (flips and rotations, symmetrically to sketch/photo pairs to preserve pairing consistency).

### Results

Three examples from the test set: facial composite → real photo → model prediction:

![Results — composite, real, predicted](/images/facial-composite-results.png)

### Evaluation metrics: SSIM and PSNR

Visual inspection is the natural first approach for this type of task, but it does not allow reproducible model comparison. Two metrics are standard for generated image quality.

**PSNR (Peak Signal-to-Noise Ratio)** measures the ratio between the maximum possible pixel value and the power of the reconstruction noise, expressed in decibels:

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right) \qquad \text{avec} \quad \text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

where MAX is the maximum pixel value (255 for an 8-bit image), and MSE is the mean squared error between the real image $y$ and the generated image $\hat{y}$​. Higher PSNR means more faithful reconstruction. PSNR > 20 dB is generally considered acceptable for face synthesis.

SSIM (Structural Similarity Index) evaluates the perceptual similarity between two images by combining three components: luminance, contrast and structure:

$$\text{SSIM}(y, \hat{y}) = \frac{(2\mu_y\mu_{\hat{y}} + c_1)(2\sigma_{y\hat{y}} + c_2)}{(\mu_y^2 + \mu_{\hat{y}}^2 + c_1)(\sigma_y^2 + \sigma_{\hat{y}}^2 + c_2)}$$

where $\mu$ denotes the local mean, $\sigma^2$ the local variance, $\sigma_{y\hat{y}}$ the cross-covariance, and $c1, c2$ stabilization constants. SSIM ranges in $[−1,1]$, with 1 indicating perfect similarity. Unlike PSNR, it is designed to better reflect human perception of image quality.

On the example above, SSIM≈0.64 and PSNR≈19.3 dB, limited by available computer performance. For Pix2Pix applied to the same dataset, reported reference values are SSIM≈0.70 and PSNR≈18.4 dB in baseline configuration, improving to SSIM≈0.81 and PSNR≈23.0 dB with sketch gamma inversion preprocessing. These values are consistent with mine. Results depend on training duration, data quantity and augmentation, and available hardware.

---
`Python` · `TensorFlow / Keras` · `NumPy` · `Matplotlib` · `PIL`