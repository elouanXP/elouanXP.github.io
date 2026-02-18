---
title: "Sketch-to-Photo Generation for Law Enforcement"
date: 2023-08-31
draft: false
categories: ["Computer Vision", "Generative AI", "Deep Learning"]
tags: ["Python", "GAN", "Pix2Pix", "U-Net", "TensorFlow", "Computer Vision", "Image Generation"]
summary: "Built a generative AI system for the French National Gendarmerie to transform hand-drawn forensic facial composites into photorealistic portraits — iterating from autoencoders to U-Net to Pix2Pix (conditional GAN)."
cover:
  # image: "/images/dggn-sketch-to-photo.png"
  alt: "Facial composite to realistic photo"
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

**Organization:** French Ministry of Interior — Direction Générale de la Gendarmerie Nationale (DGGN)  
**Location:** Issy-les-Moulineaux, France  
**Duration:** June – August 2023  
**Supervisor:** Patrick Perrot, AI Innovation Team

During my final-year engineering internship at the French National Gendarmerie, I was tasked with building an AI system to transform hand-drawn forensic facial composites (*portrait-robots*) into photorealistic images. In parallel, I explored a text-to-face generation pipeline.

The goal was operational: give investigators a tool to produce realistic suspect portraits directly from witness descriptions, without requiring a forensic artist to be present.

---

## Why Is This Hard? The Core Problem

A facial composite is a **grayscale sketch** — it captures structure (face shape, nose, eyes) but loses everything about texture, color, skin, lighting, and photographic realism. The challenge is to **hallucinate** all that missing information in a way that is:

1. **Structurally faithful** — the generated face must match the sketch's geometry
2. **Photorealistic** — it must look like an actual photograph, not a cartoon or a blur
3. **Generalisable** — it must work on sketches it has never seen before

This is an **image-to-image translation** problem, and it requires the model to learn a complex mapping from one visual domain (sketches) to another (photographs).

---

## Dataset

**CUHK Face Sketch Database (CUFS)**

The standard academic benchmark for this task. It contains paired sketch/photo examples drawn by art students from real photographs.

| Split | Pairs |
|-------|-------|
| Training | ~1,194 |
| Test | ~100 |

**Custom Data Augmentation**

The dataset is small by deep learning standards. To increase diversity and reduce overfitting, I applied augmentation on-the-fly:

- Random horizontal flips
- Random crops and resizes
- Brightness and contrast jitter (photos only)
- Slight rotations

> **Key design choice:** augmentation was applied only to photographs for color/brightness, and symmetrically to sketch/photo pairs for geometric transforms — to preserve pairing consistency.

---

## A Quick Primer: The Building Blocks

Before describing what I built, here is a concise map of the concepts involved.

### Neural Network — the foundation

A neural network is a stack of mathematical layers that learn to transform an input (e.g. a pixel array) into an output (e.g. another image or a class label) by adjusting millions of parameters during training. Each layer applies a transformation followed by a non-linear activation function.

### Convolutional Neural Network (CNN)

For images, we use **convolutional layers** instead of dense layers. Rather than connecting every pixel to every neuron, a convolutional layer applies a small filter (e.g. 3×3) that slides across the image and detects local patterns (edges, textures, shapes). This is much more efficient and translation-invariant.

```
Input image (256×256×3)
    → Conv layer (detects edges)
    → Conv layer (detects shapes)
    → Conv layer (detects high-level features)
    → Output
```

### Encoder — compressing into a representation

An **encoder** is a CNN that progressively **downsamples** the image (using strided convolutions or pooling), compressing it into a compact representation called the **latent vector** or **bottleneck**. Spatial resolution decreases, but the number of channels (abstract features) increases.

```
256×256×3  →  128×128×64  →  64×64×128  →  32×32×256  →  4×4×512
              (more abstract, less spatial detail)
```

### Decoder — reconstructing from a representation

A **decoder** is the mirror image: it takes the compressed bottleneck and progressively **upsamples** it back to the original resolution using transposed convolutions (sometimes called "deconvolutions").

```
4×4×512  →  32×32×256  →  64×64×128  →  256×256×3
```

### Autoencoder — encoder + decoder together

An **autoencoder** combines both: encode → bottleneck → decode. Trained to reconstruct its input, it forces the network to learn a compact meaningful representation.

```
Input sketch → [Encoder] → Bottleneck → [Decoder] → Reconstructed image
```

For image translation, we feed a **sketch** as input and train the decoder to output a **photo** — so the network learns to translate domains through the bottleneck.

**Limitation:** the bottleneck is an information bottleneck. Fine spatial details (exact eye shape, hair strand positions) are lost in compression and hard to recover.

### U-Net — encoder + decoder + skip connections

The **U-Net** (Ronneberger et al., 2015 — originally for medical imaging) solves the bottleneck problem by adding **skip connections**: direct shortcuts that copy feature maps from each encoder layer directly to the corresponding decoder layer.

```
Encoder layer 1  ──────────────────────────────→  Decoder layer 5  (skip)
Encoder layer 2  ──────────────────────────────→  Decoder layer 4  (skip)
Encoder layer 3  ──────────────────────────────────→  Decoder layer 3  (skip)
        ↓                  Bottleneck                      ↑
```

This means the decoder can use **both** high-level semantic information (from the bottleneck) **and** low-level spatial details (from the skip connections). The result: much sharper, more faithful outputs.

This is why U-Net became the standard generator for image-to-image translation tasks.

### GAN — adding a "critic" to force realism

Even with U-Net, training with L1/MAE loss tends to produce **blurry** outputs. Why? Because L1 loss averages over all plausible outputs — if the network is uncertain between two plausible textures, it averages them, producing blur.

A **Generative Adversarial Network (GAN)** solves this by adding a second network called the **Discriminator**:

```
Generator (G):      Sketch → Fake Photo
Discriminator (D):  (Sketch, Photo) → Real or Fake?
```

They are trained adversarially:
- **G** tries to generate images realistic enough to fool **D**
- **D** tries to distinguish real photos from G's fakes

The resulting loss landscape forces **G** to produce sharp, realistic details — because any blurriness is immediately detected by **D** as fake.

$$\mathcal{L}_{\text{GAN}} = \mathbb{E}[\log D(x, y)] + \mathbb{E}[\log(1 - D(x, G(x)))]$$

where $x$ is the sketch (condition), $y$ is the real photo, $G(x)$ is the generated photo.

### Pix2Pix — conditional GAN for image-to-image translation

**Pix2Pix** (Isola et al., 2017) is the specific GAN architecture designed for paired image-to-image translation:

- **Generator:** U-Net (preserves spatial structure via skip connections)
- **Discriminator:** PatchGAN (classifies 70×70 overlapping patches as real/fake rather than the whole image — focuses on local texture realism)
- **Loss:** GAN adversarial loss + L1 reconstruction loss (λ=100)

{{< rawhtml >}}
$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{GAN}}(G, D)
+ \lambda \cdot \mathcal{L}_{\text{L1}}(G)
$$
{{< /rawhtml >}}


The L1 term keeps the output structurally close to the target; the GAN term forces local realism.

---

## My Approach: Iterative Architecture Search

I tested three architectures in increasing complexity, using each as a baseline for the next.

### Step 1 — Autoencoder (baseline)

**Architecture:** symmetric encoder-decoder, no skip connections  
**Loss:** MAE (L1)  
**Result:** Faces were structurally approximate but **very blurry** — the bottleneck destroyed fine detail

This was expected and useful: it confirmed that the model could learn the sketch→photo mapping direction, but needed a better architecture to preserve spatial details.

### Step 2 — U-Net (supervised, no GAN)

**Architecture:** U-Net with 5 encoder/decoder levels and skip connections  
**Loss:** MAE (L1)  
**Optimizer:** Adam (lr=2e-4, β₁=0.5)  
**Input/Output:** 256×256×3

```python
# Encoder
down1 = downsample(64,  4, batchnorm=False)(input)  # 128×128
down2 = downsample(128, 4)(down1)                   # 64×64
down3 = downsample(256, 4)(down2)                   # 32×32
down4 = downsample(512, 4)(down3)                   # 16×16
down5 = downsample(512, 4)(down4)                   # 8×8
bottleneck = downsample(512, 4)(down5)              # 4×4

# Decoder with skip connections
up1 = upsample(512, 4, dropout=True)(bottleneck)
up1 = Concatenate()([up1, down5])                   # 8×8
# ... (symmetric)
output = Conv2DTranspose(3, 4, activation='sigmoid')(up5)  # 256×256×3
```

**Result:** Significantly sharper than the autoencoder — facial structure (eyes, nose, hairline) well preserved. But still lacked photographic realism: skin texture and fine details were smooth/flat.

### Step 3 — Pix2Pix / Conditional GAN (final model)

**Generator:** U-Net (same as Step 2)  
**Discriminator:** PatchGAN — classifies 70×70 image patches as real/fake  
**Loss:** GAN adversarial + L1 (λ=100)  
**Optimizer:** Adam (lr=2e-4, β₁=0.5) for both G and D

The PatchGAN discriminator is key: by judging local patches rather than the whole image, it forces the generator to be locally realistic everywhere — not just globally convincing.

Training alternates between:
1. Update D: maximize ability to distinguish real (sketch, real photo) from fake (sketch, generated photo)
2. Update G: minimize D's ability to detect fakes + minimize L1 distance to real photo

**Result:** Most realistic outputs — skin texture, lighting, and hair detail significantly improved. Results described by supervisor as *"particularly promising"* despite limited compute (training cut short).

---

## Results

Three examples from the test set (facial composite → real photo → predicted photo):

![Results](/images/facial-composite-results.png)

**Observed strengths:**
- Facial geometry (eyes, nose, mouth positioning) faithfully preserved
- Skin tone and rough hair texture plausible
- Neutral expression consistent with ID photography standard

**Limitations:**
- Output images lack fine sharpness — more training epochs needed
- Dataset size (even with augmentation) limits generalization
- No quantitative metrics computed (FID, SSIM) — evaluation was visual

---

## Parallel Project: Text-to-Face (Partial)

The second project aimed to generate a face from a **textual description** ("male, 30s, dark short hair, angular jaw").

**What I built:**
- A **text encoder** pipeline: map a textual description to a feature vector (embeddings)
- Integration with an **external image generation API** for the visual output step

**What I did not build:**
- An end-to-end trained text-to-image model — the training pipeline had unresolved errors and the compute requirement was prohibitive on the available hardware

**Status:** Proof-of-concept exploration. The text encoding part was functional; the generation relied on an external API and was not integrated into a deployable pipeline.

---

## Operational Context

Throughout the internship I had several working sessions with forensic sketch artists and investigators to understand real operational needs:

- How are composites currently used in investigations?
- What level of realism is needed for public appeals vs. facial recognition systems?
- What are the legal constraints on AI-generated suspect images?

This grounding shaped key design decisions: prioritizing facial geometry faithfulness over creative realism, and ensuring all outputs were clearly marked as AI reconstructions.

---

## Technical Stack

`Python` · `TensorFlow / Keras` · `NumPy` · `Matplotlib` · `PIL`  
**Dataset:** CUHK Face Sketch Database (CUFS)  
**Architecture:** Pix2Pix — U-Net Generator + PatchGAN Discriminator

---

## Key Takeaways

| Architecture | Loss | Realism | Structure | Training |
|-------------|------|---------|-----------|---------|
| Autoencoder | L1 | ❌ Blurry | ⚠️ Approximate | Fast |
| U-Net (supervised) | L1 | ⚠️ Smooth | ✅ Good | Medium |
| Pix2Pix (GAN) | GAN + L1 | ✅ Best | ✅ Good | Slow |

The progression from autoencoder to Pix2Pix illustrates a fundamental principle in generative modeling: **pixel-wise loss functions alone cannot enforce perceptual realism** — you need an adversarial signal to push the generator toward the true data distribution.

---

*Internship evaluation: "Mr. Pulvéric demonstrated strong capabilities in tackling a complex subject and turning it into an operational tool. He showed excellent initiative and focused on finding practical solutions."*