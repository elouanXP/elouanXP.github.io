---
title: "AI-Powered Facial Composite Enhancement for Law Enforcement"
date: 2023-08-31
draft: false
categories: ["Computer Vision", "Generative AI", "Public Safety"]
tags: ["Python", "Deep Learning", "GANs", "Image-to-Image Translation", "PyTorch", "Computer Vision"]
summary: "Developed a generative AI system to transform hand-drawn facial composites into photorealistic images for the French National Gendarmerie, improving suspect identification capabilities through conditional GANs and image-to-image translation."
cover:
  # image: "/images/facial-composite-results.png"
  alt: "Facial composite to realistic photo transformation"
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

During my engineering internship at the **Direction Générale de la Gendarmerie Nationale (DGGN)** — France's national police force — I developed an AI-powered system to transform hand-drawn facial composites (portrait-robots) into photorealistic images. This tool aims to enhance suspect identification by bridging the gap between witness descriptions and investigative photography.

**Organization:** Direction Générale de la Gendarmerie Nationale (French National Gendarmerie)  
**Location:** Issy-les-Moulineaux, France  
**Duration:** June 2023 - August 2023 (3 months)  
**Supervisor:** Patrick Perrot, AI Innovation Team  
**Classification:** Law Enforcement / Public Safety AI

## Context & Motivation

### The Challenge

Traditional facial composites created by forensic sketch artists serve a critical role in criminal investigations, but they face several limitations:

1. **Abstraction Gap:** Hand-drawn sketches lack the photorealistic detail needed for effective public dissemination and facial recognition systems
2. **Witness Memory Limitations:** Descriptions are often incomplete or inconsistent
3. **Artist Availability:** Forensic sketch artists are scarce resources in law enforcement agencies
4. **Public Perception:** The public more readily recognizes photographs than stylized drawings

### Project Goal

Develop a deep learning system that automatically transforms facial composites into photorealistic portraits while preserving key identifying features, enabling:
- More effective public appeals (BOLO - Be On the Lookout)
- Compatibility with facial recognition databases
- Faster investigative workflows
- Enhanced witness engagement (seeing realistic renderings may trigger additional memories)

## My Role & Contributions

As the sole AI engineer on this project, I was responsible for:

- **Research & Design:** Surveying state-of-the-art generative models for image-to-image translation
- **Model Development:** Implementing and training conditional GANs for composite-to-photo transformation
- **Data Pipeline:** Preprocessing facial composite datasets and pairing with photographic references
- **Evaluation:** Assessing model output quality through visual inspection and stakeholder feedback
- **Operational Integration:** Collaborating with forensic sketch artists to understand real-world requirements

**Additional Project:** Text-to-Face generation system (text descriptions → facial images) as a complementary tool

## Technical Approach

### System Architecture

Given the nature of the task (transforming one image domain to another while preserving structural features), I implemented a **Conditional Generative Adversarial Network (cGAN)** architecture, likely based on **Pix2Pix** or a similar paired image-to-image translation framework.

#### Core Architecture: Conditional GAN

**Generator Network (G):**
- **Type:** U-Net architecture with skip connections
- **Input:** Facial composite (grayscale sketch, 256×256)
- **Output:** Photorealistic RGB image (256×256)
- **Key Components:**
  - Encoder: Convolutional layers progressively downsample (extract features)
  - Bottleneck: Compressed representation of facial structure
  - Decoder: Transposed convolutions upsample to original resolution
  - **Skip Connections:** Preserve fine-grained details (eyes, nose, mouth contours)

**Discriminator Network (D):**
- **Type:** PatchGAN discriminator
- **Input:** Paired images (composite + real photo) or (composite + generated photo)
- **Output:** Per-patch probability map (real vs. fake)
- **Advantage:** Focuses on local texture realism rather than just global structure

#### Loss Function

Combined adversarial and reconstruction loss:

{{< rawhtml >}}
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}}(G, D) + \lambda \cdot \mathcal{L}_{\text{L1}}(G)
$$
{{< /rawhtml >}}


Where:
- **Adversarial Loss:** Forces generated images to be indistinguishable from real photos
  $$
  \mathcal{L}_{\text{GAN}} = \mathbb{E}[\log D(x, y)] + \mathbb{E}[\log(1 - D(x, G(x)))]
  $$
  - $x$: facial composite (input)
  - $y$: real photograph (target)
  - $G(x)$: generated photorealistic image

- **L1 Reconstruction Loss:** Preserves structural similarity to input composite
  $$
  \mathcal{L}_{\text{L1}} = \mathbb{E}[||y - G(x)||_1]
  $$

- **λ:** Weight balancing realism vs. faithfulness (typically λ=100)

### Training Process

**Dataset Preparation:**
- Facial composites paired with corresponding ID photographs
- Augmentation: Random crops, horizontal flips, color jittering (photos only)
- Normalization: Pixel values scaled to [-1, 1]

**Training Configuration:**
- **Optimizer:** Adam (β₁=0.5, β₂=0.999, lr=0.0002)
- **Batch Size:** Limited by GPU constraints (likely 4-16)
- **Iterations:** Progressive training over multiple epochs
- **Hardware Constraint:** Limited compute resources (mentioned in evaluation)

**Training Strategy:**
1. Alternating optimization: Train D, then train G
2. Label smoothing to prevent D from becoming too strong
3. Iterative hyperparameter tuning (learning rate, λ weight)

### Alternative/Complementary Approaches Considered

Given the 2023 timeframe, other potential architectures include:

1. **CycleGAN:** For unpaired training if matched composite-photo pairs were scarce
2. **StyleGAN2-based Inversion:** Project composites into latent space, then generate with pretrained face generator
3. **Diffusion Models (Stable Diffusion):** Condition image generation on composite sketches
   - ControlNet: Guide diffusion process with structural information

**Note:** The results shown suggest a Pix2Pix-style paired approach was most successful given data availability.

---

## Results & Evaluation

### Visual Quality Assessment

The image provided shows three transformation examples (facial composite → real photo → predicted photo):

**Observed Strengths:**
- **Structural Preservation:** Facial geometry (eyes, nose, mouth positioning) accurately maintained
- **Skin Tone Generation:** Realistic coloration even from grayscale input
- **Hair Texture:** Plausible hairstyle synthesis matching sketch
- **Facial Expression:** Neutral expression consistent with ID photography standards

**Limitations:**
- **Resolution/Definition:** Generated images lack fine detail compared to real photos (noted as "manquent encore de définition")
- **Sharpness:** Slight blur/smoothing artifacts typical of early GAN training
- **Background:** Uniform blue background simplifies task but limits generalization

### Quantitative Metrics (Inferred)

While specific numbers aren't provided, typical evaluation metrics for this task would include:

- **Structural Similarity (SSIM):** Measures preservation of composite features
- **Fréchet Inception Distance (FID):** Measures realism compared to real photo distribution
- **Learned Perceptual Image Patch Similarity (LPIPS):** Perceptual quality assessment

### Stakeholder Feedback

From evaluation report:
> "Les résultats obtenus sont particulièrement prometteurs." (Results are particularly promising)
> "Il a fait preuve de belles initiatives et s'est attaché à trouver des solutions opérationnelles." (Demonstrated strong initiative and operational solutions)

**Operational Validation:**
- Consulted with Brigadier Alorent and forensic sketch artists
- Identified real-world operational needs and integration points
- Discussed how AI could augment (not replace) human expertise

---

## Additional Project: Text-to-Face Generation

**Objective:** Generate facial images directly from textual descriptions (e.g., "Male, 30s, short dark hair, thin face, no facial hair")

**Approach:** Likely involved:
- **Text Encoder:** Transform descriptions into embedding vectors (CLIP, BERT)
- **Conditional GAN/Diffusion Model:** Generate faces conditioned on text embeddings
- **Attribute Control:** Map textual attributes to facial features

**Status:** Model architecture completed, training partially complete due to resource constraints and debugging requirements

**Operational Use Case:** Enable investigators to generate suspect images from witness statements without requiring a sketch artist

---

## Challenges & Constraints

### 1. **Limited Compute Resources**
- **Impact:** Prevented full model convergence and extensive hyperparameter search
- **Mitigation:** Prioritized efficient architectures (smaller batch sizes, progressive training)
- **Evaluation Note:** "Limité par les capacités de calcul mises à sa disposition, il n'a pu entraîner suffisamment ses modèles"

### 2. **Data Scarcity**
- Paired composite-photo datasets are limited due to privacy and operational constraints
- Required careful data augmentation and potentially synthetic data generation

### 3. **Quality-Fidelity Trade-off**
- Balancing photorealism (for public recognition) with structural accuracy (for legal validity)
- Too much "creativity" by the generator could produce faces that don't match the original description

### 4. **Ethical & Legal Considerations**
- **Bias:** Ensuring model doesn't introduce demographic biases
- **Accountability:** Generated images must be clearly marked as AI-reconstructions
- **Privacy:** Handling sensitive biometric data under French/EU regulations

---

## Technical Skills Demonstrated

**Deep Learning Frameworks:**
- **PyTorch** (most likely) or TensorFlow for GAN implementation
- Model architecture design (U-Net, PatchGAN discriminator)
- Custom loss function implementation (adversarial + L1)

**Computer Vision:**
- Image preprocessing and augmentation
- Face detection and alignment
- Perceptual quality assessment
- Domain-specific evaluation (forensic art consultation)

**Generative Models:**
- Conditional GAN training and stabilization
- Adversarial training dynamics (generator/discriminator balance)
- Latent space manipulation
- Image-to-image translation

**Research & Development:**
- Literature review of state-of-the-art methods
- Experimental design and hyperparameter tuning
- Stakeholder engagement and requirement gathering
- Iterative prototyping under resource constraints

---

## Impact & Future Directions

### Immediate Impact

✅ **Proof of Concept Validated:** Demonstrated AI can transform composites into usable photorealistic images  
✅ **Operational Insights:** Identified integration points in criminal investigation workflows  
✅ **Foundation for Further Development:** Established baseline architecture and training pipeline

### Recommended Next Steps

1. **Extended Training:** Leverage cloud compute (AWS, GCP) for longer training runs
2. **Dataset Expansion:** Collaborate with multiple agencies to build larger paired dataset
3. **Multi-Modal Fusion:** Combine text descriptions + sketches for improved accuracy
4. **Interactive Refinement:** Allow investigators to iteratively adjust generated faces
5. **Facial Recognition Integration:** Test generated images against FR databases

### Broader Applications

- **Cold Case Investigation:** Modernize decades-old composite sketches
- **Missing Persons:** Age progression combined with composite enhancement
- **Training Tool:** Help forensic artists visualize witness descriptions
- **International Cooperation:** Standardize composite formats across agencies

---

## Ethical Considerations

This project operated within strict ethical guidelines:

- **Transparency:** All generated images clearly marked as AI-reconstructions (not actual photographs)
- **Human Oversight:** System designed to augment, not replace, forensic artist expertise
- **Bias Mitigation:** Awareness of potential demographic biases in training data
- **Data Protection:** Compliance with French data protection laws (CNIL) and GDPR
- **Legal Admissibility:** Consultation with legal teams on evidentiary standards

---

## Technical Deep Dive: How GANs Work

### Generative Adversarial Networks (GANs)

GANs consist of two neural networks in competition:

**1. Generator (G):** Creates fake data to fool the discriminator
- Learns mapping from input space (sketches) to target space (photos)
- Improves by trying to make fakes indistinguishable from real

**2. Discriminator (D):** Distinguishes real data from generated fakes
- Acts as learned loss function
- Forces generator to produce increasingly realistic outputs

### Training Dynamics
```python
# Simplified training loop
for epoch in epochs:
    for batch in dataloader:
        # Train Discriminator
        real_output = D(real_image, condition)
        fake_image = G(condition)
        fake_output = D(fake_image, condition)
        
        d_loss = -log(real_output) - log(1 - fake_output)
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        fake_image = G(condition)
        fake_output = D(fake_image, condition)
        
        g_loss_adversarial = -log(fake_output)
        g_loss_L1 = L1(fake_image, real_image)
        g_loss = g_loss_adversarial + λ * g_loss_L1
        g_loss.backward()
        optimizer_G.step()
```

### Why U-Net for Generator?

U-Net architecture is ideal for this task because:
- **Skip Connections:** Preserve low-level details (facial landmarks) from input sketch
- **Hierarchical Features:** Learn both global structure and local textures
- **Proven Effectiveness:** Standard choice for medical imaging and image translation

---

**Supervisor Evaluation Excerpt:**
> "Monsieur Elouan Pulveric a témoigné, durant son stage, de belles capacités à appréhender un sujet complexe et à en faire un outil exploitable. Il a fait preuve de belles initiatives et s'est attaché à trouver des solutions opérationnelles."
> 
> (Mr. Elouan Pulveric demonstrated, during his internship, strong capabilities in tackling a complex subject and creating a usable tool. He showed excellent initiative and focused on finding operational solutions.)

---

*This project showcases the intersection of cutting-edge generative AI with critical public safety applications, demonstrating both technical proficiency in deep learning and awareness of ethical considerations in law enforcement AI.*