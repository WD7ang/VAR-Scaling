# Inference-Time Scaling for Visual AutoRegressive Modeling by Searching Representative Samples

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Arxiv:2601.07293-b31b1b.svg)](https://arxiv.org/abs/2601.07293)
[![Conference](https://img.shields.io/badge/Conference-PRCV_2025-003399.svg)](https://link.springer.com/chapter/10.1007/978-981-95-5699-1_28)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Weidong Tang, Xinyan Wan, Siyu Li, and Xiumei Wang**

*School of Electronic Engineering, Xidian University*

</div>

<p align="center">
  <img src="assets/qualitative.png" width="100%">
</p>

---

## 📖 Abstract

While inference-time scaling has significantly enhanced generative quality in large language and diffusion models, its application to vector-quantized (VQ) visual autoregressive modeling (VAR) remains unexplored. We introduce **VAR-Scaling**, the first general framework for inference-time scaling in VAR.

We address the critical challenge of discrete latent spaces that prohibit continuous path search by mapping sampling spaces to quasi-continuous feature spaces via kernel density estimation (KDE). Our method employs a density-adaptive hybrid sampling strategy to optimize sample fidelity at critical scales.

<div align="center">
  <img src="assets/motivation.png" width="70%">
  <p><em>Figure 1: Comparison with Diffusion Models. While diffusion models enable metric-guided continuous search, VQ's discrete nature prevents this. Our KDE mapping transforms probability-scaled sampling spaces into continuous trajectories.</em></p>
</div>

## 💡 Methodology & Analysis

We observe that VAR scales exhibit two distinct pattern types: **general patterns** (early stages) and **specific patterns** (later stages).

### Density Analysis
Through extensive experiments, we explored the correlation between sample density in the quasi-continuous space and generation quality.

<div align="center">
  <img src="assets/explore.png" width="70%">
  <p><em>Figure 2: Experimental analysis of sample density. We find that high-density regions correspond to high-quality representative prototypes (general patterns), whereas low-density samples often result in outliers or diverse variations.</em></p>
</div>

**VAR-Scaling Strategy:**
1.  **Map** discrete samples to a quasi-continuous space using KDE.
2.  **Identify** high-density regions as representative prototypes.
3.  **Apply** Top-k sampling for high-density regions (quality) and Random-k for low-density regions (diversity).

## 🚀 News
- **[2026.01.16]** Code released! Support for both **Infinity** and **VAR** baselines.
- **[2025.08.23]** Paper accepted to **PRCV 2025**!

## 📂 Repository Structure

This repository contains the official implementation of VAR-Scaling applied to two strong baselines:

*   **[Infinity](./Infinity)**: Implementation on the high-resolution Infinity model (Text-to-Image).
*   **[VAR](./VAR)**: Implementation on the original VAR model (Class-Conditional, ImageNet).

## 🛠️ Installation

To ensure full compatibility and reproducibility, we recommend following the official installation guides for the respective base models.

*   **For VAR:** Please refer to [FoundationVision/VAR](https://github.com/FoundationVision/VAR).
*   **For Infinity:** Please refer to [FoundationVision/Infinity](https://github.com/FoundationVision/Infinity).

Once you have set up the environment for the base model, simply navigate to the corresponding folder in this repository (`./VAR` or `./Infinity`) and run our provided scripts.

## ⚡ Usage

### 1. Infinity (Text-to-Image)

We provide a comprehensive script `eval.sh` to reproduce our results on the Infinity model.

**Step 1: Prepare Weights**
Download the required weights (`infinity_2b_reg.pth`, `infinity_vae_d32reg.pth`, `flan-t5-xl`) and place them in `Infinity/weights/`.

**Step 2: Run Inference**

```bash
cd Infinity

# Run the evaluation script with VAR-Scaling enabled
bash eval.sh