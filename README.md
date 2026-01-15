# Inference-Time Scaling for Visual AutoRegressive Modeling by Searching Representative Samples

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](Coming_Soon) 
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](Coming_Soon)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Weidong Tang, Xinyan Wan, Siyu Li, and Xiumei Wang**

*School of Electronic Engineering, Xidian University*

</div>

---

## 📖 Abstract

While inference-time scaling has significantly enhanced generative quality in large language and diffusion models, its application to vector-quantized (VQ) visual autoregressive modeling (VAR) remains unexplored. We introduce **VAR-Scaling**, the first general framework for inference-time scaling in VAR.

We address the critical challenge of discrete latent spaces that prohibit continuous path search by mapping sampling spaces to quasi-continuous feature spaces via kernel density estimation (KDE). Our method employs a density-adaptive hybrid sampling strategy to optimize sample fidelity at critical scales.

![Motivation](assets/motivation.png)
*Figure 1: Comparison with Diffusion Models. While diffusion models enable metric-guided continuous search, VQ's discrete nature prevents this. Our KDE mapping transforms probability-scaled sampling spaces into continuous trajectories.*

## 💡 Methodology

We observe that VAR scales exhibit two distinct pattern types: **general patterns** (early stages) and **specific patterns** (later stages).

![Patterns](assets/patterns.png)
*Figure 2: Scales 0-1 define general patterns like spatial structures; scales 2-9 refine specific patterns. Our method targets these critical scales.*

**VAR-Scaling Strategy:**
1.  **Map** discrete samples to a quasi-continuous space using KDE.
2.  **Identify** high-density regions as representative prototypes.
3.  **Apply** Top-k sampling for high-density regions (quality) and Random-k for low-density regions (diversity).

## 🚀 News
- **[2024.08]** Code released! Support for both **Infinity** and **VAR** baselines.
- **[2024.xx]** Paper accepted to PRCV / Arxiv.

## 📂 Repository Structure

This repository contains the official implementation of VAR-Scaling applied to two baselines:

*   **[Infinity](./Infinity)**: Implementation on the high-resolution Infinity model (Text-to-Image).
*   **[VAR](./VAR)**: Implementation on the original VAR model (Class-Conditional).

## 🛠️ Installation & Usage (Infinity)

We recommend using **Anaconda** to manage the environment.

### 1. Environment Setup
```bash
cd Infinity
conda create -n infinity python=3.10
conda activate infinity

# Install PyTorch (Adjust CUDA version according to your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Dependencies
pip install -r requirements.txt

# Install MMDetection for GenEval (Required for evaluation)
pip install openmim
mim install mmcv-full==1.7.2
pip install mmdet==2.28.2