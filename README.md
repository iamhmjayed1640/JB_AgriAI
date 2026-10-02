# 🌾 JB AgriAI: Attention-Enhanced Few-Shot Rice Disease Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

**JB AgriAI** is an explainable AI platform for precision agriculture. This repository contains the real-world deployment of our research on **Cross-Domain Rice Leaf Disease Classification** using an Attention-Enhanced Few-Shot Prototypical Network.

By combining Convolutional Block Attention Modules (CBAM) with Prototypical Networks, the system effectively learns disease representations from limited data and provides transparent, actionable diagnostics using Grad-CAM visualizations.

---

## ✨ Key Features

* **🧠 Few-Shot Learning:** Classifies new diseases from highly limited training samples using episodic meta-learning.
* **👁️ CBAM Attention:** Focuses on informative spatial lesion regions rather than treating the entire leaf equally.
* **🎯 Prototype Matching:** Utilizes Euclidean distance in a 640D embedding space to match query images against learned disease prototypes.
* **🔥 Explainable AI (XAI):** Integrated Grad-CAM generates real-time heatmaps to visually explain the model's decision-making process.
* **📄 Automated Reporting:** Generates downloadable, professional PDF diagnostic reports containing confidence scores, heatmaps, and agricultural guidance.

---

## ⚙️ Methodology & Architecture

The inference pipeline deployed in this repository follows a straightforward but powerful sequence:
1. **Input:** Raw RGB image of a rice leaf (captured via smartphone or field camera).
2. **Feature Extraction:** A customized `ResNet12` backbone processes the image.
3. **Attention Mechanism:** `CBAM` refines the feature maps, highlighting pathological lesions.
4. **Embedding Space:** The image is projected into a 640-dimensional feature space.
5. **Inference:** The embedding is compared against cached class prototypes using Euclidean distance, followed by a Softmax operation to yield confidence probabilities.

---

## 🚀 Installation & Local Setup

To run this project locally on your machine, follow these steps:

**1. Clone the repository:**
```bash
git clone https://github.com/iahmjayed1640/JB-AgriAI.git
cd JB-AgriAI
