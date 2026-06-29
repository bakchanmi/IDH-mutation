# Spatio-Genomic Multi-Scale Representation Learning Framework for Multimodal IDH Mutation Prediction

## Overview

This repository contains the implementation of a multimodal deep learning framework for predicting **Isocitrate Dehydrogenase (IDH) mutation status** in glioma patients by jointly leveraging **brain MRI** and **genomic information**.

Unlike conventional unimodal approaches, this framework integrates imaging-derived spatial representations with genomic features through a multi-scale representation learning strategy, enabling complementary information from both modalities to improve prediction performance.

## Motivation

Accurate preoperative prediction of IDH mutation status is clinically important for glioma diagnosis, prognosis, and treatment planning. Although MRI provides rich spatial information about tumor morphology, it cannot fully capture the molecular characteristics of tumors. Conversely, genomic biomarkers contain molecular information but lack spatial context.

To address these limitations, this study proposes a multimodal framework that combines MRI and genomic features to learn more discriminative representations for IDH mutation prediction.

## Proposed Framework

The proposed framework consists of three major components:

* **MRI Feature Extraction**

  * CNN-based feature extraction from multi-modal brain MRI
  * Representation learning from tumor regions

* **Genomic Feature Extraction**

  * Processing genomic features including molecular information
  * Feature embedding for multimodal integration

* **Multimodal Fusion**

  * Late-fusion strategy for integrating imaging and genomic representations
  * End-to-end IDH mutation prediction

## Representation Visualization (t-SNE)

The t-SNE visualization illustrates the learned feature distributions of different feature representations. The proposed multimodal representation achieves a clearer separation between IDH-mutant and IDH-wildtype samples compared with unimodal feature representations.

<p align="center">
<img src="figures/tSNE.pdf" width="650">
</p>

## Repository Structure

```text
fusion/
genomics/
mri/
train_mri.py
train_genomic.py
train_fusion.py
preprocess_genomics.py
export_cnv_features.py
```

## Dataset

The dataset used in this study is **not included** due to data sharing restrictions.
