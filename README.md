# CNN-GAN Biometric Identification

This repository contains the final project for the **Machine Learning** course in the **Master’s program at the University of São Paulo (USP)**.

The objective was to implement a **biometric identification system**, formulated as a **multiclass classification problem**, where the model receives a facial image and predicts the corresponding identity label.

---

## Approach

The solution was implemented in **Python** using **TensorFlow**.

The architecture combines:

- **Convolutional Neural Networks (CNNs)** for feature extraction  
- **Generative Adversarial Networks (GANs)** for adversarial representation learning  

Training follows the adversarial optimization framework:

\[
\min_G \max_D V(D, G)
\]

where:

- \( G \) is the generator  
- \( D \) is the discriminator  

The learned representations are used to perform multiclass identity classification.

---

## Dataset

The model was trained and evaluated using the **CelebA (CelebFaces Attributes Dataset)**.

---

## Training Environment

Training was conducted in a home environment using a dedicated GPU.

---



