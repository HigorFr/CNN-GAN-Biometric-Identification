# CNN-GAN Biometric Identification

Este repositório contém o projeto final da disciplina **Machine Learning** do curso de **Mestrado da Universidade de São Paulo (USP)**.

O trabalho consistiu na implementação de um sistema de **identificação biométrica**, formulado como um **problema de classificação multiclasse**, no qual o modelo recebe uma imagem facial e deve prever corretamente o rótulo correspondente à identidade do indivíduo.

---

## Abordagem

A solução foi implementada em **Python**, utilizando a biblioteca **TensorFlow**.

O modelo combina:

- **Redes Neurais Convolucionais (CNNs)** para extração de características;
- **Redes Adversariais Generativas (GANs)** para aprendizado de representações robustas.

O treinamento segue o paradigma adversarial:

\[
\min_G \max_D V(D, G)
\]

onde:

- \( G \) é o gerador,
- \( D \) é o discriminador.

O sistema final realiza classificação multiclasse a partir das representações aprendidas.

---

## Dataset

Foi utilizado o dataset **CelebA (CelebFaces Attributes Dataset)** para treinamento e avaliação do modelo.

---

## Ambiente de Treinamento

O treinamento foi realizado em ambiente doméstico, utilizando GPU dedicada.

---

## Estrutura do Projeto
├── data/ # Dados
├── models/ # Modelos treinados
├── src/ # Código-fonte
│ ├── model.py
│ ├── train.py
│ └── utils.py
├── requirements.txt
└── README.md

