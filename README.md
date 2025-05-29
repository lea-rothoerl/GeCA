# Advancing Mammographic Image Generation Using Generative Cellular Automata for Improved CAD Systems
*A Master Thesis by Lea Rothörl, Master Biomedical Data Science*

![MA_PL_FI](https://github.com/user-attachments/assets/b590ca55-6185-4e44-84ef-683154532f2a)


This repository contains the code for my master thesis on mammographic image generation using Generative Cellular Automata (GeCA). This work builds substantially on the original [GeCA framework](https://github.com/xmed-lab/GeCA/) proposed by **Elbatel, Kamnitsas, and Li (2024)**.

---

## Background

In recent years, the application of generative models for synthetic expansion of medical image datasets has made significant strides, holding promise for the advancement of computer-aided diagnosis (CAD) systems.

One such approach, the **Generative Cellular Automata (GeCA) framework**, uses neural cellular automata for generative image synthesis. The original GeCA framework was proposed by **Elbatel, Kamnitsas, and Li (2024)**, and is available [here](https://github.com/xmed-lab/GeCA/).  

---

## Project Overview

This project explores the application of the GeCA framework for the synthesis of **mammographic images**. Specifically, the framework was adapted to:

- Handle mammograms from the **VinDr-Mammo dataset**, both as full images and extracted findings.
- Provide a **highly customizable** data pipeline for image generation and evaluation.
- Integrate a **breast density classifier** to assess the potential of generated images for improving a practical downstream classification task.

---

## Results Summary

- The adapted GeCA model was generally capable of generating **perceptually realistic mammograms in a low-resolution setting**.
- In breast density classification experiments, augmenting underrepresented density classes with synthetic images **improved macro F1-scores** — though effects on **binary metrics were ambivalent**.
- These findings imply the potential of GeCA-generated images for **dataset balancing** in classification tasks, warranting further exploration.

---

## Citation and Original Work

This project is a derivative of the original **GeCA framework**:

> Elbatel, A., Kamnitsas, K., & Li, W. (2024). *An Organism Starts with a Single Pix-Cell: A Neural Cellular Diffusion for High-Resolution Image Synthesis*  
> Original implementation: [https://github.com/xmed-lab/GeCA/](https://github.com/xmed-lab/GeCA/)

If you use this adapted code or findings, please also cite the original authors accordingly.
