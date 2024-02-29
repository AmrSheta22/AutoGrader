# AutoGrader: Revolutionizing Classic Handwriting Grading with AI

## Overview

Welcome to AutoGrader, a groundbreaking graduation project aimed at transforming traditional handwriting grading using Artificial Intelligence. This project, led by a team of 9 dedicated members, is divided into two main components: Handwriting Recognition and Automatic Grading. Each component has undergone extensive development to ensure accuracy, efficiency, and applicability in real-world scenarios.

## Handwriting Recognition Component

### Data Preparation

We meticulously prepared our data to meet the stringent requirements for efficient model training. The TrOCR and CRNN models emerged as potential options following a comprehensive analysis of available models and architectures.

### Model Training

The selected TrOCR model underwent training on both known and unknown datasets, with a primary focus on the IAM and EHD datasets. This critical step allowed us to evaluate its generalization performance using metrics such as Character Error Rate (CER) and Word Error Rate (WER).

### Model Optimization

Through in-depth error analysis, the TrOCR model underwent strategic adjustments to address higher mistake rates. Optimization efforts yielded varying results; some datasets, especially those with more diversity, saw a significant decrease in errors, while others witnessed improvements despite lacking diversity. This underscores the complexity of optimizing model performance and emphasizes the importance of representative and diverse data for reliable outcomes.

## Automatic Grading Component

### Dataset Compilation

We started the automatic grading component by carefully compiling relevant datasets from various reliable sources. These datasets included questions, student answers, model answers, and grades, forming the foundation for subsequent efforts.

### Data Cleaning and Preprocessing

Acquired data underwent thorough cleaning and preprocessing to ensure readiness for the next phases of the project.

### Data Analysis

Our analysis included both relational and univariate analyses, providing valuable insights guiding our subsequent actions.

### Model Exploration

In the following phase, we explored a diverse range of pre-trained models, including neural networks and mathematical models. A rigorous inspection and testing procedure was applied to each model using our data, aiming to identify a small number of excellent models that would serve as the foundation for the next stages of the project.

## Conclusion

AutoGrader, developed by a team of 9 dedicated members, represents a significant step forward in automating the grading process, leveraging advanced AI techniques for both handwriting recognition and automatic grading. The careful consideration of data quality, model selection, and continuous optimization reflects our commitment to achieving reliable and efficient outcomes. We invite you to explore and contribute to the evolution of handwriting grading with AutoGrader.
