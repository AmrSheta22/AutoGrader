# AutoGrader: Revolutionizing Classic Handwriting Grading with AI

## Overview

Welcome to AutoGrader, a groundbreaking graduation project aimed at transforming traditional handwriting grading using Artificial Intelligence. This project, led by a team of 9 dedicated members, is divided into two main components: Handwriting Recognition and Automatic Grading. Each component has undergone extensive development to ensure accuracy, efficiency, and applicability in real-world scenarios.

This project focuses on developing robust models for handwriting recognition and automatic short answer grading (ASAG). We utilize various datasets and advanced preprocessing techniques to enhance the quality of handwritten text recognition and grading models. The primary goal is to improve the accuracy and efficiency of recognizing and grading handwritten content in educational settings.

## Handwriting Recognition Component
Handwriting recognition involves converting handwritten text into machine-readable text. Our approach includes multiple phases:

### Handwriting Recognition Phases
1. **Data Collection**: 
   - We collect data from a variety of datasets to ensure a diverse and comprehensive training set.
2. **Image Pre-Processing**: 
   - **Image Binarization**: Converts grayscale images to binary, enhancing feature extraction.
   - **Noise Removal**: Eliminates unwanted artifacts to improve image clarity.
   - **Dilation**: Enhances stroke width in handwritten images.
   - **Deskewing**: Aligns handwritten text horizontally.
   - **Deslanting**: Corrects the slant in handwritten text.
   - **Rescaling**: Standardizes image dimensions.
   - **Image Inversion**: Enhances feature contrast by inverting image colors (optional).
3. **Evaluation Metrics**: 
   - We utilize standard metrics to evaluate the performance of our handwriting recognition models.
4. **Models Training**: 
   - We train models using state-of-the-art techniques and diverse datasets.
5. **Model Optimization**: 
   - Fine-tuning models to enhance performance and accuracy.
6. **Error Analysis**: 
   - Identifying and analyzing common mistakes to improve model robustness.

## Automatic Short Answer Grading (ASAG) Component
ASAG involves grading handwritten or typed short answers automatically. Our approach includes multiple phases:

### ASAG Phases
1. **Data Collection**: 
   - Collecting a dataset of short answers.
2. **Data Pre-Processing**: 
   - Cleaning and preparing the text data for analysis.
3. **Data Analysis**: 
   - Extracting relevant features from the text and understanding patterns.
4. **Model Training**: 
   - Training models using supervised learning techniques.
5. **Model Optimization**: 
   - Adjusting model parameters to enhance grading accuracy.
6. **Error Analysis**: 
   - Conducting thorough error analysis to identify areas for improvement.

## Website and Demo
To demonstrate the capabilities of our handwriting recognition and ASAG models, we have developed a user-friendly website. The website allows users to upload handwritten text and receive recognition results, as well as submit short answers for automatic grading. 

### Website Phases
1. **Frontend**: 
   - Designing a user-friendly interface for easy interaction.
2. **Backend**: 
   - Developing the server-side logic to handle data processing and model inference.
3. **Computer Vision and Customizing Exam Sheets**: 
   - Implementing computer vision techniques for recognizing handwritten text and customizing exam sheets.

### Features
- **Upload Handwritten Text**: Users can upload images of handwritten text for recognition.
- **Submit Short Answers**: Users can submit short answers for automatic grading.
- **Real-Time Feedback**: Instant results and feedback on handwriting recognition and grading.
  

### Screenshots from the Website
![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/eb770b9f-28bb-4b70-bf63-2a642246aaf1)

![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/58a7adcf-dd84-40fb-8778-c072c803ef63)

![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/5c4f8dc9-badc-4708-82ce-9e34f56d2a37)


## Conclusion
This project aims to enhance the accuracy and efficiency of handwriting recognition and automatic short answer grading. By leveraging diverse datasets and advanced preprocessing techniques, we strive to develop models that can significantly improve educational assessment processes.
