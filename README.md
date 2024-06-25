# AutoGrader: Revolutionizing Classic Handwriting Grading with AI

## Overview of the Project

Welcome to AutoGrader, a groundbreaking graduation project aimed at transforming traditional handwriting grading using Artificial Intelligence. This project, led by a team of 9 dedicated members, is divided into two main components: Handwriting Recognition and Automatic Grading. Each component has undergone extensive development to ensure accuracy, efficiency, and applicability in real-world scenarios.

This project focuses on developing robust models for handwriting recognition and automatic short answer grading (ASAG). We utilize various datasets and advanced preprocessing techniques to enhance the quality of handwritten text recognition and grading models. The primary goal is to improve the accuracy and efficiency of recognizing and grading handwritten content in educational settings.

## Handwriting Recognition Component
Handwriting recognition involves converting handwritten text into machine-readable text. Our approach includes multiple phases:

### Handwriting Recognition Phases
1. **Data Collection**: 
   - We collected data from IAM Handwriting Database 3.0 dataset to ensure comprehensive training.
   - For testing we used two datasets which are: Egyptian Handwriting Dataset (EHD), and CVL Database.
2. **Image Pre-Processing**: 
   - **Image Binarization**: Converts grayscale images to binary, enhancing feature extraction.
   - **Noise Removal**: Eliminates unwanted artifacts to improve image clarity.
   - **Dilation**: Enhances stroke width in handwritten images.
   - **Deskewing**: Aligns handwritten text horizontally.
   - **Deslanting**: Corrects the slant in handwritten text.
   - **Rescaling**: Standardizes image dimensions.
   - **Image Inversion**: Enhances feature contrast by inverting image colors.
3. **Evaluation Metrics**: 
  Our evaluation framework incorporates a range of metrics to assess the performance of our handwriting recognition models:
   - Model Inference
   - Test Data Preparation
   - Comparison of Images
   - Comparison of Labels
   - Vector Inputs
   - Generation of Predictions
   - Decoding Predictions
   - Comparison with Ground Truth
   - Character Error Rate (CER)
  
4. **Models Training**: 
   - We used a pre-trained model which is Tr-OCR for training.
5. **Model Optimization**: 
   - Fine-tuning models to enhance performance and accuracy.
6. **Error Analysis**: 
   - Identifying and analyzing common mistakes to improve model robustness.

## Automatic Short Answer Grading (ASAG) Component
ASAG involves grading handwritten or typed short answers automatically. Our approach includes multiple phases:

### ASAG Phases
1. **Data Collection**: 
   - Collecting datasets of short answers and including four columns (Question, Model Answer, Student Answer,and Grade).
   - We used datasets like: Mohler, PT-ASAG, and AR-ASAG for training and testing.
2. **Data Pre-Processing**:
   - Exploratory Data Analysis
   - Cleaning and preparing the text data for analysis.
   - Column Analysis
4. **Data Analysis**: 
   - Extracting relevant features from the text and understanding patterns.
   - Analysis was divided into two types: Single Column Analysis and Relations between Columns Analysis.
5. **Model Training**: 
   - Trying different pre-trained models that utilizes Semantic Similarity for grading like Knowledge-based models, Sentence Transformers, and BERT.
6. **Model Optimization**: 
   - Adjusting model parameters to enhance grading accuracy.
7. **Error Analysis**: 
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
  

### Website Demo
#### 1) Admin Login
![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/9d284032-c13f-464f-9e37-4680f1ef8b3c)

#### 2) Admin Registers Professors' Accounts
![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/249a7a93-6227-4cf2-9d0d-103e8e449a59)

#### 3) Professors Login
![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/7f97dc01-7b0a-4e07-bba8-1d0d599ded6d)

#### 4) Professors are Redirected to the Home Page
![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/5f0582d0-0ead-445c-8c7e-476069734d70)

#### 5) Professors Grade by uploading Student Answer and Model Answer
![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/d30c8341-1be0-4f69-9262-c4714d2eadd9)

#### 6) User Redirected to Grade Dashboard
![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/eb770b9f-28bb-4b70-bf63-2a642246aaf1)

![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/58a7adcf-dd84-40fb-8778-c072c803ef63)

![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/5c4f8dc9-badc-4708-82ce-9e34f56d2a37)

#### 7) Admins view of the dashboard
![image](https://github.com/AmrSheta22/AutoGrader/assets/78879424/c439df1d-5d60-41ac-b399-f038ab27c134)

## Files and Folder Structure
The project is organized into the following directories and files:

- **data**: Contains the datasets used for training and evaluation.
- **handwriting_recognition/data_collection**: Scripts and tools for collecting handwriting samples.
- **notebooks**: Python notebooks for experiments, model training, and evaluation.
- **scr**: Source code for the project, including models, preprocessing scripts, and utilities.
- **README.md**: This file, providing an overview of the project.

## Conclusion
The development of advanced models for handwriting recognition and automatic short answer grading (ASAG) represents a significant step forward in educational technology. By meticulously collecting diverse datasets and employing sophisticated preprocessing techniques, our models achieve high accuracy and reliability. These models can revolutionize the way educational assessments are conducted, offering fast and consistent grading of handwritten and short answer responses.

Our user-friendly website further demonstrates the practical applications of these models, allowing users to experience the capabilities firsthand. With real-time feedback and intuitive interfaces, the website showcases the potential for widespread adoption in educational settings.
