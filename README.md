🎓 Student Academic Performance Prediction

A beginner Machine Learning project that uses Logistic Regression to predict whether a student is likely to pass or fail based on academic performance indicators.

This project explores the fundamentals of supervised machine learning, including data generation, preprocessing, model training, evaluation, prediction and data visualisation.

📌 Project Overview

The goal of this project is to demonstrate how Machine Learning can be applied in education to identify patterns in student academic performance.

The model analyses four student-related features:

Study hours per week

Class attendance percentage

Previous assessment grades

Number of completed assignments

Using these features, the Logistic Regression model learns patterns from historical (synthetically generated) student data and predicts a binary outcome:

1 – Pass

0 – Fail

The project is an educational demonstration and does not use real student academic records.

🎯 Project Objectives

Understand the fundamentals of supervised machine learning.

Explore and generate datasets using Python.

Understand classification problems and binary outcomes.

Implement a Logistic Regression model using Scikit-learn.

Train and test a machine learning model.

Evaluate model performance using classification metrics.

Visualise data distributions and model predictions.

Understand how machine learning can support educational decision-making.

🛠️ Technologies & Libraries

Technology

Purpose

Python

Main programming language

Pandas

Data manipulation and analysis

NumPy

Numerical operations and synthetic data generation

Scikit-learn

Machine learning model training and evaluation

Matplotlib

Data visualisation

Seaborn

Statistical data visualisation

🧠 Machine Learning Approach

This project implements Supervised Learning, specifically a Binary Classification problem.

1. Data Generation

The project generates a synthetic dataset containing 1,000 student records.

Each record includes:

Feature

Description

study_hours

Hours spent studying per week

attendance

Student attendance percentage

previous_grades

Previous assessment performance

assignments_completed

Number of completed assignments

result

Predicted target category (Pass/Fail)

A weighted scoring mechanism and sigmoid probability function are used to generate the student outcomes.

2. Data Preprocessing

The dataset is separated into:

Features (X): Student academic performance indicators.

Target (y): Student pass/fail outcome.

The data is divided into:

80% Training data

20% Testing data

A StandardScaler is used to standardise the input features before model training.

3. Model Training

The project uses Logistic Regression from Scikit-learn.

Logistic Regression is a classification algorithm that estimates the probability of an outcome belonging to a particular class.

The model uses a sigmoid function to convert its calculated score into a probability between 0 and 1.

4. Model Evaluation

The trained model is evaluated using:

Accuracy Score

Classification Report

Confusion Matrix

These metrics help assess how effectively the model classifies student academic outcomes.

5. Student Prediction

After training, the model is used to predict the academic outcome of a new student using the following example:

Feature

Value

Study hours

8

Attendance

85%

Previous grades

72%

Assignments completed

5

The model returns:

Predicted academic outcome (Pass/Fail).

Probability of passing.

📊 Data Visualisations

The project includes several visualisations to help understand the dataset and Logistic Regression.

Pass/Fail Distribution: Displays the distribution of students across the two outcome categories.

Study Hours vs Previous Grades: Explores the relationship between study hours and previous assessment performance.

Attendance Distribution: Compares attendance patterns between students who pass and fail.

Sigmoid Function: Demonstrates how Logistic Regression converts a linear score into a probability.

Decision Boundary: Visualises how the model separates the two classification categories.

Feature Coefficients: Shows the influence and direction of the model's input features.

📁 Project Structure

First-ML-project/
│
├── main.py
├── student_data.csv
└── README.md

⚙️ Installation & Setup

Prerequisites

Make sure you have installed:

Python 3.10 or later

pip (Python package manager)

Git

1. Clone the repository

git clone https://github.com/mahlori050207/First-ML-project.git

2. Navigate into the project directory

cd First-ML-project

3. Create a virtual environment

python -m venv venv

Activate the environment:

Windows:

venv\Scripts\activate

Linux / macOS:

source venv/bin/activate

4. Install dependencies

pip install numpy pandas matplotlib seaborn scikit-learn

5. Run the project

python main.py

The program will generate the synthetic dataset, train the Logistic Regression model, display evaluation results and generate visualisations.

🚀 Future Improvements

Some potential improvements for this project include:

Experimenting with Decision Tree classification.

Comparing Logistic Regression with Random Forest.

Introducing TensorFlow and PyTorch.

Using real-world anonymised academic datasets.

Saving trained models for future predictions.

Building an interactive student performance prediction application.

Exploring feature engineering and hyperparameter tuning.

📚 What I Learned

Through this project, I explored the foundations of Machine Learning and gained practical experience with:

Supervised learning and classification.

Dataset generation and preprocessing.

Training and testing machine learning models.

Logistic Regression and the sigmoid function.

Model evaluation and performance metrics.

Understanding decision boundaries.

Visualising machine learning results.

This project represents an important step in my journey towards becoming an AI and Machine Learning Engineer.

👨‍💻 Author

Mahlori Baloyi

GitHub: @mahlori050207
