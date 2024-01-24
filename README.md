# Project-4
Final Project

### Team : Temitope Ajose – Adeogun, Kaj Kabilan, Tamunosaki Miller, Esala kaluperuma, Kaiser

# Predicting Diabetes onset : A Machine learning Journey

## Table of Contents

* [Introduction](#Introduction)
* [Data Overview](#Data_Overview)
* [Cleaning and Handling the Data](#Data_Cleaning)
* [Designing Models](#Designing-Models)
* [Results](#Results)
* [Challenges](#Challenges)
* [Conclusion](#Conclusion)


## Introduction

We're on a mission to use medical info like age, gender, BMI, and blood glucose levels to predict if someone is likely to have diabetes. The dataset covers ages 18 to 75, with each person giving us important details.

We're exploring this data to find patterns that could help us spot diabetes early. By using simple and detailed info, we hope to make predictions that can be useful for both people and doctors. Our aim is to use data to make healthcare decisions that aren't just reactions but are based on information, helping both patients and healthcare providers.

So, let's dive into this medical data and see if we can find signs of diabetes and maybe even prevent it.

## Data_Overview

## Data_Cleaning
We read in the csv onto jupyter notebook in order to clean the data. The codes below shows how we removed any null values and 0 values. Our dataset was pretty clean as it was.

<img width="600" alt="cld" src="https://github.com/KajK0121/Project-4/assets/140313204/c2f2b910-c79d-44ce-97ef-177edc6dde8b">

We then exported the cleaned data to SQLite and tested the database in Jupyter Notebook
<img width="493" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/5a051fba-898f-4549-a771-aa4aaacf7370">
<img width="500" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/85040ac1-7382-4be7-b590-1626b1ae0a83">

## Designing-Models
<img width="500" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/76b68c3d-3b6a-4af7-8c36-97f3d89e1b20">

<img width="600" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/ae566b25-7718-45d7-8318-ebdb73b6f421">

### Algorithms 

* [Logistic Regression](#Logistic-Regression)
* [The Random Forest Classifier](#The_Random_Forest_Classifier)
* [Support Vector machine](#Support_Vector_machine)
* [Decision Trees](#Decision_Trees)

#### Logistic Regression


#### The Random Forest Classifier


#### Support Vector machine


#### Decision Trees



## Results
When we assess how well our models are performing, we rely on two key tools: 
the Confusion Matrix and the Classification Report. These tools help us 
understand how our models are making predictions.

#### True Negatives (TN)
True Negatives, represented as TN, tell us how many times our model correctly 
said something was 'negative' when it actually was. Think of it as when the 
model gets it right in saying 'no.'

#### False Positives (FP)
False Positives, or FP, are when our model incorrectly says something is 'positive'
when it's not. It's like a false alarm when it should have been 'no.'

#### False Negatives (FN)
False Negatives, FN, happen when our model wrongly says something is 
'negative' when it's actually 'positive.' It's like missing something important when
it's actually there.

### Introduction to Classification Report
Now, let's talk about the Classification Report. It's another tool that gives us a 
detailed report card for our models, showing how they perform in various 
aspects.

#### Logistic Regression Accuracy
Our Logistic Regression model is about 87% accurate. This means it's right about
87% of the time when making predictions.
Random Forest Accuracy

#### The Random Forest model
The Random Forest model is even more accurate, at around 90%. This means it 
gets it right about 90% of the time.

### SVM Accuracy
SVM, another model, has an accuracy rate of about 85%. It's correct about 85% 
of the time.

### Decision Tree Accuracy
The Decision Tree model is quite accurate too, at around 88%. It's right about 
88% of the time.

### Random Forest and Decision Tree Precision and Recall
Looking closely at Random Forest and Decision Tree, they both do a great job in 
balancing precision, recall, and F1-Score, all at around 90%
Logistic Regression and Decision Tree Trade-off
On the other hand, Logistic Regression and Decision Tree, while not as accurate 
as Random Forest, show a balanced trade-off between precision and recall, with 
a weighted F1-Score of 87.75%. 

### Summary of Model Performance
In summary, Random Forest is the most accurate, making it a top choice when 
accuracy is vital. Random Forest and Decision Tree also balance precision and 
recall well, making them effective for classifying various cases. We choose the 
model that suits our project's specific needs

## Challenges

## Conclusion
