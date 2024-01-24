Final Project

### Team : Temitope Ajose – Adeogun, Kaj Kabilan, Tamunosaki Miller, Esala kaluperuma, Kaiser

# Predicting Diabetes onset : A Machine learning Journey

## Table of Contents

* [Introduction](#Introduction)
* [Data Overview](#Data_Overview)
* [Cleaning and Handling the Data](#Data_Cleaning)
* [Dataset_Visualisations](Dataset_Visalisations)
* [Designing Models](#Designing-Models)
* [Results](#Results)
* [Challenges](#Challenges)
* [Conclusion](#Conclusion)
* [Refences](References)


## Introduction

We're on a mission to use medical info like age, gender, BMI, and blood glucose levels to predict if someone is likely to have diabetes. The dataset covers ages 18 to 75, with each person giving us important details.

<img width="400" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/9c862f4e-d654-4ba0-9188-4277147f2a3e">

We're exploring this data to find patterns that could help us spot diabetes early. By using simple and detailed info, we hope to make predictions that can be useful for both people and doctors. Our aim is to use data to make healthcare decisions that aren't just reactions but are based on information, helping both patients and healthcare providers.

So, let's dive into this medical data and see if we can find signs of diabetes and maybe even prevent it.

## Data_Overview

We defined our machine learning workflow in 4 stages:

* Data Collection & Preprocessing of Data
* Identifying Suitable Prediction Algorithms (Models)
* Model Training & Validation
* Performance Assessment (Results)


## Data_Cleaning
We read in the csv onto jupyter notebook in order to clean the data. The codes below shows how we removed any null values and 0 values. Our dataset was pretty clean as it was.

<img width="600" alt="cld" src="https://github.com/KajK0121/Project-4/assets/140313204/c2f2b910-c79d-44ce-97ef-177edc6dde8b">

We then exported the cleaned data to SQLite and tested the database in Jupyter Notebook
<img width="493" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/5a051fba-898f-4549-a771-aa4aaacf7370">
<img width="500" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/85040ac1-7382-4be7-b590-1626b1ae0a83">

## Dataset_Visalisations

Here are some images showcasing the various factors in our dataset and their characteristics. This is all plays a part in the prediction of diabetes.

### Distribution of HbA1c Level
<img width="600" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/14fe07ee-e75b-4bc9-b72a-5c3aefa76166">

The majority of the dataset falls between 2 and 6.5. A smaller proportion represents levels over 6.5, implying that a large proportion of the dataset is non-diabetic or prediabetic. The curved line shows the average level in the distribution.

### Age-Frequency Histogram & Gender Distribution Chart
<img width="1000" alt="collab" src="https://github.com/KajK0121/Project-4/assets/140313204/1d7237d7-94e3-4e4a-9f92-557f185cdfeb">
The histogram displays the distribution of ages in the dataset. The blue line represents a smoothed curve that illustrates the continuous distribution of ages in the data. Although there are outliers, all ages between 18 and 75 are well-represented.

Looking at the Gender Distribution Chart we can see females dominate the dataset. A negligible proportion of the individuals in the dataset did not mention their gender.

### BMI-Frequency Histogram & Distribution of Blood Glucose Chart
<img width="700" alt="22" src="https://github.com/KajK0121/Project-4/assets/140313204/bf00681b-0d2e-4303-a8e6-606004cd9316">

1) It is approximately normally distributed, although a single outlier bin appears in the centre. The curved line shows the smoothed estimation of BMI, indicating a peak at the centre due to the outlier
2) A large proportion of the dataset consists of blood glucose levels between 50 and 200. The proportion of the dataset with blood glucose levels over 200 is very small.

### Relationships

These 5 visualisations showcases the relationship between our variables and see if there is any correlation between them.

#### Scatter Plots
<img width="800" alt="4" src="https://github.com/KajK0121/Project-4/assets/140313204/01356a7d-0b4e-44e8-8e4c-693872a09d15">

The scatter plots of features show no visual correlation between each other, indicating no linear relationship.

#### Heatmap

<img width="800" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/a06042a0-f913-4706-bade-67421f7d19d8">

The heatmap provides detailed numerical values of correlation for each pair of features. The highest correlation is between blood glucose and diabetes, while the lowest correlation is between BMI and HbA1c. There are no negative correlations, but some correlations can be negligible.


## Designing-Models

### Data Preparation

In this stage, we carefully extracted the relevant feature variables by excluding the target variable 'diabetes,' preparing the dataset for subsequent model training. We isolated the key target variable 'diabetes' as it serves as a crucial element for supervised machine learning tasks, allowing us to predict and assess outcomes accurately. Moving forward, we executed data splitting, dividing the dataset into training and test sets. This step is essential to evaluate the model's performance on unseen data, ensuring its ability to generalise beyond the training set. To enhance the model's effectiveness, we implemented feature scaling, standardising the feature variables to maintain consistent scales. Additionally, we addressed the challenge of imbalanced data by strategically undersampling the majority class (non-diabetes) during class balancing, mitigating potential impacts on model performance.

<img width="600" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/ae566b25-7718-45d7-8318-ebdb73b6f421">

### Unbalanced Data Vs After Rebalancing
<img width="500" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/1ffe4b4c-d184-4816-8ab6-196c0de07c38">
<img width="500" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/b3eba345-a148-4109-926d-cbdc9c6e7c6d">

### Algorithms 

* [Logistic Regression](#Logistic-Regression)
* [The Random Forest Classifier](#The_Random_Forest_Classifier)
* [Support Vector machine](#Support_Vector_machine)
* [Decision Trees](#Decision_Trees)

#### Logistic Regression
<img width="800" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/56e90fdc-d301-4525-b953-32a066bcc007">

In the model training phase, we employed the Logistic Regression algorithm to train our predictive model, implementing a maximum iteration limit of 1000 to ensure effective convergence during the training process. Subsequently, we utilized the trained Logistic Regression model to generate predictions on the test dataset. Moving on to model evaluation, we thoroughly assessed the model's performance by calculating key metrics such as accuracy, confusion matrix, and a classification report. This comprehensive evaluation provided insights into the model's predictive capabilities and overall effectiveness. To facilitate further analysis and presentation, we meticulously documented the results, including model parameters and accuracy, in a DataFrame, offering a structured and accessible format for in-depth scrutiny and communication of our findings.

#### The Random Forest Classifier
<img width="800" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/e3f44502-6cc6-4854-8a59-d0493b24c22c">

In the advanced stage of model training, we opted for the Random Forest Classifier, leveraging its robust capabilities. To enhance model performance, we implemented hyperparameter tuning using GridSearchCV, systematically exploring combinations for the number of estimators, maximum depth, and maximum features. This process enabled us to pinpoint the optimal set of hyperparameters for our Random Forest model. Subsequently, we evaluated the model's performance on the test dataset, calculating crucial metrics such as accuracy, confusion matrix, and a comprehensive classification report. To consolidate and communicate our findings effectively, we documented the results, including the best-performing model's parameters and accuracy, in a DataFrame. This structured documentation serves as a valuable resource for further analysis and presentation, highlighting the key aspects of our optimized Random Forest model.

#### Support Vector machine

We then opted for the Support Vector Machine (SVM) algorithm to predict diabetes, leveraging its effectiveness in handling complex datasets. The model was trained on the designated training dataset using default parameters, laying the foundation for subsequent predictions. Utilizing the trained SVM model, we made predictions on the test dataset, obtaining outcomes for further evaluation.

For a comprehensive assessment of the SVM model's performance, we employed key metrics including accuracy, a confusion matrix, and a detailed classification report. These metrics provided valuable insights into the model's predictive capabilities and overall effectiveness in distinguishing between diabetes and non-diabetes cases.

To streamline and communicate our results effectively, we meticulously documented the details of the SVM model in the optimization results table. This documentation includes essential information such as the model name, parameters, and the achieved accuracy, creating a structured resource for deeper analysis and clear presentation of our SVM model's capabilities.

#### Decision Trees

In addition in the model training phase, we instantiated a Decision Tree model with a specified random state to ensure reproducibility. This model was then trained on resampled training data using the `fit` method, creating a foundation for subsequent predictions.

Moving on to prediction, we generated labels for the test dataset using the trained Decision Tree model, obtaining the predicted outcomes for further analysis.

For a robust assessment of model performance, we employed cross-validation on the resampled data. This approach provided a more reliable measure of the Decision Tree model's accuracy by computing the mean accuracy across multiple validation folds. Additionally, we calculated key metrics, including accuracy, a confusion matrix, and a detailed classification report, offering a comprehensive understanding of the model's predictive capabilities.

To document and communicate our results effectively, we recorded the details of the Decision Tree model in the optimization results table. This documentation encompasses crucial information such as the model name, parameters, and the achieved accuracy, serving as a structured resource for further analysis and transparent presentation of our Decision Tree model's performance.


## Results
When we assess how well our models are performing, we rely on two key tools: 
the Confusion Matrix and the Classification Report. These tools help us 
understand how our models are making predictions.

<img width="500" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/b7d622a9-621a-4db6-a6f1-2669d9893268">

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
<img width="500" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/943b9c87-4f93-48d9-938a-acd94900d7c4">
<img width="400" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/0d3138e2-8957-41b7-9573-c78789f6f4a1">

Now, let's talk about the Classification Report. It's another tool that gives us a 
detailed report card for our models, showing how they perform in various 
aspects.

#### Logistic Regression Accuracy
<img width="450" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/68b3be12-d291-436f-a881-ed86be9eb575">
<img width="450" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/1aeb37ef-b9c9-47a0-ad06-0d2886087436">

Our Logistic Regression model is about 87% accurate. This means it's right about
87% of the time when making predictions.
Random Forest Accuracy

#### The Random Forest model
<img width="450" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/33be12c0-f84c-43d3-90cd-5bfd6cd193d2">
<img width="450" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/80f3bf3e-b6d3-4ce9-abd9-8369230e229d">

The Random Forest model is even more accurate, at around 90%. This means it 
gets it right about 90% of the time.

### SVM Accuracy
<img width="450" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/9ec54598-9906-4355-9b7e-f5a0c4b99489">
<img width="450" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/7ae43c57-1763-41b2-b955-b9843f4bf78d">

SVM, another model, has an accuracy rate of about 85%. It's correct about 85% 
of the time.

### Decision Tree Accuracy
<img width="450" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/95d56fb1-7a52-4222-a5b0-f73180ea06de">
<img width="450" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/c15fdd3e-d9ce-4a84-a15f-85470459c670">

The Decision Tree model is quite accurate too, at around 88%. It's right about 
88% of the time.

### Random Forest and Decision Tree Precision and Recall
Looking closely at Random Forest and Decision Tree, they both do a great job in 
balancing precision, recall, and F1-Score, all at around 90%
Logistic Regression and Decision Tree Trade-off
On the other hand, Logistic Regression and Decision Tree, while not as accurate 
as Random Forest, show a balanced trade-off between precision and recall, with 
a weighted F1-Score of 87.75%. 


<img width="400" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/0355ba36-d565-447d-99cd-2ceb30eea9fd">
<img width="800" alt="image" src="https://github.com/KajK0121/Project-4/assets/140313204/6769a8d8-7b90-49e1-927b-8a8504be1c10">

### Summary of Model Performance
In summary, Random Forest is the most accurate, making it a top choice when 
accuracy is vital.Random Forest had the highest accuracy at 91%, while SVM had the lowest at 85%. Random Forest and Decision Tree also balance precision and 
recall well, making them effective for classifying various cases. Decision Tree and Logistic Regression achieved 87% and 86% accuracy which are the 2nd and 3rd in ranking.Overall, Random Forest performed best in accuracy, while Random Forest and Decision Tree demonstrated balanced precision and recall, suitable for classification tasks.

## Challenges
The dataset we used for building our models had an imbalance issue, with more records showing negative diabetes statuses. 
The dataset's large size slowed down processing, making code execution take longer.

## Conclusion

### For Next Time
Develop and intuitive and user –friendly interface for easy interaction with the diabetes prediction model.
Conduct through testing in a controlled environment  to validate model performance


## References
pandas.pydata.org. (n.d.). Styling — pandas 1.1.2 documentation. [online] Available at: https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html.

networkx.org. (n.d.). Gallery — NetworkX 3.2.1 documentation. [online] Available at: https://networkx.org/documentation/stable/auto_examples/index.html [Accessed 23 Jan. 2024].

Jason Brownlee (2018). What is a Confusion Matrix in Machine Learning. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/confusion-matrix-machine-learning/.

www.kaggle.com. (n.d.). Diabetes prediction dataset. [online] Available at: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data

