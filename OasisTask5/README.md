# Email Spam Detection with Machine Learning

## Objective
The objective of this project is to build a machine learning model that classifies emails or SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques.

## Dataset
- **Dataset Name:** SMS Spam Collection Dataset
- **File Used:** `spam.csv`

## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- NLTK
- WordCloud
- Scikit-learn

## Machine Learning Models
- Multinomial Naive Bayes
- Logistic Regression

## Project Workflow
1. Load the spam dataset.
2. Remove unnecessary columns.
3. Clean and preprocess the text data.
4. Remove stopwords using NLTK.
5. Convert text into numerical features using TF-IDF Vectorization.
6. Split the dataset into training and testing sets.
7. Train Naive Bayes and Logistic Regression models.
8. Evaluate the models using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
9. Generate Word Clouds for Spam and Ham messages.
10. Compare model performance and identify the best model.

## Visualizations
- Spam Word Cloud
- Ham Word Cloud

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Conclusion
The project successfully classifies spam and ham messages using Natural Language Processing and Machine Learning techniques. TF-IDF vectorization effectively converts text into numerical features, and the trained models achieve high classification performance. Recall is particularly important because missing spam messages can allow unwanted emails to reach a user's inbox.

## Author
**Vishal G**
