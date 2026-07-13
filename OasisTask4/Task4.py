import os
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download("stopwords")

csv_path = os.path.join(os.path.dirname(__file__), "spam.csv")
df = pd.read_csv(csv_path, encoding="latin1")

df = df.iloc[:, :2]
df.columns = ["label", "message"]

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

print("\nSpam and Ham Count")
print(df["label"].value_counts())

print("\nPercentage")
print(df["label"].value_counts(normalize=True) * 100)

stop_words = set(stopwords.words("english"))

def clean(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z ]", "", text)
    words = text.split()
    new = []

    for word in words:
        if word not in stop_words:
            new.append(word)

    return " ".join(new)

df["message"] = df["message"].apply(clean)

print("\nTF-IDF gives importance to important words and converts text into numerical features.")

X = df["message"]
y = df["label"]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

nb = MultinomialNB()
nb.fit(X_train, y_train)

pred1 = nb.predict(X_test)

print("\nNaive Bayes Results")
print("Accuracy :", accuracy_score(y_test, pred1))
print("Precision:", precision_score(y_test, pred1, pos_label="spam"))
print("Recall   :", recall_score(y_test, pred1, pos_label="spam"))
print("F1 Score :", f1_score(y_test, pred1, pos_label="spam"))
print("Confusion Matrix")
print(confusion_matrix(y_test, pred1))

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

pred2 = lr.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy :", accuracy_score(y_test, pred2))
print("Precision:", precision_score(y_test, pred2, pos_label="spam"))
print("Recall   :", recall_score(y_test, pred2, pos_label="spam"))
print("F1 Score :", f1_score(y_test, pred2, pos_label="spam"))
print("Confusion Matrix")
print(confusion_matrix(y_test, pred2))

spam = " ".join(df[df["label"] == "spam"]["message"])

plt.figure(figsize=(8, 5))
wc = WordCloud(background_color="white")
plt.imshow(wc.generate(spam))
plt.axis("off")
plt.title("Spam Word Cloud")
plt.show()

ham = " ".join(df[df["label"] == "ham"]["message"])

plt.figure(figsize=(8, 5))
wc = WordCloud(background_color="white")
plt.imshow(wc.generate(ham))
plt.axis("off")
plt.title("Ham Word Cloud")
plt.show()

acc1 = accuracy_score(y_test, pred1)
acc2 = accuracy_score(y_test, pred2)

if acc1 > acc2:
    print("\nBest Performing Model: Naive Bayes")
    print("Accuracy:", acc1)
else:
    print("\nBest Performing Model: Logistic Regression")
    print("Accuracy:", acc2)

print("\nConclusion")
print("Recall is an important metric because a low recall means spam messages may be classified as normal messages and reach the user's inbox.")