import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = iris.target_names[y]

print("First 5 Rows")
print(df.head())
print("\nDataset Shape")
print(df.shape)
print("\nData Types")
print(df.dtypes)
print("\nNull Values")
print(df.isnull().sum())
print("\nStatistical Summary")
print(df.describe())

sns.pairplot(df, hue="species")
plt.show()
plt.figure(figsize=(6,4))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
sns.boxplot(x="species", y="sepal length (cm)", data=df)
plt.title("Sepal Length")
plt.subplot(2,2,2)
sns.boxplot(x="species", y="sepal width (cm)", data=df)
plt.title("Sepal Width")
plt.subplot(2,2,3)
sns.boxplot(x="species", y="petal length (cm)", data=df)
plt.title("Petal Length")
plt.subplot(2,2,4)
sns.boxplot(x="species", y="petal width (cm)", data=df)
plt.title("Petal Width")
plt.tight_layout()
plt.show()

print("\nMost Discriminative Features")
print("Petal Length and Petal Width are the most useful features for classifying iris flowers.")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree")
print("Accuracy:", dt_acc)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_dt))
print("Classification Report")
print(classification_report(y_test,y_pred_dt,target_names=iris.target_names))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)
print("\nKNN")
print("Accuracy:", knn_acc)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_knn))
print("Classification Report")
print(classification_report(y_test,y_pred_knn,target_names=iris.target_names))

svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred_svm)
print("\nSVM")
print("Accuracy:", svm_acc)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_svm))
print("Classification Report")
print(classification_report(y_test,y_pred_svm,target_names=iris.target_names))

models = ["Decision Tree", "KNN", "SVM"]
accuracy = [dt_acc, knn_acc, svm_acc]

plt.figure(figsize=(6,4))
plt.bar(models, accuracy)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()

if dt_acc >= knn_acc and dt_acc >= svm_acc:
    print("\nBest Performing Model")
    print("Decision Tree")
    print("Accuracy:", dt_acc)

elif knn_acc >= dt_acc and knn_acc >= svm_acc:
    print("\nBest Performing Model")
    print("KNN")
    print("Accuracy:", knn_acc)

else:
    print("\nBest Performing Model")
    print("SVM")
    print("Accuracy:", svm_acc)

sample_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(sample_flower)
print("\nSample Prediction")
print("Flower Measurements:", sample_flower)
print("Predicted Species:", iris.target_names[prediction[0]])