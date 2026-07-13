import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

csv_path = os.path.join(os.path.dirname(__file__), "CAR DETAILS FROM CAR DEKHO.csv")
df = pd.read_csv(csv_path)

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

df = df.drop_duplicates()

df["fuel"] = df["fuel"].str.lower()
df["seller_type"] = df["seller_type"].str.lower()
df["transmission"] = df["transmission"].str.lower()
df["owner"] = df["owner"].str.lower()

current_year = datetime.now().year
df["car_age"] = current_year - df["year"]

df["brand"] = df["name"].str.split().str[0]

print(df.describe())

plt.figure(figsize=(8, 5))
sns.histplot(df["selling_price"], bins=30, kde=True)
plt.title("Selling Price Distribution")
plt.show()

print("Observation")
print("Most cars are sold at lower prices.")

plt.figure(figsize=(8, 5))
sns.boxplot(x="fuel", y="selling_price", data=df)
plt.title("Selling Price by Fuel Type")
plt.show()

print("Observation")
print("Selling price changes depending on fuel type.")

plt.figure(figsize=(8, 5))
sns.scatterplot(x="car_age", y="selling_price", data=df)
plt.title("Car Age vs Selling Price")
plt.show()

print("Observation")
print("Older cars usually have lower selling prices.")

le = LabelEncoder()

df["fuel"] = le.fit_transform(df["fuel"])
df["seller_type"] = le.fit_transform(df["seller_type"])
df["transmission"] = le.fit_transform(df["transmission"])
df["owner"] = le.fit_transform(df["owner"])
df["brand"] = le.fit_transform(df["brand"])

corr = df[["selling_price","km_driven","car_age","fuel","transmission",]].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X = df[["km_driven","fuel","seller_type","transmission","owner","car_age","brand",]]
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLinear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", mean_squared_error(y_test, y_pred_lr) ** 0.5)
print("R2 Score:", r2_score(y_test, y_pred_lr))

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", mean_squared_error(y_test, y_pred_rf) ** 0.5)
print("R2 Score:", r2_score(y_test, y_pred_rf))

if r2_score(y_test, y_pred_lr) > r2_score(y_test, y_pred_rf):
    print("\nBest Model : Linear Regression")
else:
    print("\nBest Model : Random Forest")

importance = pd.Series(rf.feature_importances_, index=X.columns)

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()