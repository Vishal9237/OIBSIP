import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

csv_path = os.path.join(os.path.dirname(__file__), "Advertising.csv")
df = pd.read_csv(csv_path)

df = df.drop("Unnamed: 0", axis=1)

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

sns.pairplot(df)
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x="TV",y="Sales",data=df)
plt.title("TV vs Sales")
plt.show()

print("Observation")
print("Sales increase as TV advertising increases.")

plt.figure(figsize=(6,4))
sns.scatterplot(x="Radio",y="Sales",data=df)
plt.title("Radio vs Sales")
plt.show()

print("Observation")
print("Radio advertising also has a positive effect on sales.")

plt.figure(figsize=(6,4))
sns.scatterplot(x="Newspaper",y="Sales",data=df)
plt.title("Newspaper vs Sales")
plt.show()

print("Observation")
print("Newspaper advertising has a weaker relationship with sales.")

plt.figure(figsize=(6,5))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X=df[["TV","Radio","Newspaper"]]
y=df["Sales"]

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

lr=LinearRegression()

lr.fit(X_train,y_train)

pred1=lr.predict(X_test)

print("\nLinear Regression")

print("MAE:",mean_absolute_error(y_test,pred1))
print("RMSE:",mean_squared_error(y_test,pred1)**0.5)
print("R2 Score:",r2_score(y_test,pred1))

rf=RandomForestRegressor(random_state=42)

rf.fit(X_train,y_train)

pred2=rf.predict(X_test)

print("\nRandom Forest")

print("MAE:",mean_absolute_error(y_test,pred2))
print("RMSE:",mean_squared_error(y_test,pred2)**0.5)
print("R2 Score:",r2_score(y_test,pred2))

if r2_score(y_test,pred1)>r2_score(y_test,pred2):
    best_pred=pred1
    print("\nBest Model : Linear Regression")
else:
    best_pred=pred2
    print("\nBest Model : Random Forest")

residual=y_test-best_pred

plt.figure(figsize=(6,4))
plt.scatter(best_pred,residual)
plt.axhline(y=0,color="red")
plt.xlabel("Predicted Sales")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

importance=pd.Series(rf.feature_importances_,index=X.columns)

importance.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()

print("\nInterpretation")

print("TV has the highest impact on sales because it has the highest feature importance in the Random Forest model.")