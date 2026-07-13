import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = os.path.join(os.path.dirname(__file__), "unemp_upto_11_2020.csv")
df = pd.read_csv(csv_path)

df.columns=df.columns.str.strip()

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

df["Date"]=pd.to_datetime(df["Date"],dayfirst=True)

region_avg=df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(10,5))
region_avg.sort_values().plot(kind="bar")
plt.title("Average Unemployment Rate by Region")
plt.ylabel("Unemployment Rate (%)")
plt.show()

print("Observation")
print("Some regions have higher unemployment than others.")

df=df.sort_values("Date")

month_avg=df.groupby(df["Date"].dt.to_period("M"))["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(10,5))
plt.plot(month_avg.index.astype(str),month_avg.values,marker="o")
plt.xticks(rotation=45)
plt.title("Month Wise Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate")
plt.show()

print("Observation")
print("Unemployment increased during the COVID period.")

plt.figure(figsize=(10,5))

state1=df[df["Region"]=="Maharashtra"]
state2=df[df["Region"]=="Karnataka"]
state3=df[df["Region"]=="Tamil Nadu"]

plt.plot(state1["Date"],state1["Estimated Unemployment Rate (%)"],label="Maharashtra")
plt.plot(state2["Date"],state2["Estimated Unemployment Rate (%)"],label="Karnataka")
plt.plot(state3["Date"],state3["Estimated Unemployment Rate (%)"],label="Tamil Nadu")

plt.legend()
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

print("Observation")
print("The unemployment rate is different for each state.")

top10=region_avg.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top10.values,y=top10.index)
plt.title("Top 10 States with Highest Unemployment")
plt.xlabel("Average Unemployment Rate")
plt.show()

print("Observation")
print("These are the states with the highest average unemployment.")

corr=df[["Estimated Unemployment Rate (%)",
"Estimated Employed",
"Estimated Labour Participation Rate (%)"]].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("Observation")
print("Employment and unemployment are negatively related.")

pre=df[df["Date"]<"2020-03-01"]
post=df[df["Date"]>="2020-03-01"]

pre_avg=pre["Estimated Unemployment Rate (%)"].mean()
post_avg=post["Estimated Unemployment Rate (%)"].mean()

print("Pre Covid Average:",pre_avg)
print("Post Covid Average:",post_avg)

plt.figure(figsize=(6,5))
plt.bar(["Pre Covid","Post Covid"],[pre_avg,post_avg])
plt.title("Pre Covid vs Post Covid")
plt.ylabel("Average Unemployment Rate")
plt.show()

print("Observation")
print("Average unemployment increased after COVID.")