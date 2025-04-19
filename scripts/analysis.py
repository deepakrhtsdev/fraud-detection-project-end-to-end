import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/preprocessed_data.csv")

print(" First 5 rows: ", df.head())
print(" Data Summary : ", df.describe())
print(" Value Counts: ", df["label"].value_counts())

#Plot

sns.countplot(x="label", data=df)
plt.title("Label Distribution")
plt.savefig("data/class_distribution.png")
print("Saved plot to inside data folder")
