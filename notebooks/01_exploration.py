import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/raw/creditcard.csv")

# basic shape
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# class balance
fraud_count = df['Class'].value_counts()
fraud_pct = df['Class'].value_counts(normalize=True) * 100
print("\nClass counts:")
print(fraud_count)
print("\nClass %:")
print(fraud_pct.round(4))

# missing values
print("\nMissing values:", df.isnull().sum().sum())

# amount and time stats
print("\nAmount stats:")
print(df['Amount'].describe())
print("\nTime stats:")
print(df['Time'].describe())

# correlation with Class
corr = df.corr()['Class'].drop('Class').sort_values(key=abs, ascending=False)
print("\nTop 10 features correlated with Class:")
print(corr.head(10))

# plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].set_title("Amount Distribution")
axes[0].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.6, label='legit', color='steelblue')
axes[0].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.8, label='fraud', color='red')
axes[0].set_xlabel("Amount")
axes[0].legend()

axes[1].set_title("Time Distribution")
axes[1].hist(df[df['Class']==0]['Time'], bins=50, alpha=0.6, label='legit', color='steelblue')
axes[1].hist(df[df['Class']==1]['Time'], bins=50, alpha=0.8, label='fraud', color='red')
axes[1].set_xlabel("Time")
axes[1].legend()

plt.tight_layout()
plt.savefig("../docs/eda_distributions.png")
plt.close()

# class imbalance bar
fig2, ax2 = plt.subplots(figsize=(5, 4))
fraud_count.plot(kind='bar', ax=ax2, color=['steelblue', 'red'])
ax2.set_title("Class Balance")
ax2.set_xticklabels(['Legit (0)', 'Fraud (1)'], rotation=0)
ax2.set_ylabel("Count")
plt.tight_layout()
plt.savefig("../docs/class_balance.png")
plt.close()

print("\ndone — plots saved to docs/")
