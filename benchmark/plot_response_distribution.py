import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
plt.figure(figsize=(10, 5))
plt.hist(df["length"], bins=50, edgecolor="black", alpha=0.7)
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.title("Response Length Distribution")
plt.axvline(df["length"].mean(), color="red", linestyle="--", label=f"Mean: {df['length'].mean():.0f}")
plt.axvline(df["length"].median(), color="orange", linestyle="--", label=f"Median: {df['length'].median():.0f}")
plt.legend()
plt.tight_layout()
plt.savefig("output_distribution.png", dpi=150)
plt.show()