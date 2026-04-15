import pandas as pd

df = pd.read_csv("dataset.csv")
df["bracket"] = (df["length"] // 32) * 32
counts = df["bracket"].value_counts().sort_index()
pcts = (counts / len(df) * 100).round(2)
for bracket, pct in pcts.items():
    print(f"{int(bracket):>4}-{int(bracket)+31:<4} tokens: {pct:>6.2f}%")