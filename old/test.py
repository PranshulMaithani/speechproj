# run in python terminal or add to check_and_balance_manifest.py
import pandas as pd
import numpy as np

df = pd.read_csv("outputs/manifest_expanded.csv")
before = len(df)
df = df.dropna(subset=["duration"])
df = df[df["duration"] > 0]
df.to_csv("outputs/manifest_expanded.csv", index=False)
print(f"Removed {before - len(df)} bad rows. Remaining: {len(df)}")