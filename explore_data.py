import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("data/team_16_train.csv")
print("=== Dataset Summary ===")
print(f"Total rows: {len(df)}")
print("\nSummary Statistics:")
print(df.describe())

features = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]
target = "days_left"
all_cols = features + [target]

# 2. Histogram of every column
plt.figure(figsize=(15, 9))
for i, col in enumerate(all_cols, 1):
    plt.subplot(2, 3, i)
    if col == "age_months":
        bins = np.arange(df[col].min() - 0.5, df[col].max() + 1.5, 1)
        plt.hist(df[col], bins=bins, color='skyblue', edgecolor='black', alpha=0.7, rwidth=0.85)
        plt.xticks(np.arange(0, int(df[col].max()) + 1, 5))
    elif col == "rings_per_day":
        bins = np.arange(df[col].min() - 0.5, df[col].max() + 1.5, 1)
        plt.hist(df[col], bins=bins, color='skyblue', edgecolor='black', alpha=0.7, rwidth=0.85)
        plt.xticks(np.arange(0, int(df[col].max()) + 1, 5))
    else:
        plt.hist(df[col], bins=35, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("histograms.png", dpi=150)
plt.close()
print("Saved histograms.png")

# 3. Scatter plots: days_left against each input
plt.figure(figsize=(16, 5))
for i, col in enumerate(features, 1):
    plt.subplot(1, 5, i)
    plt.scatter(df[col], df[target], alpha=0.5, s=15, color='coral')
    plt.title(f"{target} vs {col}")
    plt.xlabel(col)
    plt.ylabel(target)
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("scatter_vs_days_left.png", dpi=150)
plt.close()
print("Saved scatter_vs_days_left.png")

# Copy to brain artifact directory for display
artifact_dir = r"C:\Users\user\.gemini\antigravity\brain\90d4f7d4-e92e-47e8-a521-fe200e6637d4"
os.makedirs(artifact_dir, exist_ok=True)
shutil.copy("histograms.png", os.path.join(artifact_dir, "histograms.png"))
shutil.copy("scatter_vs_days_left.png", os.path.join(artifact_dir, "scatter_vs_days_left.png"))
