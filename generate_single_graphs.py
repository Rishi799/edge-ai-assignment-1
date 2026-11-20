import os
import pandas as pd
import matplotlib.pyplot as plt

# Load untouched raw data
df = pd.read_csv("data/team_16_train.csv")

output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

features = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]
target = "days_left"
all_cols = features + [target]

# 1. Individual Histograms
for col in all_cols:
    plt.figure(figsize=(7, 5))
    plt.hist(df[col], bins=35, color='#3498db', edgecolor='black', alpha=0.75)
    plt.title(f"Histogram of {col} (Raw Data)", fontsize=13, fontweight='bold')
    plt.xlabel(col, fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    file_path = os.path.join(output_dir, f"hist_{col}.png")
    plt.savefig(file_path, dpi=150)
    plt.close()
    print(f"Saved {file_path}")

# 2. Individual Scatter Plots
for col in features:
    plt.figure(figsize=(7, 5))
    plt.scatter(df[col], df[target], color='#e67e22', alpha=0.5, s=20, edgecolors='none')
    plt.title(f"{target} vs {col} (Raw Data)", fontsize=13, fontweight='bold')
    plt.xlabel(col, fontsize=11)
    plt.ylabel(target, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    file_path = os.path.join(output_dir, f"scatter_{target}_vs_{col}.png")
    plt.savefig(file_path, dpi=150)
    plt.close()
    print(f"Saved {file_path}")

print("All individual graphs generated successfully in 'graphs/' folder.")
