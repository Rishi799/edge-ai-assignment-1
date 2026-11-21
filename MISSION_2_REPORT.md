# Mission 2 Report: A Small Network, Opened Up
**Team 16 (Team MH)** · Data: `data/team_16_train.csv` (1,976 rows after dropping 24 glitches; 1,581 train / 395 validation, seed 0)

---

## 1. Parameters, Bytes, and MACs by Hand

### Calculation Rules for a `Linear(a, b)` Layer
- **Parameters:** $\text{weights} + \text{biases} = a \times b + b$
- **MACs (Multiply-Accumulates) per inference:** $a \times b$
- **Memory Footprint:**
  - `float32`: $\text{parameters} \times 4\text{ bytes}$
  - `int8`: $\text{parameters} \times 1\text{ byte}$
- *Note:* ReLU activation layers have 0 parameters and 0 MACs.

### Architecture Comparison

#### Starter Network: $5 \to 32 \to 32 \to 1$
| Layer | in → out | Parameters Formula | Parameters | float32 Bytes | MACs |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `linear1` | 5 → 32 | $5 \times 32 + 32$ | **192** | 768 B | 160 |
| `linear2` | 32 → 32 | $32 \times 32 + 32$ | **1,056** | 4,224 B | 1,024 |
| `linear3` | 32 → 1 | $32 \times 1 + 1$ | **33** | 132 B | 32 |
| **Total** | | | **1,281** | **5,124 B** (~5.00 KB) | **1,216** |

#### Chosen & Served Network: $5 \to 16 \to 16 \to 1$
| Layer | in → out | Parameters Formula | Parameters | float32 Bytes | MACs |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `linear1` | 5 → 16 | $5 \times 16 + 16$ | **96** | 384 B | 80 |
| `linear2` | 16 → 16 | $16 \times 16 + 16$ | **272** | 1,088 B | 256 |
| `linear3` | 16 → 1 | $16 \times 1 + 1$ | **17** | 68 B | 16 |
| **Total** | | | **385** | **1,540 B** (~1.50 KB) | **352** |

### Verification with the API
- Running `GET /model_info` calculates these exact metrics dynamically by inspecting the layer dimensions in `BatteryNet`:
  - `total_params`: **385**
  - `total_bytes_f32`: **1,540**
  - `total_macs`: **352**
- Running `GET /health` reports:
  - `params`: **385**
  - `val_mae`: **4.97**
- Both endpoints match our manual derivations perfectly.

---

## 2. Input Normalisation

### What Was Done
Each input feature $x$ is standardised using z-score normalization:
$$\tilde{x} = \frac{x - \mu}{\sigma}$$
where $\mu$ and $\sigma$ are the feature-wise mean and standard deviation computed strictly from the **training set only** (`train.py`), with $\sigma = 1$ if $\sigma = 0$. These statistics are serialized into `model/config.json`. The production API loads `mu` and `sd` on startup and applies the exact same transformation to incoming inference requests before the forward pass.

### Why Compute Statistics from Training Rows Only?
The validation split represents unseen future data (and acts as a proxy for the company's 400 test readings). If $\mu$ and $\sigma$ were calculated across the entire dataset, distribution parameters from the validation set would leak into the training process (data leakage), artificially deflating validation error and giving a false sense of real-world accuracy.

### Impact of Normalisation (Same $5 \to 32 \to 32 \to 1$, 1500 Epochs, Seed 0)
| Setup | Validation MAE | Notes |
| :--- | :---: | :--- |
| **Without Normalisation** | 9.59 days | Dominated by high-magnitude inputs; erratic gradient descent |
| **With Normalisation** | **4.91 days** | **~48.8% error reduction**; smooth, stable convergence |

### Why Normalisation Halves the Error
In the unnormalised dataset, feature scales differ by orders of magnitude:
- `battery_v`: ranges around 3.0 to 4.2 V ($\sigma \approx 0.24$)
- `temp_c`: ranges around 10 to 45 °C ($\sigma \approx 7.01$)
- `wifi_rssi_dbm`: ranges from -95 to -40 dBm ($\sigma \approx 12.61$)

With a global learning rate ($\eta = 0.003$), gradient steps are overwhelmed by features with large absolute numerical values (`wifi_rssi_dbm`), leaving weights connected to `battery_v` (the physically most predictive feature of battery discharge) barely updated. Standardising brings all inputs to zero mean and unit variance, allowing balanced gradient updates across all feature weights simultaneously.

---

## 3. Architecture Width & Depth Sweep

All configurations were trained with input normalisation, Adam optimizer ($\text{lr} = 0.003$), batch size 128, 1,500 epochs, and seed 0. Full checkpoints are saved in `model_runs/`. The resulting trade-off curve is rendered in `graphs/model_tradeoff_plot.png`.

| Architecture | Layers | Parameters | float32 Bytes | MACs | Val MAE (days) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `[4]` | 1 | 29 | 116 | 24 | 13.06 |
| `[8]` | 1 | 57 | 228 | 48 | 15.42 |
| `[16]` | 1 | 113 | 452 | 96 | 10.52 |
| `[32]` | 1 | 225 | 900 | 192 | 9.96 |
| `[64]` | 1 | 449 | 1,796 | 384 | 15.42 |
| **`[4, 4]`** | 2 | 49 | 196 | 44 | 8.95 |
| **`[8, 8]`** | 2 | 129 | 516 | 112 | 6.02 |
| **`[16, 16]` (Served)** | **2** | **385** | **1,540** | **352** | **4.97** |
| **`[32, 32]`** | 2 | 1,281 | 5,124 | 1,216 | 4.91 |
| **`[64, 64]`** | 2 | 4,609 | 18,436 | 4,480 | 4.65 |

### Analysis of the Trade-off Curve
1. **Depth Substantially Outperforms Width:**
   - Two-layer architectures consistently outperform single-layer models at comparable or even much lower parameter budgets.
   - For example, `[8, 8]` (129 parameters, 6.02 days) vastly outperforms `[64]` (449 parameters, 15.42 days).
   - *Physical rationale:* In Mission 1, exploratory analysis showed non-linear relationships such as the battery discharge S-curve and inverse hyperbolic relationships ($\sim 1/\text{rings}$). Two stacked linear layers separated by ReLU can construct piece-wise linear convex/concave composite shapes far more efficiently than a single wide hidden layer.
2. **Diminishing Returns Beyond `[16, 16]`:**
   - Transitioning from `[16, 16]` (385 params) to `[32, 32]` (1,281 params) quadruples the compute (+245% MACs) for a negligible gain of **0.06 days**.
   - `[64, 64]` uses 12× more parameters and 12.7× more MACs for a tiny improvement of **0.32 days**.
3. **Skeptic Check — Why did 1-layer width 8 and 64 both report 15.42?**
   - Examining full precision outputs shows:
     - Hidden `8`: **15.4157 days** (Train MAE: 15.82 days)
     - Hidden `64`: **15.4217 days** (Train MAE: 15.79 days)
   - These are distinct training trajectories that coincidentally round to the same two-decimal value.
   - Both models suffer from **severe structural underfitting** (train MAE $\approx$ val MAE $\approx$ 15.4–15.8 days). A single ReLU layer with fixed learning rate and 1,500 epochs lacks the inductive bias to model the coupled multi-variable non-linearities of this battery degradation process.

---

## 4. Overfitting vs. Generalization Analysis (Chosen Architecture: `[16, 16]`)

We recorded the progression of Training MAE vs. Validation MAE across epochs for `[16, 16]`:

| Epoch | Train MAE (days) | Val MAE (days) | Gap ($\text{Val} - \text{Train}$) |
| :---: | :---: | :---: | :---: |
| 0 | 74.09 | 72.03 | -2.06 |
| 300 | 5.71 | 5.58 | -0.13 |
| 600 | 5.14 | 5.49 | +0.35 |
| 900 | 4.93 | 5.34 | +0.41 |
| 1200 | 4.67 | 5.13 | +0.46 |
| **1499 (Final)** | **4.40** | **4.97** | **+0.57** |

### Evaluation Across Model Sizes
To verify whether larger models begin memorizing training noise, we compared final train and val errors:
- **`[16, 16]` (Chosen):** Train MAE = **4.40 days**, Val MAE = **4.97 days** (Gap = **+0.57 days**, Ratio = **1.13**)
- **`[32, 32]`:** Train MAE = **3.95 days**, Val MAE = **4.91 days** (Gap = **+0.96 days**, Ratio = **1.24**)
- **`[64, 64]`:** Train MAE = **3.17 days**, Val MAE = **4.65 days** (Gap = **+1.48 days**, Ratio = **1.47**)

### Is the `[16, 16]` Model Overfitting?
**No. The model exhibits a textbook healthy generalization profile without pathological overfitting:**
1. **Tight Generalization Gap:** A gap of only **+0.57 days** (a 1.13 validation-to-train ratio) demonstrates that training performance translates directly to unseen data.
2. **Continued Validation Progress:** Validation loss decreases monotonically through the final epoch rather than leveling off or diverging upwards (which would characterize overfitting).
3. **High Data-to-Parameter Ratio:** With 1,581 clean training samples and only 385 parameters (~4.1 training instances per parameter), the model capacity is appropriately constrained, preventing memorization.

---

## 5. Final Model Selection & Expected Error on Company Readings

### Selected Production Model: $5 \to 16 \to 16 \to 1$
- **Parameters:** 385
- **Memory Footprint (float32):** 1,540 bytes (~1.50 KB)
- **Inference Compute:** 352 MACs per forward pass
- **SRAM Fit:** Seamlessly fits into microcontroller memory (e.g. ESP32-S3's 300 KB budget using <0.5% of capacity).

### Predicted MAE on Company's 400 Unseen Readings: **5.1 – 5.5 days**

#### Scientific Justification
1. **Validation Anchor:** Our local validation split yields an MAE of **4.97 days** on 395 held-out samples.
2. **Model Selection Optimism:** Because model architecture selection (`[16, 16]`) was guided by validation metrics, validation error is slightly optimistic by $\sim 0.1\text{--}0.2\text{ days}$.
3. **Finite Sample Variance:** On an evaluation set of size $N = 400$, standard empirical confidence intervals span approximately $\pm 0.3\text{ days}$.
4. **Boundary Extrapolation:** Real-world company readings may touch feature extremes (e.g., high temperature spikes or weak Wi-Fi edge areas) where local training density is sparser.

Therefore, an expected MAE of **5.1 to 5.5 days** is an honest, rigorously grounded prediction for production evaluation.
