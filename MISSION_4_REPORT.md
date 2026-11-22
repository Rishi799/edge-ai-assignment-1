# Mission 4 Report: Would It Fit Inside the Doorbell?
**Team 16 (Team MH)** · Board: ESP32-S3, about 512 KB SRAM (about **300 KiB free** for our program),
8 MiB PSRAM, 16 MB flash.

**Budgets used everywhere below:** 1 KiB = 1,024 bytes, so 300 KiB SRAM = **307,200 bytes**
(`SRAM_BUDGET_BYTES` in `app/main.py`), and 8 MiB PSRAM = **8,388,608 bytes**.

Every number in sections 1–3 comes from `python client/width_table.py`, which calls `describe()`
from `app/main.py`. That is the **same function** `/model_info` uses, so the report and the server
always count the same way.

---

## 1. Our model: 5 → 16 → 16 → 1

| | Value |
|---|---|
| Parameters (weights + biases) | **385** |
| float32 (4 bytes each) | **1,540 bytes** (0.50% of 300 KiB) |
| int8 (1 byte each) | **385 bytes** |
| MACs per prediction | **352** |
| Largest pair of activation buffers | **32 values = 128 bytes float32 (32 bytes int8)** |

**Activation memory, explained.** Activations are the numbers passed between layers during one
prediction. When computing a layer, the chip must hold that layer's input and output at the same
time, but it can throw away earlier layers' values. The biggest pair is `linear2` (16 in + 16 out =
32 values). The other layers need 5 + 16 = 21 and 16 + 1 = 17 values. Even keeping every activation
(5 + 16 + 16 + 1 = 38 values = 152 bytes) would be tiny.

**Live check:** `GET /model_info` returns `total_bytes_int8: 385`, `fits_300kb_sram_f32: true`,
`fits_300kb_sram_int8: true`.

| Layer | in → out | Parameters | float32 bytes | int8 bytes | MACs | Activations held (in + out) |
|---|---|---|---|---|---|---|
| linear1 | 5 → 16 | 96 | 384 | 96 | 80 | 21 |
| linear2 | 16 → 16 | 272 | 1,088 | 272 | 256 | **32** |
| linear3 | 16 → 1 | 17 | 68 | 17 | 16 | 17 |
| **Total** | | **385** | **1,540** | **385** | **352** | peak **32** |

---

## 2. Width table for 5 → w → w → 1

| w | Parameters | float32 bytes | int8 bytes | Fits 300 KiB SRAM (f32) | Fits 300 KiB SRAM (int8) | Fits 8 MiB PSRAM (f32) | Fits 8 MiB PSRAM (int8) | MACs |
|---|---|---|---|---|---|---|---|---|
| 32 | 1,281 | 5,124 | 1,281 | yes | yes | yes | yes | 1,216 |
| 128 | 17,409 | 69,636 | 17,409 | yes | yes | yes | yes | 17,152 |
| 256 | 67,585 | 270,340 | 67,585 | yes | yes | yes | yes | 67,072 |
| 512 | 266,241 | 1,064,964 | 266,241 | **no** | yes | yes | yes | 265,216 |
| 2048 | 4,210,689 | 16,842,756 | 4,210,689 | **no** | **no** | **no** | yes | 4,206,592 |

(Peak activation memory stays small: 2 × w values, from 256 bytes at w = 32 up to 16,384 bytes at
w = 2048, float32. The weights are what run out of room.)

**What it shows:** parameters grow with **w²**, because the middle `w → w` layer has w × w weights.
Doubling the width roughly quadruples the memory. int8 buys 4× the memory, which is only **2×
the width**.

---

## 3. Exact width where the weights stop fitting in 300 KiB

**Formula.** For 5 → w → w → 1:

params(w) = (5w + w) + (w·w + w) + (w + 1) = **w² + 8w + 1**

(Check: w = 16 → 256 + 128 + 1 = 385 ✓; w = 32 → 1,024 + 256 + 1 = 1,281 ✓.)

**float32:** 4 × (w² + 8w + 1) ≤ 307,200 → w² + 8w + 1 ≤ 76,800
→ w ≤ −4 + √(16 + 76,799) = −4 + 277.16 = **273.16**

**int8:** w² + 8w + 1 ≤ 307,200
→ w ≤ −4 + √(16 + 307,199) = −4 + 554.27 = **550.27**

**Checked with `describe()` (the script searches width by width, so this doesn't rely on the algebra):**

| | Largest width that fits | Bytes at that width | Next width | Bytes at next width |
|---|---|---|---|---|
| float32 | **w = 273** | 306,856 ≤ 307,200 ✓ | w = 274 | 309,076 ✗ |
| int8 | **w = 550** | 306,901 ≤ 307,200 ✓ | w = 551 | 308,010 ✗ |

Our 16,16 model is about **17× narrower** than the float32 limit. In practice the true limit is a bit
lower: the program code, the Wi-Fi stack and the activations also need some of that 300 KiB.

---

## 4. Extrapolated int8 kernel time

**Method:** time ≈ MACs × (clock cycles per MAC) ÷ clock speed.

**The speed we assume, and where it came from:**
- **Clock: 240 MHz.** The ESP32-S3 has a dual-core Xtensa LX7 CPU running at up to 240 MHz, with
  128-bit SIMD (vector) instructions for neural-network work.
  Source: [Espressif ESP32-S3 product page](https://www.espressif.com/en/products/socs/esp32-s3) and
  [ESP32-S3 datasheet](https://mouser.com/datasheet/2/891/Espressif_ESP32_S3_Datasheet-2904763.pdf).
- **Cycles per MAC: measured by Espressif.** Espressif's ESP-NN library benchmarks a
  **fully connected layer**, which is exactly what our `Linear` layers are, with 271 inputs and
  3 outputs (271 × 3 = **813 MACs**), int8, on an ESP32-S3 at 240 MHz with the tensors in internal
  memory:
  - plain C ("ANSI C"): **8,439 cycles** → 8,439 ÷ 813 = **10.4 cycles per MAC**
  - SIMD-optimised: **1,543 cycles** → 1,543 ÷ 813 = **1.9 cycles per MAC**

  Source: [ESP-NN v1.5.0 performance table, `docs/performance.md`](https://github.com/espressif/esp-nn/blob/ace42ac7721119cd543fc40ffe719e601fa8f20f/docs/performance.md)
  (linked from the [ESP-NN component page](https://components.espressif.com/components/espressif/esp-nn/versions/1.5.0/readme)).

So we assume **10.4 cycles per MAC, a conservative plain-C rate** (about 23 million MACs per
second), and also show the optimised rate of 1.9 cycles per MAC (about 126 million MACs per second).

| Model | MACs | Plain C (10.4 cycles/MAC) | Optimised (1.9 cycles/MAC) | Weights fit in SRAM? |
|---|---|---|---|---|
| **16,16 (ours)** | **352** | **≈ 15.2 µs** (3,654 cycles) | **≈ 2.8 µs** (668 cycles) | yes (f32 and int8) |
| 32,32 | 1,216 | ≈ 53 µs | ≈ 10 µs | yes |
| 256,256 | 67,072 | ≈ 2.9 ms | ≈ 0.53 ms | yes |
| 512,512 | 265,216 | ≈ 11.5 ms | ≈ 2.1 ms | int8 only |
| 2048,2048 | 4,206,592 | ≈ 182 ms | ≈ 33 ms | no (int8 needs PSRAM, which is slower) |

For 352 MACs, the extrapolated int8 kernel estimates are 15.2 µs (plain C) and 2.8 µs
(optimised). Full prediction latency on the board remains unmeasured.

---

## 5. Should the doorbell call our server or run the model itself?

We recommend local inference. The parameters occupy 1,540 bytes in float32. Two 16-value
float32 activation buffers occupy another 128 B. Total runtime RAM depends on the implementation.

Local prediction keeps ring counts in the home and allows a low-battery warning while Wi-Fi is down.
A server call took 40.87 ms at the median with keep-alive. Opening a new connection each time
raised the median to 594.24 ms, a roughly 553 ms difference in round-trip time. The experiment
did not measure radio energy or deployment costs.

A server remains useful for training and updates. When readings do need uploading, Mission 3
measured a 263× batch speedup for 400 readings (22,127 ms individually versus 84.0 ms together).
