# Mission 4 Report: Would It Fit Inside the Doorbell?
**Team 16 (Team MH)** · Board: ESP32-S3, about 512 KB SRAM (about **300 KB free** for our program),
8 MB PSRAM, 16 MB flash.

**Budgets used everywhere below:** 1 KB = 1,024 bytes, so 300 KB SRAM = **307,200 bytes**
(`SRAM_BUDGET_BYTES` in `app/main.py`), and 8 MB PSRAM = **8,388,608 bytes**.

Every number in sections 1–3 comes from `python client/width_table.py`, which calls `describe()`
from `app/main.py`. That is the **same function** `/model_info` uses, so the report and the server
always count the same way.

---

## 1. Our model: 5 → 16 → 16 → 1

| | Value |
|---|---|
| Parameters (weights + biases) | **385** |
| float32 (4 bytes each) | **1,540 bytes** (0.50% of 300 KB) |
| int8 (1 byte each) | **385 bytes** |
| MACs per prediction | **352** |
| Largest activation memory needed at once | **32 values = 128 bytes float32 (32 bytes int8)** |

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

| w | Parameters | float32 bytes | int8 bytes | Fits 300 KB SRAM (f32) | Fits 300 KB SRAM (int8) | Fits 8 MB PSRAM (f32) | Fits 8 MB PSRAM (int8) | MACs |
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

## 3. Exact width where the weights stop fitting in 300 KB

**Formula.** For 5 → w → w → 1:

params(w) = (5w + w) + (w·w + w) + (w + 1) = **w² + 8w + 1**

(Check: w = 16 → 256 + 128 + 1 = 385 ✓; w = 32 → 1,024 + 256 + 1 = 1,281 ✓.)

**float32:** 4 × (w² + 8w + 1) ≤ 307,200 → w² + 8w + 1 ≤ 76,800
→ w ≤ −4 + √(16 + 76,799) = −4 + 277.13 = **273.13**

**int8:** w² + 8w + 1 ≤ 307,200
→ w ≤ −4 + √(16 + 307,199) = −4 + 554.27 = **550.27**

**Checked with `describe()` (the script searches width by width, so this doesn't rely on the algebra):**

| | Largest width that fits | Bytes at that width | Next width | Bytes at next width |
|---|---|---|---|---|
| float32 | **w = 273** | 306,856 ≤ 307,200 ✓ | w = 274 | 309,076 ✗ |
| int8 | **w = 550** | 306,901 ≤ 307,200 ✓ | w = 551 | 308,010 ✗ |

Our 16,16 model is about **17× narrower** than the float32 limit. In practice the true limit is a bit
lower: the program code, the Wi-Fi stack and the activations also need some of that 300 KB.

---

## 4. Time for one prediction on the board

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
| **16,16 (ours)** | **352** | **≈ 15 µs** (3,654 cycles) | **≈ 3 µs** (668 cycles) | yes (f32 and int8) |
| 32,32 | 1,216 | ≈ 53 µs | ≈ 10 µs | yes |
| 256,256 | 67,072 | ≈ 2.9 ms | ≈ 0.53 ms | yes |
| 512,512 | 265,216 | ≈ 11.5 ms | ≈ 2.1 ms | int8 only |
| 2048,2048 | 4,206,592 | ≈ 182 ms | ≈ 33 ms | no (int8 needs PSRAM, which is slower) |

**Our estimate: one prediction takes about 15 µs** (0.015 ms) on the doorbell, or about 3 µs with
Espressif's optimised kernels.

**Caveats:**
- The benchmark is for **int8**. As float32, the S3's single-precision FPU has no SIMD path, so use
  the plain-C rate as the realistic one. Either way, it's microseconds.
- The two ReLUs (32 comparisons) and normalising 5 inputs add only a few hundred cycles more, about
  1–2 µs.
- Real timing would be measured on the board with `esp_timer_get_time()`. We have no board, so this
  is an estimate.

**Compared with the server:** 15 µs on the device vs a **41 ms** typical round trip to our Codespace
(Mission 3). That's about **2,700× faster**, and about **39,000×** faster than the 585 ms first
request.

---

## 5. Should the doorbell call our server or run the model itself?

Numbers from Mission 3 (laptop → Codespace, 21 Nov 2026):

| | Call the server | Run on the doorbell |
|---|---|---|
| **Latency** | p50 **41 ms**, p95 311 ms, p99 700 ms; first request **585 ms**; with no keep-alive, about 594 ms each | **≈ 15 µs** (section 4): about 2,700× faster than the typical round trip |
| | The model itself was only 0.23 ms of that; **98.9% was network** | No network at all |
| **Privacy** | Every reading (battery, temperature, how often the bell rings, Wi-Fi strength) leaves the house. `rings_per_day` reveals when people visit or are home. | Data stays on the device. Only the result needs to be sent, if anything. |
| **Cost** | A server running 24/7, paid per hour, and it scales with the number of doorbells. | Free after it ships: 1.5 KB of flash and microseconds of CPU per prediction. |
| **Power** | The Wi-Fi radio is the most power-hungry part of the board. A TLS handshake alone took about 550 ms of radio time in our test. | 352 multiply-adds is a negligible amount of energy compared with the radio. |
| **Wi-Fi down** | No prediction at all: the doorbell can't warn that its battery is dying, possibly right when it matters. | Still works. It can show a local warning (LED or beep) and report when the Wi-Fi returns. |

**Our recommendation: run the model on the doorbell.**
1. **It fits easily.** 1,540 bytes as float32 is 0.5% of the free SRAM, so we don't even need int8.
   The activations need only 128 bytes.
2. **It's faster.** About 15 µs on the device vs 41 ms typical and 585 ms worst-case-first over the
   network. Mission 3 showed 98.9% of a server request is network, not model.
3. **It's private.** Ring counts reveal when people are home. Keeping them on the device means they
   never have to leave the house.
4. **It's cheaper.** No server running 24/7, and no cost per doorbell as the product grows.
5. **It saves battery.** The irony of the alternative: to predict the battery life, the server
   version must switch on the Wi-Fi radio, the biggest drain on the battery. The on-device
   version spends a few thousand CPU cycles.
6. **It works offline.** With the Wi-Fi down, the doorbell can still warn "battery low" with its
   LED or chime, which is exactly when a dying battery matters most.

**The server still has a job: training and updates, not predictions.** The doorbell can send
occasional anonymous readings (in batches, as Mission 3 showed, about 263× cheaper than one call
each) so we can retrain, then ship new weights (1.5 KB) in a firmware update. The doorbell can also
report its prediction the next time it's online anyway.

**When the server would win:** if the model were too big for the device (wider than about 273 at
float32 or 550 at int8, per section 3, or beyond 8 MB PSRAM), or if the prediction needed data from
many doorbells at once.
