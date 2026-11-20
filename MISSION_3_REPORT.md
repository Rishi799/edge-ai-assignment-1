# Mission 3 Report: Serve It, and Time Every Stage
**Team 16 (Team MH)** · Model served: 5 → 16 → 16 → 1 (385 params, val MAE 4.97 days)

Every item follows the loop: **predict with a number → measure → explain the gap.**
Sections marked ✏️ **TO FILL** need our own predictions or the Codespace runs.

---

## 0. Before Mission 3: the LLM's prediction
✏️ **TO FILL:** ask an LLM *"In a FastAPI server running a tiny PyTorch model, which stage of a web
request will be the slowest?"* and paste its answer here word for word.

> LLM answer: …

## 1. Our predictions (written before measuring)
✏️ **TO FILL** before reading section 4:

| Quantity | Our prediction |
|---|---|
| Model inference time for one reading | … ms |
| Laptop → Codespace round trip | … ms |
| Slowest stage of a request | … |

---

## 2. Input ranges: glitches get HTTP 422
Code: `class Reading` in `app/main.py`. The limits come from **what real hardware can report**, not
from our data's min and max, so a real reading slightly outside our data still gets an answer.

| Field | Range accepted | Why | Glitch in our data |
|---|---|---|---|
| `battery_v` | 2.5 to 4.5 V | Li-ion cut-off is about 2.5–3.0 V; full charge is 4.2 V (4.35 V for HV cells) | 0.0 V |
| `temp_c` | -40 to 85 °C | Industrial electronics rating, covers any outdoor doorbell | -127 °C (DS18B20 error code) |
| `rings_per_day` | 0 to 1000 | A count can't be negative; 1000/day is far beyond any front door | none |
| `wifi_rssi_dbm` | -120 to -10 dBm | RSSI is always negative; -120 is below any receiver's sensitivity, about -10 is right next to the router | 0 dBm |
| `age_months` | 0 to 240 | Can't be negative; 20 years is far beyond a doorbell's life | none |
| any field | must be a finite number | `NaN` / `Infinity` are never real readings (`allow_inf_nan=False`) | — |

### Test results (local server)
| Reading sent | Expected | Got |
|---|---|---|
| Normal reading | 200 | **200**, 139.72 days |
| `battery_v: 0` | 422 | **422** |
| `temp_c: -127` | 422 | **422** |
| `wifi_rssi_dbm: 0` | 422 | **422** |
| `rings_per_day: -3` | 422 | **422** |
| `age_months: -1` | 422 | **422** |
| `age_months` missing | 422 | **422** |
| `temp_c: "hot"` | 422 | **422** |
| `temp_c: NaN` / `Infinity` | 422 | **422** (was 500 before the fix below) |
| Heatwave `temp_c: 49` (real, outside our data) | 200 | **200** |
| `battery_v: 3.30` (real, below our data) | 200 | **200** |
| Batch with one glitch row | 422 | **422** (the whole batch is refused) |

✏️ **TO FILL:** screenshots of three 422 replies from `/docs`.

**Bug found while testing:** `NaN` was first answered with **500 Internal Server Error**, not 422.
The rule had worked: Pydantic refused NaN. But the 422 reply copied the bad input back
into its error message, and `NaN` is not valid JSON, so building the reply crashed.
Fix: `e.json(include_input=False)`. The reply still says which field was wrong and why.

---

## 3. Stage timing
Each stage in `run()` (`app/main.py`) has its own stopwatch (`time.perf_counter()`), the same way
`parse` was timed in the starter:

| Stage | What it does |
|---|---|
| `parse` | Read the request body and validate it against `Reading` (ranges included) |
| `normalise` | Turn readings into a NumPy array and apply `(x - MU) / SD` |
| `inference` | Run the network with `torch.no_grad()` |
| `build` | Clip to ≥ 0, round, build the reply dictionary |
| `server_total` | Everything above, from start to finish |

---

## 4. Measurements

### 4a. Local run (laptop → server on the same laptop), `--n 200`
Run on Windows against `http://127.0.0.1:8000`. Files: `bench_20261120_222425.csv` (keep-alive)
and `bench_20261120_222429.csv` (no keep-alive).

| Stage (p50 / p95 / p99 ms) | keep-alive | `--no-keepalive` |
|---|---|---|
| **first request** | 4.8 | 21.4 |
| round_trip | 2.93 / 4.19 / 5.35 | 14.83 / 29.72 / 31.13 |
| parse | 0.042 / 0.087 / 0.141 | 0.069 / 0.122 / 0.230 |
| normalise | 0.035 / 0.058 / 0.074 | 0.058 / 0.082 / 0.127 |
| inference | 0.233 / 0.411 / 0.552 | 0.383 / 0.696 / 0.935 |
| build | 0.018 / 0.034 / 0.045 | 0.024 / 0.043 / 0.053 |
| server_total | 0.355 / 0.606 / 0.800 | 0.565 / 0.995 / 1.286 |
| network_and_framework | 2.56 / 3.68 / 4.86 | 14.33 / 29.17 / 30.46 |

**`localhost` vs `127.0.0.1` (surprise found while measuring):** with `--url http://localhost:8000`,
the first request took **2073 ms**, and `--no-keepalive` took about 2 s for *every* request.
With `127.0.0.1` the first request took **4.8 ms**. The cause is on the Windows client side, not
the server: `localhost` is tried as IPv6 (`::1`) first, uvicorn only listens on IPv4, and Windows
waits about 2 s before falling back to IPv4. With keep-alive this cost is paid only once, on the
first connection; without keep-alive it's paid on every request.

### 4b. Laptop → Codespace (the real one)
✏️ **TO FILL:** in the Codespace, start the server and set port 8000 to **Public**. Then, on the laptop:
```
python client/bench.py --url https://<codespace>-8000.app.github.dev --n 200
python client/bench.py --url https://<codespace>-8000.app.github.dev --n 200 --no-keepalive
```
and **inside the Codespace terminal**:
```
python client/bench.py --url http://localhost:8000 --n 200
```

| Stage (p50 / p95 / p99 ms) | Laptop → Codespace | Laptop → Codespace, `--no-keepalive` | Inside Codespace (localhost) |
|---|---|---|---|
| first request | | | |
| round_trip | | | |
| parse | | | |
| normalise | | | |
| inference | | | |
| build | | | |
| server_total | | | |
| network_and_framework | | | |

### 4c. 400 single `/predict` calls vs one `/predict_batch` of 400
Script: `client/batch_vs_single.py` (same random readings as `bench.py`, fixed seed, keep-alive).

| | Local result |
|---|---|
| 400 single calls | **1102 ms** total (2.75 ms per reading; the server was busy for only 134.6 ms of it) |
| 1 batch of 400 | **7.4 ms** total (0.018 ms per reading; server busy 2.5 ms) |
| Speed-up | **about 150×** |
| Same answers? | **Yes**, largest difference 0.0 days (grading check "Batch") |
| Batch stages | parse 1.35 · normalise 0.56 · inference 0.35 · build 0.23 ms |

✏️ **TO FILL:** run it against the Codespace address as well:
`python client/batch_vs_single.py --url https://<codespace>-8000.app.github.dev --n 400`

---

## 5. Where does the time go? (draft from local numbers)
Typical request (keep-alive, local p50): **round trip 2.93 ms**

```
network + framework  ██████████████████████████████████████████  2.56 ms  (87%)
inference            ████                                         0.23 ms  ( 8%)
parse                █                                            0.04 ms
normalise            █                                            0.04 ms
build                ▏                                            0.02 ms
```
The model itself is a small part of every request. Most of the time goes to sending the request,
HTTP handling in uvicorn/FastAPI, and sending the reply. Over the internet to a Codespace,
the network share will be much larger still.

✏️ **TO FILL:** redraw this with the **laptop → Codespace** p50 numbers, then answer:
*Was the LLM right about the slowest stage?*

## 6. Explanations
**Why is `--no-keepalive` slower?** With keep-alive, `bench.py` reuses one open connection for every
request. Without it, every request opens a new connection first (TCP handshake) and closes it after.
Locally that cost about +12 ms per request (2.9 → 14.8 ms at p50). Over HTTPS to a Codespace,
each new connection also needs a **TLS handshake** (several extra round trips across the internet), so
✏️ **TO FILL:** expect, then measure, a much bigger gap.

**Why is the first request slower?** It has to open the connection (plus TLS over HTTPS). Python and
PyTorch may also still be warming up (code paths and memory allocated the first time).
✏️ **TO FILL:** compare first-request times in 4b.

**Why is localhost inside the Codespace faster than laptop → Codespace?** ✏️ **TO FILL:** the request
never leaves the machine: no internet distance, no TLS, and no GitHub port-forwarding proxy in between.

**Why is one batch about 150× faster than 400 singles?** 400 singles = 400 round trips, each paying the
network + framework cost (about 2.6 ms here). One batch = **1** round trip. Inside the server, the
network also runs all 400 rows as one matrix multiplication, which takes about the same time as one
row (inference 0.35 ms for 400 rows vs 0.23 ms for 1). The per-request overhead dominates, not the model.
