"""Mission 4: would a 5 -> w -> w -> 1 network fit inside the doorbell (ESP32-S3)?

Uses describe() from app/main.py -- the SAME function that /model_info uses -- so the table here
and the server always count parameters, bytes and MACs the same way.

Run from the repo folder:   python client/width_table.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.main import SRAM_BUDGET_BYTES, describe, net  # noqa: E402
from app.model import BatteryNet  # noqa: E402

PSRAM_BYTES = 8 * 1024 * 1024  # 8 MB = 8,388,608 bytes


def peak_activation_values(network):
    """Largest number of activation values held at once: one layer's input and its output together."""
    return max(L["in"] + L["out"] for L in describe(network)["layers"])


def fits(info, bytes_per_param, budget):
    return info["total_params"] * bytes_per_param <= budget


# --- 1. Our served model --------------------------------------------------------------------
info = describe(net)
act = peak_activation_values(net)
print("Served model:", " -> ".join([str(info["layers"][0]["in"])] + [str(L["out"]) for L in info["layers"]]))
print(f"  params {info['total_params']:,}   float32 {info['total_bytes_f32']:,} B   int8 {info['total_bytes_int8']:,} B"
      f"   MACs {info['total_macs']:,}")
print(f"  peak activations: {act} values = {act * 4} B float32 / {act} B int8")
print(f"  share of 300 KB SRAM (float32 weights): {100 * info['total_bytes_f32'] / SRAM_BUDGET_BYTES:.2f}%\n")

# --- 2. Width table ---------------------------------------------------------------------------
yn = lambda b: "yes" if b else "NO"
print(f"{'w':>5} {'params':>10} {'float32 B':>12} {'int8 B':>10} {'SRAM f32':>9} {'SRAM int8':>10}"
      f" {'PSRAM f32':>10} {'PSRAM int8':>11} {'MACs':>10} {'peak act f32 B':>15}")
for w in [32, 128, 256, 512, 2048]:
    m = BatteryNet((w, w))
    d = describe(m)
    print(f"{w:>5} {d['total_params']:>10,} {d['total_bytes_f32']:>12,} {d['total_bytes_int8']:>10,}"
          f" {yn(d['fits_300kb_sram_f32']):>9} {yn(d['fits_300kb_sram_int8']):>10}"
          f" {yn(fits(d, 4, PSRAM_BYTES)):>10} {yn(fits(d, 1, PSRAM_BYTES)):>11}"
          f" {d['total_macs']:>10,} {peak_activation_values(m) * 4:>15,}")

# --- 3. Exact width where the weights stop fitting in 300 KB ----------------------------------
# params(w) = (5w + w) + (w*w + w) + (w + 1) = w^2 + 8w + 1. Search instead of trusting the algebra.
print()
for name, bpp in [("float32", 4), ("int8", 1)]:
    w = 1
    while describe(BatteryNet((w + 1, w + 1)))["total_params"] * bpp <= SRAM_BUDGET_BYTES:
        w += 1
    a, b = describe(BatteryNet((w, w))), describe(BatteryNet((w + 1, w + 1)))
    print(f"{name:>7}: largest width that fits = {w} ({a['total_params'] * bpp:,} B <= {SRAM_BUDGET_BYTES:,});"
          f" w = {w + 1} needs {b['total_params'] * bpp:,} B -> does not fit")
