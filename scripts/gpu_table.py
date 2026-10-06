#!/usr/bin/env python3
"""Prints the GPU results in benchmarks/results/*.json as the markdown table
of docs/06-gpu-results.md: one row per op and shape, one column per
implementation and device, median time ± sample standard deviation."""
import json
import pathlib

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def fmt(ns):
    return f"{ns / 1e6:.2f} ms" if ns >= 1e6 else f"{ns / 1e3:.1f} µs"


rows, columns = {}, []
for path in sorted(RESULTS.glob("*.json")):
    for b in json.loads(path.read_text())["benchmarks"]:
        column = f"{b['impl']} ({b['config']})"
        if column not in columns:
            columns.append(column)
        cell = f"{fmt(b['median_time'])} ± {fmt(b['sample_stddev'])}, {b['rate']:.0f} GFLOP/s"
        rows.setdefault((b["op"], b["shape"]), {})[column] = cell

print("| op and shape | " + " | ".join(columns) + " |")
print("| --- |" + " ---: |" * len(columns))
for (op, shape), cells in rows.items():
    print(f"| {op} {shape} | " + " | ".join(cells.get(c, "—") for c in columns) + " |")
