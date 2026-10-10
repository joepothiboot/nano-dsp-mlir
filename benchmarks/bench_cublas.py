#!/usr/bin/env python3
import json
import os
import time

import numpy as np

SAMPLES = 10
MIN_SAMPLE_S = 0.05


def inexact(shape, seed):
    i = np.arange(np.prod(shape), dtype=np.int64)
    v = ((i + seed) * 2654435761) % 1000003

    return (v.astype(np.float32) / np.float32(997.0) - np.float32(500.0)).reshape(shape)


def bound_used(a, b, c):
    k = a.shape[1]
    exact = a.astype(np.float64) @ b.astype(np.float64)
    u = 2.0**-24
    gamma = k * u / (1 - k * u)
    bound = gamma * (np.abs(a).astype(np.float64) @ np.abs(b).astype(np.float64))

    return float(np.max(np.abs(c - exact) / bound))


def time_samples(run):
    run()
    n = 1

    while True:
        t0 = time.perf_counter()

        for _ in range(n):
            run()

        if time.perf_counter() - t0 >= MIN_SAMPLE_S:
            break

        n *= 2

    samples = []

    for _ in range(SAMPLES):
        t0 = time.perf_counter()

        for _ in range(n):
            run()

        samples.append((time.perf_counter() - t0) / n)

    return min(samples), float(np.median(samples)), float(np.std(samples, ddof=1))


def main():
    import cupy as cp

    assert os.environ.get("CUPY_TF32", "0") == "0", "unset CUPY_TF32: TF32 is not an f32 baseline"
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    rows = []

    for n in [256, 512, 1024, 2048]:
        a, b = inexact((n, n), 1), inexact((n, n), 2)
        da, db = cp.asarray(a), cp.asarray(b)
        c = cp.matmul(da, db).get()
        used = bound_used(a, b, c)

        if used > 1:
            raise SystemExit(f"cuBLAS {n}: outside the reordering bound ({used:.2f})")

        def run():
            cp.matmul(da, db)
            cp.cuda.Device().synchronize()

        best, median, stddev = time_samples(run)
        shape = f"{n}x{n}x{n}"
        rows.append(
            {
                "name": f"matmul/{shape}/cublas", "op": "matmul", "shape": shape, "impl": "cublas",
                "config": device, "real_time": best * 1e9, "median_time": median * 1e9,
                "sample_stddev": stddev * 1e9, "time_unit": "ns", "aggregate": "min", "samples": SAMPLES,
                "ops": 2 * n**3, "rate": 2 * n**3 / (best * 1e9), "rate_unit": "GFLOP/s",
                "bytes": 3 * n * n * 4, "intensity": 2 * n**3 / (3 * n * n * 4),
                "checked": "within-bound", "bound_used": used,
            }
        )

    context = {"device": device, "timing": "cupy.matmul + synchronize per call; best of 10 samples of >= 50 ms"}
    print(json.dumps({"context": context, "benchmarks": rows}, indent=1))


if __name__ == "__main__":
    main()
