#!/usr/bin/env python3
import json
import pathlib
import sys

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "benchmarks"))

from bench_cublas import SAMPLES, bound_used, inexact, time_samples

SIZES = [256, 512, 1024, 2048]

CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
]


@triton.autotune(configs=CONFIGS, key=["m", "n", "k"])
@triton.jit
def matmul_kernel(a, b, c, m, n, k, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for t in range(0, k, BLOCK_K):
        x = tl.load(a + rm[:, None] * k + (t + rk)[None, :])
        y = tl.load(b + (t + rk)[:, None] * n + rn[None, :])
        acc = tl.dot(x, y, acc, input_precision="ieee")

    tl.store(c + rm[:, None] * n + rn[None, :], acc)


def launch(da, db, dc, size):
    def grid(meta):
        return (size // meta["BLOCK_M"], size // meta["BLOCK_N"])

    matmul_kernel[grid](da, db, dc, size, size, size, enable_fp_fusion=False)


def unfused_reference(a, b):
    c = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)

    for kk in range(a.shape[1]):
        c += a[:, kk : kk + 1] * b[kk : kk + 1, :]

    return c


def main():
    device = torch.cuda.get_device_name(0)
    rows = []

    for size in SIZES:
        a, b = inexact((size, size), 1), inexact((size, size), 2)
        da, db = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
        dc = torch.empty_like(da)

        launch(da, db, dc, size)
        torch.cuda.synchronize()
        got = dc.cpu().numpy()

        used = bound_used(a, b, got)
        exact = np.array_equal(got.view(np.uint32), unfused_reference(a, b).view(np.uint32))
        checked = "bit-exact" if exact else "within-bound"

        if used > 1:
            raise SystemExit(f"triton {size}: outside the reordering bound ({used:.2f})")

        print(f"PASS triton   {size}^3  {checked} (bound used {used:.4f})", file=sys.stderr)

        def run():
            launch(da, db, dc, size)
            torch.cuda.synchronize()

        best, median, stddev = time_samples(run)
        shape = f"{size}x{size}x{size}"
        ops = 2 * size**3
        bytes_moved = 3 * size * size * 4
        rows.append(
            {
                "name": f"matmul/{shape}/triton", "op": "matmul", "shape": shape, "impl": "triton",
                "config": device, "real_time": best * 1e9, "median_time": median * 1e9,
                "sample_stddev": stddev * 1e9, "time_unit": "ns", "aggregate": "min", "samples": SAMPLES,
                "ops": ops, "rate": ops / (best * 1e9), "rate_unit": "GFLOP/s",
                "bytes": bytes_moved, "intensity": ops / bytes_moved,
                "checked": checked, "bound_used": used,
                "triton_config": str(matmul_kernel.best_config),
            }
        )

    context = {
        "device": device,
        "triton": triton.__version__,
        "torch": torch.__version__,
        "timing": f"one launch + synchronize per call; best of {SAMPLES} samples of >= 50 ms",
    }
    print(json.dumps({"context": context, "benchmarks": rows}, indent=1))


if __name__ == "__main__":
    main()
