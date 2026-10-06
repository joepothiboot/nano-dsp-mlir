# Mojo GPU kernels 🖥️

`mojo/nanodsp/gpu.mojo` runs `matmul` and `conv2d` on a GPU, and keeps the
repo's main promise there: the results are **bit-identical** to the CPU
kernels and the scalar C++ reference, not just close. It runs on Apple
silicon GPUs and NVIDIA GPUs from Turing (T4) on, from the same source.

Measured numbers are in [`06-gpu-results.md`](06-gpu-results.md).

## 🧩 API

```mojo
from max.gpu.host import DeviceContext
from nanodsp import Tensor
from nanodsp.gpu import BLOCKED, matmul_gpu, conv2d_gpu

var ctx = DeviceContext()
var c = matmul_gpu[DType.float32, BLOCKED](ctx, a, b)   # same bits as matmul(a, b)
var y = conv2d_gpu(ctx, image, filter, stride_h=2)       # same bits as conv2d(...)
```

Two layers, so a caller can keep data on the device:

| Layer               | Takes                         | Does                                              |
| ------------------- | ----------------------------- | ------------------------------------------------- |
| `matmul_gpu`, `conv2d_gpu` | `Tensor`s              | copy to the device, launch, copy back, wait       |
| `enqueue_matmul`, `enqueue_conv2d` | `DeviceBuffer`s | launch only; the caller synchronizes              |

`Conv2dShape` checks a convolution's shapes once (with the same errors as
`conv2d`) and uploads the 13 integers the kernel needs, so repeated launches
skip both. The GPU module is separate from the `nanodsp` package root because
it needs the `max-core` package; the CPU kernels don't.

## ⚡ Kernels

| Kernel             | Per thread    | Memory reuse                                                   |
| ------------------ | ------------- | -------------------------------------------------------------- |
| matmul `NAIVE`     | 1 output      | none                                                           |
| matmul `TILED`     | 1 output      | 16×16 tiles of `a` and `b` in shared memory                    |
| matmul `BLOCKED`   | 4×4 outputs   | 64×16 and 16×64 shared tiles, the 4×4 outputs in registers     |
| conv2d             | 1 output      | F varies fastest across threads: filter reads and output writes are contiguous, input reads are broadcasts |

## 🧪 Why the bits match

The CPU rules from [`05-soundness.md`](05-soundness.md) carry over:

1. **Same order.** Every output sums the same products in the order of the
   naive loop nest: `k` upward for matmul; `ky`, `kx`, `c` for conv2d. Tiling
   splits `k` into consecutive chunks walked in order, which keeps that
   order, exactly like the CPU schedule's reduction blocking. The last chunk
   stops at `k` rather than adding padded zeros.
2. **No fusion.** Multiply and add stay separate. That depends on
   `--fp-mode contract=off`, the same flag the CPU kernels use, because by
   default Mojo fuses `acc + a * b` into an FMA on the GPU too.

Point 2 was measured on the M2 GPU before writing any kernel: 1024 threads
each summed 256 inexact products.

| Build                                | Matches separate mul + add | Matches FMA |
| ------------------------------------ | -------------------------: | ----------: |
| `--fp-mode contract=off`             |                1024 / 1024 |  515 / 1024 |
| default                              |   (CPU fused too)          | 1024 / 1024 |

So the GPU honors the flag, and the inputs are sensitive enough to tell the
two apart. On an NVIDIA T4 `test-gpu` passed too, and every benchmark output
matched the CPU kernel bit for bit ([`06-gpu-results.md`](06-gpu-results.md)).

### Tests

`pixi run test-gpu` (`mojo/tests/test_gpu.mojo`) compares every GPU result
with the CPU kernel **bit for bit** (`to_bits()`), on inexact inputs:

- every matmul variant on 7 shapes, including partial tiles in every
  dimension and `k` smaller than one tile;
- conv2d with batch 2, odd channel counts, strides and dilation, a 64→64
  layer, and the integration test's golden values;
- a **negative control**: the same matmul with fused multiply-add gives
  different bits on these inputs, so the exact comparison would catch a
  fused kernel;
- empty shapes and shape errors.

The CPU kernels are tested against the golden values and the C++ reference
(`pixi run test-mojo`), so the GPU inherits those numbers. Reversing the
naive kernel's `k` loop, or conv2d's channel loop, makes the tests fail at
the first output; both mutations were tried.

## ▶️ Running

### Apple silicon (M1–M5)

Needs macOS 15+, Xcode 16+ and its Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`).

```bash
pixi run test-gpu    # 8 tests, bit-exact against the CPU kernels
pixi run bench-gpu   # writes build/bench/results-gpu.json
```

### NVIDIA on Colab or Kaggle (T4)

Choose a T4 runtime (Kaggle's P100 is pre-Turing, which Mojo doesn't
support). In a notebook:

```
!nvidia-smi --query-gpu=name,driver_version --format=csv
!curl -fsSL https://pixi.sh/install.sh | bash
import os; os.environ["PATH"] = os.path.expanduser("~/.pixi/bin") + ":" + os.environ["PATH"]
!git clone --depth 1 https://github.com/joepothiboot/nano-dsp-mlir
%cd nano-dsp-mlir
```

If the driver is older than 580, point Mojo at the system `ptxas` (Colab's
T4 had 580.82.07 in October 2026, so this wasn't needed there):

```
%env MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
```

Then:

```
!pixi run test-gpu
!pixi run bench-gpu
!python benchmarks/bench_cublas.py > build/bench/results-cublas.json
from google.colab import files
files.download("build/bench/results-gpu.json"); files.download("build/bench/results-cublas.json")
```

`test-gpu` must pass first: it answers whether the NVIDIA path also honors
`contract=off`. `bench_cublas.py` uses CuPy from the notebook's own Python,
not pixi's. Copy the two files into `benchmarks/results/` as
`gpu-t4.json` and `cublas-t4.json` and fill in
[`06-gpu-results.md`](06-gpu-results.md) with `python3 scripts/gpu_table.py`.

## 🚧 Limits

- f32 only in the tests and benchmarks; the kernels are generic over `dtype`.
- One conv2d kernel, without shared-memory reuse of the input.
- The BLOCKED tile shape is fixed (64×64×16, 4×4 per thread); no sweep yet.
- Each benchmark call launches and synchronizes, so small sizes include
  launch latency.
