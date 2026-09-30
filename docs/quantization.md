# int8 quantization: `dsp.qmatmul` 🔢

`dsp.qmatmul` is the dialect's one integer op: an int8 matmul with per-tensor
zero points and fixed-point requantization. Integer pipelines on DSPs and
NPUs (Hexagon HVX, Ethos-U, most mobile NPUs) and TFLite's quantized kernels
compute a quantized fully connected layer this way.

## 📐 Semantics

A real value is `scale · (q − zero_point)`, with one scale and zero point per
tensor.

```
acc[i, j] = Σₖ (lhs[i, k] − lhs_zp) · (rhs[k, j] − rhs_zp)        i32
s         = 31 + shift
scaled    = (acc · multiplier + 2^(s−1)) >> s                      i64
out[i, j] = clamp(out_zp + scaled, −128, 127)                      i8
```

- **The rescale is fixed point.** `lhs_scale · rhs_scale / out_scale` is
  encoded as a Q0.31 `multiplier`, normalized to `[2^30, 2^31)`, plus a right
  `shift` in `[0, 31]`. The represented scale is in `[2^-32, 1)`. There's no
  floating point anywhere, which is the point on hardware without a fast FPU
  path.
- **Rounding is half up.** `>>` is an arithmetic shift, which is floor
  division, so adding `2^(s−1)` first rounds ties toward +∞: 0.5 → 1,
  −100.5 → −100. This is TFLite's single-rounding
  `MultiplyByQuantizedMultiplier`. It rounds once, from the exact product,
  unlike gemmlowp's double rounding.
- **The i32 accumulator can't overflow.** Each factor is at most 255 in
  magnitude, so the verifier requires `K · 255² < 2^31` (`K ≤ 33025`).
- **Saturation, not wraparound.** Values outside int8 clamp to −128 or 127.

The verifier also checks zero points against `[−128, 127]`, the multiplier's
normalization and the shift range. ODS's `I32Attr` accessors return
`uint32_t`, so every read sign-extends explicitly. The first version of the
lowering didn't, and an `out_zp` of −5 became 4294967291.

## 🔽 Lowering

`-convert-dsp-to-linalg` emits two `linalg.generic` ops. It's the one op that
isn't a single generic:

1. **Accumulate:** iterators `(m, n, k)`, destination a zero-filled
   `tensor<MxNxi32>`, body `acc + (extsi(a) − lhs_zp) · (extsi(b) − rhs_zp)`.
2. **Requantize:** elementwise `i32 → i8`, body `extsi` to i64, `muli`,
   `addi` of the rounding term, `shrsi`, `addi` of `out_zp`, `maxsi`/`minsi`,
   `trunci`.

Keeping them separate means the accumulate has the same shape as the f32
matmul, so Stage 3 tiles and vectorizes it with no special case, and the
requantize is an ordinary elementwise op. Integer arithmetic is associative,
so any schedule is bit-exact here. The integration test still runs both the
unscheduled and the scheduled path against the goldens.

## 🧪 Three implementations, one set of numbers

| Implementation | Where                                                     |
| -------------- | --------------------------------------------------------- |
| MLIR lowering  | `test/Integration/DSPToLinalg/qmatmul.mlir`               |
| Mojo SIMD      | `mojo/nanodsp/quant.mojo`, `mojo/tests/`                  |
| C++ reference  | `reference/nanodsp_ref.h`, `reference/test_reference.cpp` |

Two golden cases are shared by all three:

- **Case 1:** zero points 3 and −2, scale 1/16, `out_zp` −5. It hits
  saturation at both ends and both rounding ties.
- **Case 2:** zero points 0, `multiplier` 1518500250 (≈ 0.7071), shift 5. It
  exercises a multiplier that isn't a power of two.

The Mojo kernel is also checked against a naive loop nest on a 5×37×19
problem that spans the full int8 range.

## 🚧 Not covered yet

- **Per-channel quantization.** There's one scale per tensor, not one per
  output column. Per-channel is what most int8 conv and matmul layers use in
  practice.
- **Scales ≥ 1.** They would need a left shift, which requantization after a
  matmul rarely needs.
- **Bias add.** An i32 bias before requantization isn't included.
- **No HVX-specific lowering yet.** On Hexagon, the accumulate maps to
  `vrmpy` (4-way int8 dot products into i32). That's P3.
