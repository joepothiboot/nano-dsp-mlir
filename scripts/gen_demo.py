#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_INPUT = ROOT / "demo" / "matmul.mlir"
BIT_EXACT = ROOT / "test" / "Integration" / "Schedule" / "bit-exact.mlir"

MACHINES = {
    "host-neon": {
        "llc": ["-mtriple=aarch64-apple-darwin"],
        "isa": "AArch64 NEON",
        "vmul": r"^\s+fmul\.4s\b",
        "vadd": r"^\s+fadd\.4s\b",
        "smul": r"^\s+fmul\s+s\d",
        "fma": r"^\s+(fmla|fmadd)\b",
    },
    "x86-avx2": {
        "llc": ["-mtriple=x86_64-unknown-linux-gnu", "-mattr=+avx2"],
        "isa": "x86-64 AVX2",
        "vmul": r"^\s+vmulps\b.*%ymm",
        "vadd": r"^\s+vaddps\b.*%ymm",
        "smul": r"^\s+v?mulss\b",
        "fma": r"^\s+vfmadd",
    },
}

STOCK_LOWERING = [
    "-one-shot-bufferize=bufferize-function-boundaries",
    "-buffer-deallocation-pipeline", "-convert-linalg-to-loops",
    "-convert-scf-to-cf", "-expand-strided-metadata", "-lower-affine",
    "-convert-arith-to-llvm", "-finalize-memref-to-llvm",
    "-convert-func-to-llvm", "-convert-cf-to-llvm",
    "-reconcile-unrealized-casts",
]


def find_llvm_bin():
    for cand in [os.environ.get("LLVM_BIN"), "/opt/homebrew/opt/llvm/bin",
                 "/usr/local/opt/llvm/bin"]:
        if cand and (Path(cand) / "llc").exists():
            return Path(cand)

    llc = shutil.which("llc")

    if llc:
        return Path(llc).parent

    sys.exit("Can't find llc. Set LLVM_BIN to the LLVM bin directory.")


def run(cmd, stdin=None):
    proc = subprocess.run([str(c) for c in cmd], input=stdin, cwd=ROOT,
                          capture_output=True, text=True)

    if proc.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, cmd))}\n{proc.stderr}")

    return proc.stdout


def strip_comments(text):
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("//")]

    return "\n".join(lines).strip() + "\n"


def target_models():
    src = (ROOT / "lib" / "Schedule" / "TargetModel.cpp").read_text()
    pat = re.compile(
        r'\{"(?P<name>[\w-]+)",\s*/\*vectorBits=\*/(?P<bits>\d+),\s*'
        r"/\*numVectorRegs=\*/(?P<regs>\d+),\s*/\*cacheBytes=\*/(?P<cache>[\d *]+),"
        r"\s*/\*cacheFraction=\*/(?P<frac>[\d.]+)")
    models = {}

    for m in pat.finditer(src):
        cache = eval(m["cache"])
        models[m["name"]] = {
            "vectorBits": int(m["bits"]), "regs": int(m["regs"]),
            "lanes": int(m["bits"]) // 32, "cacheBytes": cache,
            "budgetBytes": int(cache * float(m["frac"])),
            "cacheFraction": float(m["frac"]),
        }

    return models


def i64_array(attr_text, name):
    m = re.search(name + r" = array<i64: ([\d, ]+)>", attr_text)

    return [int(x) for x in m[1].split(",")]


def asm_excerpt(asm, vmul_re):
    lines = asm.splitlines()
    muls = [i for i, l in enumerate(lines) if re.search(vmul_re, l)]

    if not muls:
        return ""

    start = muls[0]

    while start > 0 and not re.match(r"^(\.?LBB|[;#] %bb)", lines[start]):
        start -= 1

    end = muls[-1]

    while end < len(lines) - 1 and not re.match(r"^\s+(b|b\.\w+|j\w+)\s", lines[end]):
        end += 1

    body = [l.split(";")[0].split("#")[0].rstrip() if not re.match(r"^(\.?LBB|[;#] %bb)", l) else l.split(";")[0].rstrip()
            for l in lines[start:end + 1]]

    return "\n".join(l for l in body if l.strip()) + "\n"


def count_values(printed):
    return re.findall(r"-?\d+", "\n".join(
        l for l in printed.splitlines() if not l.startswith("Unranked")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-dir", default=str(ROOT / "build"))
    ap.add_argument("--skip-tests", action="store_true",
                    help="don't run ./test.sh (the test badge is omitted)")
    args = ap.parse_args()

    build = Path(args.build_dir)
    opt = build / "bin" / "nanodsp-opt"

    if not opt.exists():
        sys.exit(f"{opt} not found. Run ./test.sh first.")

    llvm = find_llvm_bin()
    translate, llc = llvm / "mlir-translate", llvm / "llc"
    mlir_opt, runner = llvm / "mlir-opt", llvm / "mlir-runner"
    libdir = llvm.parent / "lib"
    shlib = ".dylib" if sys.platform == "darwin" else ".so"
    runner_args = ["-e", "main", "--entry-point-result=void",
                   f"--shared-libs={libdir / ('libmlir_runner_utils' + shlib)}",
                   f"--shared-libs={libdir / ('libmlir_c_runner_utils' + shlib)}"]

    source = DEMO_INPUT.read_text()
    linalg = run([opt, DEMO_INPUT, "-convert-dsp-to-linalg"])
    shape = re.search(r"tensor<(\d+)x(\d+)xf32>, %b: tensor<\d+x(\d+)xf32>", source)
    m, k, n = (int(x) for x in shape.groups())

    models = target_models()
    targets = {}

    for name, mach in MACHINES.items():
        emitted = run([opt, DEMO_INPUT, "-convert-dsp-to-linalg",
                       f"-nanodsp-emit-schedule=target={name}"])
        generic = next(l for l in emitted.splitlines() if "nanodsp.tag" in l)
        sched = emitted[emitted.index("  module attributes {transform.with_named_sequence}"):]
        sched = "\n".join(l[2:] for l in sched.rstrip().splitlines()[:-1]) + "\n"
        l3 = run([opt, DEMO_INPUT, "-convert-dsp-to-linalg",
                  f"-nanodsp-optimize=target={name}"])
        llvm_dialect = run([opt, DEMO_INPUT, "-convert-dsp-to-linalg",
                            f"-nanodsp-optimize=target={name}",
                            "-nanodsp-lower-to-llvm"])
        asm = run([llc, "-O2", *mach["llc"]],
                  stdin=run([translate, "--mlir-to-llvmir"], stdin=llvm_dialect))
        count = lambda key: len(re.findall(mach[key], asm, re.M))
        targets[name] = {
            **models[name],
            "isa": mach["isa"],
            "loopRanges": i64_array(generic, "nanodsp.loop_ranges"),
            "cacheTile": i64_array(generic, "nanodsp.cache_tile"),
            "regTile": i64_array(generic, "nanodsp.reg_tile"),
            "workingSetBytes": int(re.search(
                r"nanodsp.working_set_bytes = (\d+)", generic)[1]),
            "schedule": sched,
            "l3": l3,
            "asm": asm_excerpt(asm, mach["vmul"]),
            "asmStats": {k2: count(k2) for k2 in ("vmul", "vadd", "smul", "fma")},
        }

    def execute(extra, lowering):
        ir = run([opt, BIT_EXACT, "-convert-dsp-to-linalg", *extra, *lowering[0]])

        if lowering[1]:
            ir = run([mlir_opt, *lowering[1]], stdin=ir)

        return count_values(run([runner, *runner_args], stdin=ir))

    ref = execute([], ([], STOCK_LOWERING))
    ours = (["-nanodsp-lower-to-llvm"], None)
    runs = [
        ("host-neon", "Generated schedule", ["-nanodsp-optimize=target=host-neon"], "same"),
        ("x86-avx2", "Generated schedule", ["-nanodsp-optimize=target=x86-avx2"], "same"),
        ("matmul-8x12-neon", "Hand-written schedule",
         ["-nanodsp-optimize=schedule-file=schedules/matmul-8x12-neon.mlir"], "same"),
        ("split_reduction", "Negative control: reorders the sum",
         ["-nanodsp-optimize=schedule-file="
          "test/Integration/Schedule/Inputs/split-reduction.mlir"], "differ"),
    ]
    proof = {"values": len(ref), "runs": []}

    for label, desc, extra, expect in runs:
        vals = execute(extra, ours)
        differ = sum(a != b for a, b in zip(ref, vals)) + abs(len(ref) - len(vals))
        proof["runs"].append({"name": label, "desc": desc, "expect": expect,
                              "differ": differ})

    tests = None

    if not args.skip_tests:
        out = subprocess.run([ROOT / "test.sh"], capture_output=True, text=True).stdout
        passed = re.search(r"Passed\s*:\s*(\d+)", out)
        failed = re.search(r"Failed\s*:\s*(\d+)", out)

        if passed:
            tests = {"passed": int(passed[1]),
                     "total": int(passed[1]) + (int(failed[1]) if failed else 0)}

    gpu = []

    for path in sorted((ROOT / "benchmarks" / "results").glob("*.json"), key=lambda p: (p.name.startswith("cublas"), p.name)):
        for b in json.loads(path.read_text())["benchmarks"]:
            if b["op"] == "conv2d" or b["shape"] == "2048x2048x2048":
                gpu.append({k2: b[k2] for k2 in ("op", "shape", "impl", "config", "rate", "median_time", "checked")})

    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    version = run([llc, "--version"])
    data = {
        "generated": datetime.date.today().isoformat(),
        "commit": commit,
        "llvm": re.search(r"LLVM version (\S+)", version)[1],
        "tests": tests,
        "shape": {"m": m, "n": n, "k": k},
        "levels": {"dsp": strip_comments(source), "linalg": linalg},
        "targets": targets,
        "proof": proof,
        "gpu": gpu,
    }

    template = (ROOT / "demo" / "template.html").read_text()
    payload = json.dumps(data).replace("</", "<\\/")
    out_dir = build / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(template.replace("__DEMO_DATA__", payload))
    print(out_dir / "index.html")


if __name__ == "__main__":
    main()
