const DATA = JSON.parse(document.getElementById("demo-data").textContent);
const TARGETS = Object.keys(DATA.targets);
let state = { target: TARGETS[0], level: 2 };

try {
  const t = localStorage.getItem("nd-target");
  if (TARGETS.includes(t)) state.target = t;
} catch (e) {}

const esc = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const kib = (b) =>
  ((b / 1024) % 1 === 0 ? b / 1024 : (b / 1024).toFixed(1)) + " KiB";

const fmt = (n) => n.toLocaleString("en-US");

function highlight(src) {
  return esc(src)
    .replace(/vector&lt;[^&]*&gt;/g, (m) => `<span class="t-vec">${m}</span>`)
    .replace(
      /(tile_sizes )\[([\d, ]+)\]/g,
      (_, a, b) => `${a}[<span class="t-num">${b}</span>]`,
    )
    .replace(
      /\b(arith\.mulf|arith\.addf|transform\.structured\.\w+|fmul\.4s|fadd\.4s|vmulps|vaddps)\b/g,
      (m) => `<span class="t-op">${m}</span>`,
    );
}

const LEVELS = [
  {
    tag: "L1",
    name: "dsp",
    cap: () =>
      "What the user writes: one dsp.matmul on static-shape tensors. No loops yet.",
    code: () => DATA.levels.dsp,
  },
  {
    tag: "L2",
    name: "linalg.generic",
    cap: () =>
      "-convert-dsp-to-linalg: one linalg.generic with (m, n, k) indexing maps and an unfused multiply-add body. This uniform op is what schedules target.",
    code: () => DATA.levels.linalg,
  },
  {
    tag: "L3",
    name: "Tiled + vectorized",
    cap: () =>
      `-nanodsp-optimize=target=${state.target}: cache-tile loops around register-tile loops around a vector multiply and add. Switch the target to see the loop nest change.`,
    code: () => DATA.targets[state.target].l3,
  },
  {
    tag: "L5",
    name: "Machine code",
    cap: () =>
      `-nanodsp-lower-to-llvm, then llc for ${DATA.targets[state.target].isa}: the inner loop body. Vector multiplies and adds, no scalar multiplies, no FMA.`,
    code: () => DATA.targets[state.target].asm,
  },
];

function renderSegs() {
  document.querySelectorAll("[data-seg]").forEach((seg) => {
    seg.innerHTML = "";

    TARGETS.forEach((t) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = t;
      b.setAttribute("aria-pressed", String(t === state.target));

      b.addEventListener("click", () => {
        state.target = t;

        try {
          localStorage.setItem("nd-target", t);
        } catch (e) {}

        render();
      });

      seg.appendChild(b);
    });
  });
}

function renderLevels() {
  const wrap = document.getElementById("levels");
  wrap.innerHTML = "";

  LEVELS.forEach((lv, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "level";
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(i === state.level));
    b.innerHTML = `<span class="tag">${lv.tag}</span><span class="name">${esc(lv.name)}</span>`;

    b.addEventListener("click", () => {
      state.level = i;
      render();
    });

    wrap.appendChild(b);
  });

  const lv = LEVELS[state.level];
  document.getElementById("level-caption").textContent = lv.cap();
  document.getElementById("level-code").innerHTML = highlight(lv.code());
}

function renderTarget() {
  const t = DATA.targets[state.target];
  const dims = ["m", "n", "k"];
  const tile = (a) => a.join(" × ");

  const stats = [
    [
      "SIMD",
      `${t.vectorBits}-bit`,
      `${t.regs} registers × ${t.lanes} f32 lanes`,
    ],
    [
      "Register tile (m × n)",
      tile(t.regTile.slice(0, 2)),
      `k = ${t.regTile[2]} keeps the sum order`,
    ],
    [
      "Cache tile (m × n × k)",
      tile(t.cacheTile),
      `of ${tile(t.loopRanges)} loops`,
    ],
    [
      "Vector multiplies",
      fmt(t.asmStats.vmul),
      `${fmt(t.asmStats.smul)} scalar · ${fmt(t.asmStats.fma)} FMA`,
    ],
  ];

  document.getElementById("stats").innerHTML = stats
    .map(
      ([k, v, n]) =>
        `<div class="stat"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span><span class="n">${esc(n)}</span></div>`,
    )
    .join("");

  const pctWs = (t.workingSetBytes / t.cacheBytes) * 100;
  const pctBudget = (t.budgetBytes / t.cacheBytes) * 100;
  document.getElementById("bar-fill").style.width = pctWs + "%";
  document.getElementById("bar-mark").style.left = `calc(${pctBudget}% - 1px)`;

  document
    .getElementById("bar")
    .setAttribute(
      "aria-label",
      `Cache tile working set ${kib(t.workingSetBytes)} of a ${kib(t.budgetBytes)} budget in a ${kib(t.cacheBytes)} cache`,
    );

  document.getElementById("bar-left").textContent =
    `cache tile working set: ${kib(t.workingSetBytes)}`;

  document.getElementById("bar-right").textContent =
    `budget ${kib(t.budgetBytes)} = ${Math.round(t.cacheFraction * 100)}% of ${kib(t.cacheBytes)} L1D`;

  document.getElementById("sched").innerHTML = highlight(t.schedule);
  document.getElementById("asm").innerHTML = highlight(t.asm);
  document.getElementById("asm-label").textContent = `Inner loop, ${t.isa}`;
}

function renderStatic() {
  const b = [];

  if (DATA.tests) {
    b.push([
      `${DATA.tests.passed}/${DATA.tests.total} lit tests pass`,
      DATA.tests.passed === DATA.tests.total,
    ]);
  }

  const exact = DATA.proof.runs.every(
    (r) => (r.expect === "same") === (r.differ === 0),
  );

  b.push([exact ? "bit-exact" : "bit-exact check failed", exact]);
  b.push([`LLVM ${DATA.llvm}`, false]);

  b.push([
    `${DATA.shape.m}×${DATA.shape.k} · ${DATA.shape.k}×${DATA.shape.n} f32 matmul`,
    false,
  ]);

  document.getElementById("badges").innerHTML = b
    .map(([t, ok]) => `<span class="badge${ok ? " ok" : ""}">${esc(t)}</span>`)
    .join("");

  document.getElementById("proof-sub").textContent =
    `Matmul, three conv2d variants and add + relu, run unscheduled and under each schedule below. All ${fmt(DATA.proof.values)} outputs are compared as raw 32-bit patterns.`;

  document.getElementById("runs").innerHTML = DATA.proof.runs
    .map((r) => {
      const good = (r.expect === "same") === (r.differ === 0);

      const asExpected = r.expect === "same" ? good : !good;
      const cls = asExpected ? "ok" : "bad";

      const res =
        r.differ === 0 ? "0 values differ" : `${fmt(r.differ)} values differ`;

      return `<div class="run ${cls}"><span class="who">${esc(r.name)}</span><span class="what">${esc(r.desc)}</span><span class="res">${res}</span></div>`;
    })
    .join("");

  const ms = (ns) =>
    ns >= 1e6 ? `${(ns / 1e6).toFixed(2)} ms` : `${(ns / 1e3).toFixed(0)} µs`;

  const big = (DATA.gpu || []).filter(
    (r) => r.op === "matmul" && r.shape === "2048x2048x2048",
  );

  const chart = document.getElementById("gpu-chart");

  if (big.length) {
    const top = Math.max(...big.map((r) => r.rate));

    const name = (r) =>
      r.impl === "cublas" ? "cuBLAS" : r.impl.replace("mojo-gpu-", "");

    const devices = [...new Set(big.map((r) => r.config))];

    chart.setAttribute(
      "aria-label",
      "Matmul 2048³ throughput: " +
        big
          .map((r) => `${r.config} ${name(r)} ${r.rate.toFixed(0)} GFLOP/s`)
          .join(", "),
    );

    chart.innerHTML =
      `<div class="gpu-legend"><span><i style="background:var(--series-ours)"></i>Mojo kernel, bit-exact</span>` +
      `<span><i style="background:var(--series-lib)"></i>cuBLAS, within the error bound</span>` +
      `<span>matmul 2048³, GFLOP/s, higher is faster</span></div>` +
      devices
        .map(
          (d) =>
            `<div class="gpu-group">${esc(d)}</div>` +
            big
              .filter((r) => r.config === d)
              .map((r) => {
                const pct = (100 * r.rate) / top;

                const color =
                  r.impl === "cublas"
                    ? "var(--series-lib)"
                    : "var(--series-ours)";

                const tip = `${esc(d)} · ${esc(name(r))}: ${ms(r.median_time)} median, ${r.rate.toFixed(0)} GFLOP/s, ${esc(r.checked)}`;

                return `<div class="gpu-row" data-tip="${tip}"><span class="lbl">${esc(name(r))}</span><span class="gpu-track"><span class="gpu-bar" style="width:calc((100% - 48px) * ${(pct / 100).toFixed(4)});background:${color}"></span><span class="gpu-val">${Math.round(r.rate).toLocaleString("en-US")}</span></span></div>`;
              })
              .join(""),
        )
        .join("") +
      `<div class="gpu-tip" id="gpu-tip"></div>`;

    const tipEl = document.getElementById("gpu-tip");

    chart.querySelectorAll(".gpu-row").forEach((row) => {
      row.addEventListener("mousemove", (e) => {
        const box = chart.getBoundingClientRect();
        tipEl.innerHTML = row.dataset.tip;
        tipEl.style.display = "block";

        const x = Math.min(
          e.clientX - box.left + 12,
          box.width - tipEl.offsetWidth,
        );

        tipEl.style.left = `${Math.max(0, x)}px`;
        tipEl.style.top = `${e.clientY - box.top + 14}px`;
      });

      row.addEventListener("mouseleave", () => (tipEl.style.display = "none"));
    });
  }

  document.getElementById("gpu-runs").innerHTML = (DATA.gpu || [])
    .map((r) => {
      const shape = r.shape
        .replace("2048x2048x2048", "2048³")
        .replace("->", " → ");

      const variant = r.impl
        .replace("mojo-gpu-", "")
        .replace("mojo-gpu", "one thread per output");

      return `<div class="run ok"><span class="who">${esc(r.op)} ${esc(shape)}</span><span class="what">${esc(variant)} · ${esc(r.config)} · ${ms(r.median_time)} · ${esc(r.checked).replace("-", "\u2011")}</span><span class="res">${r.rate.toFixed(0)} GFLOP/s</span></div>`;
    })
    .join("");

  document.getElementById("footer").innerHTML =
    `<span>Every IR listing, tile size, instruction count and bit count on this page is real tool output, generated on ${esc(DATA.generated)} from commit <code>${esc(DATA.commit)}</code>.</span>` +
    `<span>Regenerate with <code>python3 scripts/gen_demo.py</code>.</span>`;
}

function render() {
  renderSegs();
  renderLevels();
  renderTarget();
}

renderStatic();
render();
