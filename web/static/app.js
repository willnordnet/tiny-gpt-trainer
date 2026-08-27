/* Viewer front end.
 *
 * No framework and no build step: this is one page that listens to an
 * EventSource and redraws some SVG. Charts are drawn by hand because a loss
 * curve is a polyline over a linear scale, and writing that out is more in
 * keeping with a repo whose whole premise is that the machinery should be
 * visible.
 */

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// State. Everything the charts draw lives here, keyed so that a duplicated
// event (a reconnect replaying history) overwrites rather than double-plots.
// ---------------------------------------------------------------------------

const state = {
  steps: new Map(),      // step -> {loss, lr, grad_norm, tok_s}
  evals: new Map(),      // step -> val_loss
  samples: new Map(),    // step -> text
  baseline: null,        // ln(vocab_size), from the trainer's header
  totalSteps: null,
  lastStep: 0,
  running: false,
  scrubPinned: false,    // true once the user drags, so live samples stop stealing focus
};

const sortedPairs = (map) => [...map.entries()].sort((a, b) => a[0] - b[0]);

// ---------------------------------------------------------------------------
// Tiny SVG chart helpers
// ---------------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs, text) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, value);
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Map data coordinates onto the viewBox, leaving room for axis labels. */
function makeScale(box, xRange, yRange) {
  const pad = { left: 42, right: 10, top: 10, bottom: 20 };
  const w = box.w - pad.left - pad.right;
  const h = box.h - pad.top - pad.bottom;
  const [x0, x1] = xRange;
  const [y0, y1] = yRange;
  const xSpan = x1 - x0 || 1;
  const ySpan = y1 - y0 || 1;
  return {
    pad, w, h,
    x: (v) => pad.left + ((v - x0) / xSpan) * w,
    // SVG y grows downward, so a larger loss must map to a smaller y.
    y: (v) => pad.top + h - ((v - y0) / ySpan) * h,
  };
}

function polyline(points, scale, color, width = 1.5) {
  const d = points.map(([x, y]) => `${scale.x(x).toFixed(1)},${scale.y(y).toFixed(1)}`).join(" ");
  return el("polyline", { points: d, fill: "none", stroke: color, "stroke-width": width,
                          "stroke-linejoin": "round", "stroke-linecap": "round" });
}

/** The main loss chart: train, val, the ln(vocab) reference, and the point
 *  where validation loss stops improving. */
function drawLossChart() {
  const svg = $("lossChart");
  svg.replaceChildren();
  const box = { w: 560, h: 240 };

  const train = sortedPairs(state.steps).map(([s, d]) => [s, d.loss]);
  const val = sortedPairs(state.evals);
  if (!train.length) return;

  const xs = train.map((p) => p[0]);
  const xMax = state.totalSteps || Math.max(...xs);
  const finite = [...train, ...val].map((p) => p[1]).filter(Number.isFinite);
  const yTop = Math.max(state.baseline || 0, ...finite) * 1.05;
  const yBottom = Math.max(0, Math.min(...finite) - 0.2);

  const scale = makeScale(box, [0, xMax], [yBottom, yTop]);

  // Horizontal gridlines with value labels.
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const value = yBottom + ((yTop - yBottom) * i) / ticks;
    const y = scale.y(value);
    svg.appendChild(el("line", { x1: scale.pad.left, x2: box.w - scale.pad.right,
                                 y1: y, y2: y, stroke: "#222c38", "stroke-width": 1 }));
    svg.appendChild(el("text", { x: scale.pad.left - 6, y: y + 3, "text-anchor": "end" },
                       value.toFixed(1)));
  }

  // x labels at either end.
  svg.appendChild(el("text", { x: scale.pad.left, y: box.h - 6 }, "0"));
  svg.appendChild(el("text", { x: box.w - scale.pad.right, y: box.h - 6, "text-anchor": "end" },
                     String(xMax)));

  // The "learned nothing" reference. Anything above this line means the model
  // is doing worse than guessing uniformly from the vocabulary.
  if (state.baseline && state.baseline <= yTop) {
    const y = scale.y(state.baseline);
    svg.appendChild(el("line", { x1: scale.pad.left, x2: box.w - scale.pad.right, y1: y, y2: y,
                                 stroke: "var(--baseline)", "stroke-width": 1.5,
                                 "stroke-dasharray": "5 4" }));
    svg.appendChild(el("text", { x: box.w - scale.pad.right - 2, y: y - 5,
                                 "text-anchor": "end", fill: "#9d80e8" },
                       `ln(vocab) = ${state.baseline.toFixed(2)}`));
  }

  svg.appendChild(polyline(train, scale, "var(--train)"));

  if (val.length) {
    svg.appendChild(polyline(val, scale, "var(--val)", 2));
    for (const [step, loss] of val) {
      svg.appendChild(el("circle", { cx: scale.x(step), cy: scale.y(loss), r: 2.2,
                                     fill: "var(--val)" }));
    }
    markOverfitting(svg, scale, val);
  }
}

/** Mark where validation loss bottomed out, if it has since risen.
 *
 * This is the single most instructive thing a small model on a small corpus
 * does, and it is easy to miss in a scrolling log: train loss keeps falling
 * and looks like progress long after the model stopped generalising. */
function markOverfitting(svg, scale, val) {
  let best = val[0];
  for (const point of val) if (point[1] < best[1]) best = point;

  const last = val[val.length - 1];
  const roseAgain = last[1] > best[1] * 1.05 && last[0] > best[0];
  const note = $("overfitNote");

  if (!roseAgain) { note.textContent = ""; return; }

  const x = scale.x(best[0]);
  svg.appendChild(el("line", { x1: x, x2: x, y1: scale.pad.top, y2: scale.pad.top + scale.h,
                               stroke: "var(--bad)", "stroke-width": 1, "stroke-dasharray": "3 3" }));
  svg.appendChild(el("text", { x: x + 4, y: scale.pad.top + 10, fill: "#f85149" },
                     `best val ${best[1].toFixed(2)} @ ${best[0]}`));
  note.textContent = `overfitting past step ${best[0]}`;
}

/** A minimal sparkline: no axes, just the shape. */
function drawSpark(id, points, color) {
  const svg = $(id);
  svg.replaceChildren();
  if (points.length < 2) return;
  const box = { w: 180, h: 56 };
  const values = points.map((p) => p[1]).filter(Number.isFinite);
  if (!values.length) return;

  const scale = makeScale(box, [points[0][0], points[points.length - 1][0]],
                          [Math.min(...values), Math.max(...values) || 1]);
  scale.pad.left = 4;
  svg.appendChild(polyline(points, scale, color, 1.2));
  svg.appendChild(el("text", { x: 4, y: 10 }, values[values.length - 1].toPrecision(3)));
}

function redraw() {
  drawLossChart();
  const pairs = sortedPairs(state.steps);
  drawSpark("lrChart", pairs.map(([s, d]) => [s, d.lr]), "var(--ok)");
  drawSpark("gnChart", pairs.map(([s, d]) => [s, d.grad_norm]), "var(--bad)");
  drawSpark("tsChart", pairs.map(([s, d]) => [s, d.tok_s]), "var(--ink-dim)");
}

// Redraw at most once per frame: a 2000-step run emits 200 step events plus
// evals and samples, and redrawing synchronously on each would thrash.
let redrawQueued = false;
function scheduleRedraw() {
  if (redrawQueued) return;
  redrawQueued = true;
  requestAnimationFrame(() => { redrawQueued = false; redraw(); });
}

// ---------------------------------------------------------------------------
// Raw log pane
// ---------------------------------------------------------------------------

const LOG_MAX_LINES = 4000;

function appendLog(event) {
  const pane = $("log");
  // Only autoscroll if the user is already at the bottom, so scrolling back
  // to read something does not get yanked away by the next line.
  const atBottom = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;

  const line = document.createElement("div");
  line.className = event.type;
  if (event.type === "stage_start") {
    line.textContent = `\n$ ${event.command}`;
    line.className = "stage";
  } else if (event.type === "stage_end") {
    line.textContent = `[exit ${event.returncode}] ${event.stage}`;
    line.className = event.returncode === 0 ? "stage" : "err";
  } else if (event.type === "pipeline_failed" || event.type === "pipeline_error") {
    line.textContent = JSON.stringify(event);
    line.className = "err";
  } else {
    line.textContent = event.text ?? "";
  }
  pane.appendChild(line);

  while (pane.childElementCount > LOG_MAX_LINES) pane.removeChild(pane.firstChild);
  if (atBottom) pane.scrollTop = pane.scrollHeight;
}

// ---------------------------------------------------------------------------
// Sample timeline
// ---------------------------------------------------------------------------

function refreshScrubber(jumpToLatest) {
  const samples = sortedPairs(state.samples);
  const scrub = $("scrub");
  scrub.max = Math.max(0, samples.length - 1);
  scrub.disabled = samples.length === 0;
  $("sampleCount").textContent = `${samples.length} sample${samples.length === 1 ? "" : "s"}`;
  if (samples.length && jumpToLatest && !state.scrubPinned) {
    scrub.value = samples.length - 1;
  }
  showSample();
}

function showSample() {
  const samples = sortedPairs(state.samples);
  if (!samples.length) return;
  const index = Math.min(Number($("scrub").value), samples.length - 1);
  const [step, text] = samples[index];
  $("sampleStep").textContent = String(step);
  $("sampleText").textContent = text;
}

// ---------------------------------------------------------------------------
// Event stream
// ---------------------------------------------------------------------------

let source = null;

function connect() {
  if (source) source.close();
  source = new EventSource("/api/events");
  source.onmessage = (message) => handle(JSON.parse(message.data));
  source.onerror = () => { /* EventSource reconnects on its own */ };
}

function handle(event) {
  switch (event.type) {
    case "idle":
      return;

    case "step":
      state.steps.set(event.step, event);
      state.totalSteps = event.total;
      state.lastStep = event.step;
      $("rStep").textContent = `${event.step}/${event.total}`;
      $("rTrain").textContent = event.loss.toFixed(4);
      $("rToks").textContent = Math.round(event.tok_s).toLocaleString();
      scheduleRedraw();
      break;

    case "eval":
      state.evals.set(state.lastStep, event.val_loss);
      $("rVal").textContent = event.val_loss.toFixed(4);
      scheduleRedraw();
      break;

    case "baseline":
      state.baseline = event.baseline;
      scheduleRedraw();
      break;

    case "sample":
      state.samples.set(event.step, event.text);
      refreshScrubber(true);
      break;

    case "checkpoint":
      loadCheckpoints();
      break;

    case "gate":
      setBadge("badgeGate", event.passed ? "gate: PASS" : "gate: FAIL",
               event.passed ? "ok" : "bad");
      break;

    case "stage_start":
      setBadge("badgeStage", `stage: ${event.stage} (${event.index + 1}/${event.total})`, "on");
      break;

    case "pipeline_done":
      setRunning(false);
      setBadge("badgeState", "done", "ok");
      loadCheckpoints();
      break;

    case "pipeline_failed":
      setRunning(false);
      setBadge("badgeState", `failed in ${event.stage}`, "bad");
      showError(`Stage "${event.stage}" exited ${event.returncode}. See the raw log.`);
      break;

    case "pipeline_error":
      setRunning(false);
      setBadge("badgeState", "error", "bad");
      showError(event.error);
      break;

    case "eof":
      setRunning(false);
      break;

    case "replayed":
      return;   // bookkeeping only; nothing to show
  }

  if (event.text !== undefined || event.type.startsWith("stage_") ||
      event.type.startsWith("pipeline_")) {
    appendLog(event);
  }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function setBadge(id, text, cls) {
  const badge = $(id);
  badge.textContent = text;
  badge.className = `badge${cls ? " " + cls : ""}`;
}

function showError(message) {
  const banner = $("error");
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function clearError() { $("error").classList.add("hidden"); }

function setRunning(running) {
  state.running = running;
  $("start").disabled = running;
  $("stop").disabled = !running;
  if (running) setBadge("badgeState", "running", "on");
  else setBadge("badgeStage", "—", "");
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

$("start").onclick = async () => {
  clearError();
  try {
    // Reset, so a second run does not draw on top of the first.
    state.steps.clear(); state.evals.clear(); state.samples.clear();
    state.baseline = null; state.scrubPinned = false;
    $("log").replaceChildren();
    refreshScrubber(true); redraw();
    setBadge("badgeGate", "gate: not run", "");

    let source = null;
    const file = $("file").files[0];
    if (file) {
      setBadge("badgeState", "uploading", "on");
      const uploaded = await fetch("/api/upload", { method: "POST", body: file })
        .then(async (r) => { const d = await r.json();
                             if (!r.ok) throw new Error(d.error); return d; });
      source = uploaded.source;
      setBadge("badgeSource", `corpus: ${uploaded.bytes.toLocaleString()} bytes`, "on");
    } else {
      setBadge("badgeSource", "corpus: existing tokens", "");
    }

    setRunning(true);
    await postJSON("/api/run", {
      source,
      preset: $("preset").value,
      steps: Number($("steps").value),
      prompt: $("prompt").value,
      overfit_gate: $("gate").checked,
      new_tokenizer: $("newTokenizer").checked,
    });
    connect();
  } catch (error) {
    setRunning(false);
    setBadge("badgeState", "error", "bad");
    showError(String(error.message || error));
  }
};

$("stop").onclick = async () => {
  await postJSON("/api/stop");
  setBadge("badgeState", "stopping", "bad");
};

$("scrub").oninput = () => { state.scrubPinned = true; showSample(); };

// ---------------------------------------------------------------------------
// Checkpoints, next-token lab, attention
// ---------------------------------------------------------------------------

// Every panel that picks a checkpoint is filled from one fetch, so a new
// checkpoint written mid-run appears in all of them at once.
const CHECKPOINT_SELECTS = ["labCkpt", "genCkpt"];

async function loadCheckpoints() {
  const { checkpoints } = await fetch("/api/checkpoints").then((r) => r.json());

  for (const id of CHECKPOINT_SELECTS) {
    const select = $(id);
    const previous = select.value;
    select.replaceChildren();
    for (const checkpoint of checkpoints) {
      const option = document.createElement("option");
      option.value = checkpoint.path;
      const val = checkpoint.val_loss === null ? "" : ` · val ${checkpoint.val_loss.toFixed(3)}`;
      option.textContent = `${checkpoint.preset} step ${checkpoint.step}${val}`;
      select.appendChild(option);
    }
    // Keep the user's choice across a reload if that checkpoint still exists.
    if (previous && checkpoints.some((c) => c.path === previous)) select.value = previous;
  }

  if ($("labCkpt").value) refreshLab();
}

$("refreshCkpts").onclick = loadCheckpoints;

// Debounce: dragging a slider fires continuously, and each request is a real
// forward pass. 120ms is short enough to feel live and long enough that a
// drag across the track is a handful of requests rather than fifty.
function debounce(fn, ms) {
  let timer = null;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

async function refreshLab() {
  const checkpoint = $("labCkpt").value;
  if (!checkpoint) return;

  const topk = Number($("topk").value);
  const topp = Number($("topp").value);
  const temp = Number($("temp").value);
  $("tempVal").textContent = temp.toFixed(2);
  $("topkVal").textContent = topk === 0 ? "off" : String(topk);
  $("toppVal").textContent = topp.toFixed(2);

  let data;
  try {
    data = await postJSON("/api/next-token", {
      checkpoint, prompt: $("labPrompt").value,
      temperature: temp, top_k: topk, top_p: topp,
    });
  } catch (error) { showError(String(error.message || error)); return; }

  $("entropy").textContent = data.entropy.toFixed(3);
  $("entropyRaw").textContent = data.entropy_raw.toFixed(3);
  $("entropyUniform").textContent = data.uniform_entropy.toFixed(3);

  const bars = $("bars");
  bars.replaceChildren();
  const peak = Math.max(...data.candidates.map((c) => Math.max(c.prob, c.prob_after)), 1e-9);

  for (const candidate of data.candidates) {
    const row = document.createElement("div");
    row.className = "bar-row" + (candidate.eliminated ? " cut" : "");

    const token = document.createElement("span");
    token.className = "tok";
    token.textContent = JSON.stringify(candidate.token);

    const track = document.createElement("span");
    track.className = "bar-track";
    const raw = document.createElement("span");
    raw.className = "bar-raw";
    raw.style.width = `${(candidate.prob / peak) * 100}%`;
    const after = document.createElement("span");
    after.className = "bar-after";
    after.style.width = `${(candidate.prob_after / peak) * 100}%`;
    track.append(raw, after);

    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = `${(candidate.prob_after * 100).toFixed(1)}%`;

    row.append(token, track, pct);
    bars.appendChild(row);
  }

  // Fill the attention layer/head pickers from whatever this checkpoint is.
  populateAttnPickers();
}

const refreshLabSoon = debounce(refreshLab, 120);
for (const id of ["temp", "topk", "topp"]) $(id).oninput = refreshLabSoon;
$("labPrompt").oninput = refreshLabSoon;
$("labCkpt").onchange = refreshLab;

$("commit").onclick = () => {
  // Append the highest-probability surviving token and re-run: autoregression,
  // one step at a time, by hand.
  const first = $("bars").querySelector(".bar-row:not(.cut) .tok");
  if (!first) return;
  $("labPrompt").value += JSON.parse(first.textContent);
  refreshLab();
};

let attnMeta = { n_layers: 6, n_heads: 4 };

function populateAttnPickers() {
  for (const [id, count] of [["attnLayer", attnMeta.n_layers], ["attnHead", attnMeta.n_heads]]) {
    const select = $(id);
    if (select.childElementCount === count) continue;
    const previous = select.value;
    select.replaceChildren();
    for (let i = 0; i < count; i++) {
      const option = document.createElement("option");
      option.value = String(i);
      option.textContent = String(i);
      select.appendChild(option);
    }
    if (previous) select.value = Math.min(Number(previous), count - 1);
  }
}

$("drawAttn").onclick = async () => {
  const checkpoint = $("labCkpt").value;
  if (!checkpoint) { showError("No checkpoint selected."); return; }

  let data;
  try {
    data = await postJSON("/api/attention", {
      checkpoint, prompt: $("attnPrompt").value,
      layer: Number($("attnLayer").value), head: Number($("attnHead").value),
    });
  } catch (error) { showError(String(error.message || error)); return; }

  attnMeta = { n_layers: data.n_layers, n_heads: data.n_heads };
  populateAttnPickers();

  const table = document.createElement("table");
  table.className = "attn";

  const header = document.createElement("tr");
  header.appendChild(document.createElement("th"));
  for (const token of data.tokens) {
    const th = document.createElement("th");
    th.className = "colhead";
    th.textContent = token.replace(/\n/g, "\\n").slice(0, 6);
    header.appendChild(th);
  }
  table.appendChild(header);

  data.weights.forEach((row, i) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.className = "rowhead";
    th.textContent = data.tokens[i].replace(/\n/g, "\\n").slice(0, 10);
    tr.appendChild(th);
    for (const weight of row) {
      const td = document.createElement("td");
      // Square-root the weight for display only. Attention is usually spiky,
      // and on a linear ramp everything but the peak reads as black; this
      // keeps the small-but-nonzero weights visible without claiming they
      // are larger than they are.
      td.style.background = weight > 0
        ? `rgba(106,169,255,${Math.sqrt(weight).toFixed(3)})`
        : "transparent";
      td.title = weight.toFixed(4);
      tr.appendChild(td);
    }
    table.appendChild(tr);
  });

  const wrap = $("attnWrap");
  wrap.replaceChildren(table);
  if (data.truncated) {
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = `Prompt truncated to its last ${data.tokens.length} tokens.`;
    wrap.appendChild(note);
  }
};

// ---------------------------------------------------------------------------
// Generation panel
//
// The server answers /api/generate with newline-delimited JSON rather than
// server-sent events, because EventSource can only issue GET requests and the
// prompt belongs in a body. So this reads the response as a stream and splits
// it on newlines itself.
// ---------------------------------------------------------------------------

let genAbort = null;

function syncGenLabels() {
  $("genTempVal").textContent = Number($("genTemp").value).toFixed(2);
  const topk = Number($("genTopk").value);
  $("genTopkVal").textContent = topk === 0 ? "off" : String(topk);
  $("genToppVal").textContent = Number($("genTopp").value).toFixed(2);
}
for (const id of ["genTemp", "genTopk", "genTopp"]) $(id).oninput = syncGenLabels;
syncGenLabels();

function setGenerating(active) {
  $("genRun").disabled = active;
  $("genStop").disabled = !active;
}

$("genStop").onclick = () => {
  // Aborting the fetch is the whole stop mechanism. The server's next write
  // raises BrokenPipeError inside the sink, which unwinds out of generate()
  // and ends the loop -- so no stop endpoint is needed.
  if (genAbort) genAbort.abort();
};

$("genRun").onclick = async () => {
  const checkpoint = $("genCkpt").value;
  if (!checkpoint) { showError("No checkpoint to generate from yet."); return; }
  clearError();

  const prompt = $("genPrompt").value;
  const output = $("genOut");

  // Echo the prompt dim, then stream the continuation bright after it, so the
  // whole thing reads as one passage while it stays obvious where the model
  // took over. The server sends only the continuation.
  output.replaceChildren();
  const echo = document.createElement("span");
  echo.className = "echo";
  echo.textContent = prompt;
  const written = document.createElement("span");
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  cursor.textContent = " ";
  output.append(echo, written, cursor);

  $("genStats").textContent = "generating…";
  setGenerating(true);
  genAbort = new AbortController();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: genAbort.signal,
      body: JSON.stringify({
        checkpoint,
        prompt,
        max_tokens: Number($("genMax").value),
        temperature: Number($("genTemp").value),
        top_k: Number($("genTopk").value),
        top_p: Number($("genTopp").value),
      }),
    });

    // An error is answered as a normal JSON body before any stream starts, so
    // the page never has to guess about a half-written response.
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.error || response.statusText);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });

      // A chunk can split a line, so keep the unterminated tail for next time.
      const lines = buffered.split("\n");
      buffered = lines.pop();

      for (const line of lines) {
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.delta !== undefined) {
          written.textContent += event.delta;
          output.scrollTop = output.scrollHeight;
        } else if (event.done) {
          $("genStats").textContent =
            `step ${event.step} · prompt ${event.prompt_tokens} tokens · ` +
            `context ${event.context_len} · ${event.tokens_per_second} tok/s · ` +
            `${event.seconds}s` +
            (event.truncated
              ? ` · TRUNCATED: prompt + generation exceeds the ${event.context_len}-token ` +
                `context, so the model lost the start of its own prompt partway through`
              : "");
        }
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      $("genStats").textContent = "stopped";
    } else {
      $("genStats").textContent = "—";
      showError(String(error.message || error));
    }
  } finally {
    cursor.remove();
    setGenerating(false);
    genAbort = null;
  }
};

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

(async function boot() {
  const status = await fetch("/api/status").then((r) => r.json());
  setRunning(status.running);
  if (status.running) connect();
  await loadCheckpoints();
})();
