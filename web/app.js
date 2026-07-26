import { loadModel } from "./src/model.js";
import { BeatTracker } from "./src/tracker.js";
import { AccentProfile } from "./src/accent.js";
import { BeatGrid } from "./src/grid.js";
import { estimateBar, MeterTally, CANDIDATES } from "./src/downbeat_decode.js";

const CIRCUMFERENCE = 2 * Math.PI * 98;
const els = {
  status: document.getElementById("status"),
  statusText: document.getElementById("statusText"),
  ringWrap: document.getElementById("ringWrap"),
  ringProgress: document.getElementById("ringProgress"),
  ringPulse: document.getElementById("ringPulse"),
  bpm: document.getElementById("bpm"),
  nextBeat: document.getElementById("nextBeat"),
  steps: document.getElementById("steps"),
  meter: document.getElementById("meter"),
  beatInBar: document.getElementById("beatInBar"),
  activation: document.getElementById("activation"),
  cpu: document.getElementById("cpu"),
  downbeat: document.getElementById("downbeat"),
  confidenceLabel: document.getElementById("confidenceLabel"),
  confidenceBar: document.getElementById("confidenceBar"),
  startBtn: document.getElementById("startBtn"),
  tabBtn: document.getElementById("tabBtn"),
  nudgeBtn: document.getElementById("nudgeBtn"),
  resetBtn: document.getElementById("resetBtn"),
  deviceSelect: document.getElementById("deviceSelect"),
  meterSelect: document.getElementById("meterSelect"),
  dancer: document.getElementById("dancer"),
};

const grid = new BeatGrid(4);
const meterTally = new MeterTally(grid.meter);
let accent = null;
let tracker = null;
let audioCtx = null;
let lastBeatAt = null;   // performance.now() of the last beat, for the countdown
let barStartAt = null;   // performance.now() of the last "1", for the ring
let cpuAvg = 0;
let wasSilent = null;

// --- downbeats: bidirectional model in a worker, over a rolling window
const WINDOW_HOPS = 600;   // 6 s of context
const EVERY_HOPS = 600;    // one window every 6 s (~23% of a core)
let dbWorker = null;
let dbReady = false;
let hopCounter = 0;
let audioRing = null;      // circular buffer of WINDOW_HOPS * hopSize samples
let ringPos = 0;
let hopSize = 441;
let beatLog = [];          // [{ index, frame }] for the phase vote

function setStatus(text, cls) {
  els.statusText.textContent = text;
  els.status.className = `status ${cls ?? ""}`;
}

function renderSteps(active, intensity = 1, predicted = false) {
  const meter = grid.meter;
  if (els.steps.childElementCount !== meter) {
    els.steps.style.gridTemplateColumns = `repeat(${meter}, minmax(0, 1fr))`;
    els.steps.replaceChildren(
      ...Array.from({ length: meter }, (_, i) => {
        const step = document.createElement("div");
        step.className = "step";
        step.textContent = String(i + 1);
        return step;
      })
    );
  }
  [...els.steps.children].forEach((el, i) => {
    const isActive = i === active - 1;
    el.classList.toggle("active", isActive);
    el.classList.toggle("downbeat", i === 0);
    el.classList.toggle("predicted", isActive && predicted);
    // only the box that lights up carries intensity: proportional to the energy
    // that actually sounded on that hit (1 = strong, near 0 = almost nothing)
    if (isActive) el.style.setProperty("--accent", intensity.toFixed(3));
  });
}

/**
 * The robot moves on two clocks: arms and bounce on every beat, hips and legs
 * once per bar. That is what makes the meter readable — in 4/4 you see four arm
 * swings per hip shift.
 */
let dancerBeat = 0;
let dancerBar = 0;

function danceBeat(inBar, intensity) {
  const d = els.dancer;
  d.classList.remove("idle");
  d.style.setProperty("--hit", intensity.toFixed(3));
  if (grid.period) {
    // the motion tracks the tempo instead of running at a fixed speed
    d.style.setProperty("--beat-ms", `${Math.round(grid.period * 450)}ms`);
  }
  dancerBeat ^= 1;
  d.dataset.beat = String(dancerBeat);
  if (inBar === 1) {
    dancerBar ^= 1;
    d.dataset.bar = String(dancerBar);
  }
  d.classList.toggle("downbeat", inBar === 1);
  // restart the squash even if two beats land close together
  d.classList.remove("hit");
  void d.offsetWidth;
  d.classList.add("hit");
  window.setTimeout(() => d.classList.remove("hit"), 90);
}

function flashBeat(inBar, predicted, lateMs = 0) {
  // a predicted beat is noticed a quarter period late; subtracting it here keeps
  // the phase ring in place even though the flash arrives late
  lastBeatAt = performance.now() - lateMs;
  let intensity = 1;
  if (accent) {
    const hit = accent.onBeat(inBar);
    intensity = hit.intensity;
    els.activation.textContent = `${tracker.activation.toFixed(2)} / e${hit.intensity.toFixed(2)}`;
  }
  if (inBar === 1) barStartAt = lastBeatAt;
  renderSteps(inBar, intensity, predicted);
  danceBeat(inBar, intensity);
  els.beatInBar.textContent = predicted ? `${inBar} ·` : String(inBar);
  // the CSS defines .ring-wrap.hit .ring-pulse and .ring-progress.downbeat
  els.ringProgress.classList.toggle("downbeat", inBar === 1);
  els.ringWrap.classList.remove("hit");
  void els.ringWrap.offsetWidth;
  els.ringWrap.classList.add("hit");
  window.setTimeout(() => els.ringWrap.classList.remove("hit"), 150);
}

function onGridBeats(beats, now) {
  for (const { index, inBar, predicted, time } of beats) {
    // the frame comes from the beat's own time: a filled beat belongs to its
    // hop, not to the hop where we noticed it was missing
    beatLog.push({ index, frame: Math.round(time * 100) });
    if (beatLog.length > 200) beatLog.shift();
    flashBeat(inBar, predicted, (now - time) * 1000);
  }
  if (beats.length === 0) return;
  const conf = grid.confidence;
  els.confidenceLabel.textContent = `${Math.round(conf * 100)}%`;
  els.confidenceBar.style.width = `${conf * 100}%`;
  setStatus(conf > 0.58 ? "locked" : "listening", conf > 0.58 ? "locked" : "");
}

function frame() {
  const silent = tracker !== null && currentStream !== null && tracker.silent;
  if (silent !== wasSilent) {
    wasSilent = silent;
    if (silent) {
      // the last BPM would hang on screen and look like it is still measuring
      setStatus("silence", "");
      els.bpm.textContent = "--.-";
      els.beatInBar.textContent = "-";
      els.ringProgress.style.strokeDashoffset = String(CIRCUMFERENCE);
      renderSteps(0);
      els.dancer.classList.add("idle");
      els.dancer.classList.remove("downbeat");
      // without this, when audio comes back the ring starts with the old tempo
      // before a single new beat has arrived
      lastBeatAt = null;
      barStartAt = null;
    }
  }
  if (silent) {
    els.nextBeat.textContent = "---";
    els.cpu.textContent = cpuAvg.toFixed(2);
    requestAnimationFrame(frame);
    return;
  }
  if (grid.period && lastBeatAt !== null) {
    const now = performance.now();
    // the ring sweeps a whole bar and closes on the 1: at 120 BPM that is one
    // turn every 2 s instead of every 0.5 s. The beat-level pulse is already
    // carried by the flash, the step boxes and the dancer's arms.
    if (barStartAt !== null) {
      const bar = grid.meter * grid.period;
      const phase = Math.min(1, (now - barStartAt) / 1000 / bar);
      els.ringProgress.style.strokeDashoffset = String(CIRCUMFERENCE * (1 - phase));
    }
    els.bpm.textContent = grid.bpm.toFixed(1);
    const sinceBeat = (now - lastBeatAt) / 1000;
    els.nextBeat.textContent = String(Math.max(0, Math.round((grid.period - sinceBeat) * 1000)));
  }
  els.cpu.textContent = cpuAvg.toFixed(2);
  requestAnimationFrame(frame);
}

let workletReady = false;
let currentStream = null;
let sourceNode = null;
let hopNode = null;

async function ensureEngine() {
  if (tracker) return;
  setStatus("loading model", "");
  const [manifest, binBuf] = await Promise.all([
    fetch("public/models/madmom-beats.json").then((r) => r.json()),
    fetch("public/models/madmom-beats.bin").then((r) => r.arrayBuffer()),
  ]);
  const model = loadModel(manifest, binBuf);
  tracker = new BeatTracker(model, manifest, binBuf);
  accent = new AccentProfile(model, grid.meter);

  // The model is tied to 44100 Hz: hops of 441 samples are exactly 10 ms only at
  // that rate. If the context runs at another rate, the whole grid shifts.
  audioCtx = new AudioContext({ sampleRate: manifest.sample_rate });
  if (audioCtx.sampleRate !== manifest.sample_rate) {
    throw new Error(
      `the browser opened audio at ${audioCtx.sampleRate} Hz and the model needs ${manifest.sample_rate}`
    );
  }
  await audioCtx.audioWorklet.addModule("worklet.js");
  workletReady = true;
  els.meter.textContent = METER_LABEL[grid.meter];

  hopSize = manifest.hop_size;
  audioRing = new Float32Array(WINDOW_HOPS * hopSize);
  startDownbeatWorker();
}

function startDownbeatWorker() {
  if (dbWorker) return;
  dbWorker = new Worker("downbeat-worker.js", { type: "module" });
  dbWorker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === "ready") {
      dbReady = true;
      return;
    }
    if (msg.type === "error") {
      console.error("downbeat worker:", msg.message);
      return;
    }
    if (msg.type !== "activations") return;

    const acts = new Float32Array(msg.activations);
    const inWindow = beatLog.filter(
      (x) => x.frame >= msg.startFrame && x.frame < msg.startFrame + msg.numFrames
    );
    // automatic mode uses CANDIDATES (3, 4 and 6); if the meter is pinned by
    // hand, add it so the phase vote knows what to look for
    const cands =
      meterLocked && !CANDIDATES.includes(meterLocked) ? [...CANDIDATES, meterLocked] : CANDIDATES;
    const est = estimateBar(acts, msg.numFrames, msg.startFrame, inWindow, cands);
    if (!est) {
      els.downbeat.textContent = "no evidence";
      return;
    }
    if (meterLocked === null) setMeter(meterTally.push(est.ranking));

    // the phase comes from the candidate matching the meter currently in effect
    const entry = est.ranking.find((r) => r.meter === grid.meter);
    if (!entry || entry.margin < 0.25 || manualNudgeRecent()) return;
    grid.barOrigin = entry.origin;
    els.downbeat.textContent = `pos ${entry.origin + 1} (${Math.round(entry.margin * 100)}%)`;
  };
  dbWorker.postMessage({ type: "init", base: new URL(".", location.href).href });
}

let meterLocked = null;
let lastNudgeAt = -1e9;
function manualNudgeRecent() {
  return performance.now() - lastNudgeAt < 20000;
}

// the notation people actually write for each bar length; nobody writes "6/4"
const METER_LABEL = { 2: "2/4", 3: "3/4", 4: "4/4", 5: "5/4", 6: "6/8" };

function setMeter(next) {
  if (next === grid.meter) return;
  grid.meter = next;
  if (accent) accent.setMeter(next);
  els.meter.textContent = METER_LABEL[next] ?? `${next} beats`;
  renderSteps(1, 1);
}

function pushAudioToRing(hop) {
  audioRing.set(hop, ringPos);
  ringPos = (ringPos + hopSize) % audioRing.length;
  hopCounter++;

  if (!dbReady || hopCounter < WINDOW_HOPS || hopCounter % EVERY_HOPS !== 0) return;
  // 2.7 s of worker time on silence, plus noise fed into the meter vote
  if (tracker.silent) return;
  // unroll the circular buffer into time order
  const flat = new Float32Array(audioRing.length);
  flat.set(audioRing.subarray(ringPos), 0);
  flat.set(audioRing.subarray(0, ringPos), audioRing.length - ringPos);
  dbWorker.postMessage(
    { type: "window", audio: flat.buffer, startFrame: hopCounter - WINDOW_HOPS },
    [flat.buffer]
  );
}

/** Hooks a MediaStream into the tracker, replacing the previous one if any. */
function attach(stream, label) {
  if (sourceNode) sourceNode.disconnect();
  if (hopNode) hopNode.disconnect();
  if (currentStream) for (const t of currentStream.getTracks()) t.stop();

  currentStream = stream;
  resetState();

  sourceNode = audioCtx.createMediaStreamSource(stream);
  hopNode = new AudioWorkletNode(audioCtx, "hop-collector");
  hopNode.port.onmessage = (ev) => {
    const t0 = performance.now();
    const hop = ev.data;
    const beat = tracker.pushHop(hop);
    // the accent looks back into this buffer whenever a beat lands
    accent.pushFrame(tracker.features.specLinear);
    const now = hopCounter / 100;
    onGridBeats(grid.step(now, beat !== null, tracker.silent), now);
    pushAudioToRing(hop);
    cpuAvg = 0.95 * cpuAvg + 0.05 * (performance.now() - t0);
  };
  sourceNode.connect(hopNode);
  // the worklet emits no audio, but connecting it to the destination keeps it running
  hopNode.connect(audioCtx.destination);
  setStatus(`listening: ${label}`, "");
}

/** Fills the selector. PipeWire/PulseAudio monitors show up here. */
async function refreshDevices(selectedId) {
  const devices = (await navigator.mediaDevices.enumerateDevices())
    .filter((d) => d.kind === "audioinput");
  els.deviceSelect.innerHTML = "";
  for (const d of devices) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    // on Linux the monitors are named "Monitor of <sink>"
    const isMonitor = /monitor/i.test(d.label);
    opt.textContent = (isMonitor ? "🔊 " : "🎤 ") + (d.label || d.deviceId);
    els.deviceSelect.appendChild(opt);
  }
  els.deviceSelect.disabled = devices.length === 0;
  if (selectedId) els.deviceSelect.value = selectedId;
  const monitors = devices.filter((d) => /monitor/i.test(d.label)).length;
  if (monitors === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.disabled = true;
    opt.textContent = "— no monitors found; try \"Tab audio\" —";
    els.deviceSelect.appendChild(opt);
  }
  return devices;
}

async function useInput(deviceId) {
  await ensureEngine();
  setStatus("asking for audio permission", "");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  const label = stream.getAudioTracks()[0]?.label ?? "input";
  await refreshDevices(stream.getAudioTracks()[0]?.getSettings().deviceId);
  attach(stream, label);
}

async function useTabAudio() {
  await ensureEngine();
  setStatus("pick a tab and tick \"share audio\"", "");
  // Chrome requires asking for video even when only audio is wanted.
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  if (stream.getAudioTracks().length === 0) {
    for (const t of stream.getTracks()) t.stop();
    throw new Error("that tab was shared without audio: try again and tick \"share audio\"");
  }
  for (const t of stream.getVideoTracks()) t.stop();  // the video is unused
  attach(stream, "shared tab");
}

// The browser's own errors here are famously unhelpful ("Permission denied by
// user" is the same message whether you clicked Block, dismissed the dialog, or
// blocked the site months ago), so they get translated into what to actually do.
function explain(err) {
  switch (err.name) {
    case "NotAllowedError":
      return "permission denied — click the icon on the left of the address bar, allow the microphone, and reload (cancelling the tab picker gives this too)";
    case "NotFoundError":
      return "no audio input found: check that some capture device exists";
    case "NotReadableError":
      return "the device is busy: another app is holding the microphone";
    case "OverconstrainedError":
      return "that device is no longer available; pick another source";
    default:
      return err.message ?? "error";
  }
}

const guard = (fn) => () => {
  fn().catch((err) => {
    console.error(err);
    setStatus(explain(err), "error");
  });
};

els.startBtn.addEventListener("click", guard(() => useInput(els.deviceSelect.value || null)));
els.tabBtn.addEventListener("click", guard(useTabAudio));
els.deviceSelect.addEventListener("change", guard(() => useInput(els.deviceSelect.value)));

els.meterSelect.addEventListener("change", () => {
  const v = els.meterSelect.value;
  meterLocked = v ? Number(v) : null;
  if (meterLocked !== null) setMeter(meterLocked);
  meterTally.reset(grid.meter);
  els.downbeat.textContent = "--";
});

els.nudgeBtn.addEventListener("click", () => {
  grid.barOrigin = grid.index;
  lastNudgeAt = performance.now();
  els.downbeat.textContent = "manual";
});

function resetState() {
  if (tracker) tracker.reset();
  if (accent) accent.reset();
  grid.reset();
  grid.barOrigin = 0;
  lastBeatAt = null;
  beatLog = [];
  hopCounter = 0;
  ringPos = 0;
  wasSilent = null;
  meterTally.reset(grid.meter);
  if (audioRing) audioRing.fill(0);
  els.downbeat.textContent = "--";
  els.confidenceBar.style.width = "0%";
  els.confidenceLabel.textContent = "0%";
  els.bpm.textContent = "--.-";
  els.dancer.classList.add("idle");
}

els.resetBtn.addEventListener("click", resetState);

els.ringProgress.style.strokeDasharray = String(CIRCUMFERENCE);
els.dancer.classList.add("idle");
renderSteps(1);
requestAnimationFrame(frame);
