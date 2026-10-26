// Demo de avances del robot que baila. Clips: ataques precalculados (offline y tiempo real, exportados por
// dataset/tools/export_demo.py) con su audio. En vivo: lo mismo que dataset/.../vivo.html, con el tracker de
// beats de BeatTrackerJS y el detector de ataques graves/agudos (rt/onsets.js).
import { loadModel } from "/src/model.js";
import { BeatTracker } from "/src/tracker.js";
import { BeatGrid } from "./grid.js";
import { OnsetDetector } from "./rt/onsets.js";
import { Choreo, planBar } from "./choreo.js";

const $ = (s) => document.querySelector(s);
const here = (p) => new URL(p, import.meta.url).href;
const dancer = $('#dancer');
const roll = $('#roll');
const LEAD_S = 0.06;
const COL = { g: '#ffb23f', a: '#2dd9e5' };

// círculo de carga arriba del robot, mientras carga o escucha y todavía no baila
function setLoading(text) {
  $('#loading').classList.toggle('on', !!text);
  if (text) $('#loading-txt').textContent = text;
}

const choreo = new Choreo(dancer, (st) => {
  $('#r-step').textContent = st.text;
  $('#r-guide').textContent = st.by;
});

const DEFAULT_CLIP = 'the-white-stripes_seven-nation-army';
const S = { source: null, clip: null, mode: 'off', bars: [], plans: [], cur: -1 };
const audio = new Audio();
audio.preload = 'auto';

const stepsOf = (pos, n) => new Set(pos.map((p) => Math.round(p * n)).filter((s) => s < n));
const inter = (a, b) => new Set([...a].filter((x) => b.has(x)));

// ---------------------------------------------------------------- fuentes

async function init() {
  const clips = await fetch(here('clips/index.json')).then((r) => r.json());
  const nav = $('#sources');
  clips.sort((a, b) => (b.id === DEFAULT_CLIP) - (a.id === DEFAULT_CLIP));
  for (const c of clips) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = `${c.title} · ${c.artist.split(' (')[0]}`;
    b.dataset.id = c.id;
    b.onclick = () => selectClip(c.id, true);
    nav.append(b);
  }
  const live = document.createElement('button');
  live.type = 'button';
  live.className = 'live';
  live.textContent = '🎤 Probalo en vivo';
  live.dataset.id = 'live';
  live.onclick = selectLive;
  nav.append(live);
  selectClip(DEFAULT_CLIP, false);
  loadModels().catch((e) => console.error(e));
}

function markSource(id) {
  for (const b of $('#sources').children) b.classList.toggle('on', b.dataset.id === id);
  const isLive = id === 'live';
  $('#clip-controls').hidden = isLive;
  $('#live-controls').hidden = !isLive;
  $('#legend').toggleAttribute('data-live', isLive);
}

async function selectClip(id, autoplay) {
  stopLive();
  S.source = 'clip';
  markSource(id);
  audio.pause();
  const clip = await fetch(here(`clips/${id}.json`)).then((r) => r.json());
  S.clip = clip;
  audio.src = here(`clips/${id}.mp3`);
  $('#bpm').textContent = `${Math.round(clip.bpm)} bpm`;
  buildPlans();
  choreo.reset();
  S.cur = -1;
  dancer.classList.add('idle');
  setHint();
  renderPattern(null);
  if (autoplay) audio.play().catch(() => {});
}

function buildPlans() {
  const key = S.mode;
  S.bars = S.clip.bars.map((b) => ({ start: b.start, end: b.end, beats: b.beats, graves: b[key].graves, agudos: b[key].agudos }));
  S.plans = S.bars.map((b, i) => planBar(S.bars.slice(Math.max(0, i - 4), i), b.beats,
    S.bars.slice(Math.max(0, i - 64), i).map((x) => x.graves.length + x.agudos.length)));
}

function setHint() {
  if (S.source === 'live') {
    $('#hint').textContent = 'Poné música en otra pestaña (tildá "compartir audio") o cerca del micrófono. Tarda unos segundos en engancharse.';
  } else {
    const c = S.clip;
    const from = c.from_s ? ` desde ${Math.floor(c.from_s / 60)}:${String(Math.round(c.from_s % 60)).padStart(2, '0')}` : '';
    $('#hint').textContent = `${c.artist} — ${c.title}${from}. ` + (S.mode === 'rt'
      ? 'Baila con lo que detecta en tiempo real.'
      : 'Baila con la referencia offline: usa stems separados y mira el tema entero.');
  }
}

for (const b of document.querySelectorAll('.seg button')) {
  b.onclick = () => {
    S.mode = b.dataset.mode;
    for (const x of document.querySelectorAll('.seg button')) x.classList.toggle('on', x === b);
    if (S.clip) { buildPlans(); S.cur = -1; setHint(); }
  };
}

audio.onplay = () => { $('#play').textContent = '⏸'; dancer.classList.remove('idle'); };
audio.onpause = () => { $('#play').textContent = '▶'; dancer.classList.add('idle'); };
audio.onended = () => { audio.currentTime = 0; };
$('#play').onclick = () => (audio.paused ? audio.play() : audio.pause());
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && S.source === 'clip' && e.target.tagName !== 'BUTTON') { e.preventDefault(); $('#play').click(); }
});

// ---------------------------------------------------------------- patrón (16 semicorcheas, graves y agudos)

function renderPattern(view) {
  const svg = $('#pattern');
  const W = svg.clientWidth || 600, H = svg.clientHeight || 76, padL = 54, padR = 4;
  const n = view?.n || 16, cw = (W - padL - padR) / n, rh = (H - 6) / 2;
  let html = '';
  [['agudos', 'a'], ['graves', 'g']].forEach(([reg, k], r) => {
    const y = 3 + r * rh, conf = view ? view[reg] : new Set();
    html += `<text class="row-lbl" x="0" y="${y + rh / 2 + 4}">${reg}</text>`;
    for (let s = 0; s < n; s++) {
      const x = padL + s * cw + (s % 4 === 0 ? 3 : 1.5);
      html += `<rect class="${conf.has(s) ? `cell-${k}` : 'cell-off'}" x="${x}" y="${y + 3}" width="${cw - 4.5}" height="${rh - 6}" rx="4"/>`;
      if (view && reg === 'graves' && view.anchorOn && s === view.anchor) {
        html += `<rect class="anchor" x="${x - 1.5}" y="${y + 1.5}" width="${cw - 1.5}" height="${rh - 3}" rx="5"/>`;
      }
    }
  });
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = html;
}

function patternView(history, beats, plan) {
  const n = beats * 4, same = history.filter((b) => b.beats === beats).slice(-2);
  const conf = (reg) => (same.length === 2 ? inter(stepsOf(same[0][reg], n), stepsOf(same[1][reg], n)) : new Set());
  const graves = conf('graves');
  return { n, graves, agudos: conf('agudos'), anchor: plan?.anchor ?? 0, anchorOn: graves.size > 0 };
}

// ---------------------------------------------------------------- roll: ataques en el tiempo

function sizeCanvas() {
  const dpr = window.devicePixelRatio || 1, r = roll.getBoundingClientRect();
  if (roll.width !== Math.round(r.width * dpr) || roll.height !== Math.round(r.height * dpr)) {
    roll.width = Math.round(r.width * dpr);
    roll.height = Math.round(r.height * dpr);
  }
  return { ctx: roll.getContext('2d'), dpr, W: r.width, H: r.height };
}

/** rows: [{label, reg, hollow, hits: [t]}]; lines: [{t, strong}]; ventana [t0, t1], cursor en tNow. */
function drawRoll(rows, lines, t0, t1, tNow) {
  const { ctx, dpr, W, H } = sizeCanvas();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const padL = 92, x = (t) => padL + ((t - t0) / (t1 - t0)) * (W - padL - 8);
  for (const l of lines) {
    if (l.t < t0 || l.t > t1) continue;
    ctx.strokeStyle = l.strong ? 'rgba(143,163,167,0.7)' : 'rgba(143,163,167,0.18)';
    ctx.lineWidth = l.strong ? 1.5 : 1;
    ctx.beginPath(); ctx.moveTo(x(l.t), 4); ctx.lineTo(x(l.t), H - 4); ctx.stroke();
  }
  const rh = (H - 8) / rows.length;
  ctx.font = '12px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  rows.forEach((row, i) => {
    const y = 4 + rh * (i + 0.5);
    ctx.fillStyle = '#8fa3a7';
    ctx.fillText(row.label, 6, y);
    for (const t of row.hits) {
      if (t < t0 || t > t1) continue;
      const past = t <= tNow, age = tNow - t;
      const r = past && age < 0.12 ? 7.5 : 5.5;
      ctx.globalAlpha = past ? 1 : 0.3;
      ctx.beginPath(); ctx.arc(x(t), y, r, 0, Math.PI * 2);
      if (row.hollow) { ctx.strokeStyle = COL[row.reg]; ctx.lineWidth = 2; ctx.stroke(); }
      else { ctx.fillStyle = COL[row.reg]; ctx.fill(); }
    }
    ctx.globalAlpha = 1;
  });
  ctx.strokeStyle = '#e9f0ee';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x(tNow), 2); ctx.lineTo(x(tNow), H - 2); ctx.stroke();
}

function clipRows() {
  const abs = (key, reg) => S.clip.bars.flatMap((b) => b[key][reg].map((p) => b.start + p * (b.end - b.start)));
  if (!S.clip.rows) {
    S.clip.rows = [
      { label: 'agudos offline', reg: 'a', hollow: true, hits: abs('off', 'agudos') },
      { label: 'agudos en vivo', reg: 'a', hollow: false, hits: abs('rt', 'agudos') },
      { label: 'graves offline', reg: 'g', hollow: true, hits: abs('off', 'graves') },
      { label: 'graves en vivo', reg: 'g', hollow: false, hits: abs('rt', 'graves') },
    ];
    S.clip.lines = S.clip.bars.flatMap((b) => Array.from({ length: b.beats }, (_, k) =>
      ({ t: b.start + (k / b.beats) * (b.end - b.start), strong: k === 0 })));
  }
  return S.clip;
}

// ---------------------------------------------------------------- clip: animación

function barAt(t) {
  let lo = 0, hi = S.bars.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (S.bars[mid].start <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

function clipFrame() {
  if (!S.bars.length) return;
  const t = audio.currentTime, i = barAt(t + LEAD_S), b = S.bars[i];
  const span = b.end - b.start;
  const playing = !audio.paused && t + LEAD_S >= S.bars[0].start;
  if (playing) {
    choreo.update(S.plans[i], t, b.start, span);
    dancer.classList.remove('idle');
    dancer.classList.toggle('downbeat', (t - b.start) / span < 1 / b.beats);
  } else if (!audio.paused) dancer.classList.add('idle');
  if (i !== S.cur) {
    S.cur = i;
    renderPattern(t + LEAD_S >= S.bars[0].start ? patternView(S.bars.slice(Math.max(0, i - 2), i), b.beats, S.plans[i]) : null);
  }
  setLoading(audio.paused ? null : audio.readyState < 3 ? 'cargando…' : !playing ? 'escuchando…' : null);
  const c = clipRows(), win = span * 2.2;
  drawRoll(c.rows, c.lines, t - win * 0.7, t + win * 0.3, t);
}

// ---------------------------------------------------------------- en vivo (como vivo.js)

const DANCE_BEATS = 4;
const FINALIZE_AFTER_S = 0.2;
const HALF_ABOVE = 145, HALF_BELOW = 125;
const SILENCE_RESET_S = 1;
const IDLE_AFTER_S = 2;
const L = { tracker: null, detector: null, ctx: null, stream: null, src: null, node: null, grid: new BeatGrid(DANCE_BEATS) };
let hops, lastHopPerf, lastBeat, barStart, pending, plan, factor, silentFor, lastDanceBpm, clock, notLiveSince;
const onsets = [], bars = [], densities = [], beatLines = [];

function resetLive() {
  hops = 0; lastHopPerf = 0; lastBeat = null; barStart = null; pending = []; plan = null;
  factor = 1; silentFor = 0; lastDanceBpm = 0; clock = 0; notLiveSince = null;
  onsets.length = 0; bars.length = 0; densities.length = 0; beatLines.length = 0;
}
resetLive();

function forgetBars() {
  barStart = null; pending = []; bars.length = 0; densities.length = 0; plan = null;
}

function planNext() {
  plan = planBar(bars.slice(-4), DANCE_BEATS, densities);
  renderPattern(patternView(bars.slice(-2), DANCE_BEATS, plan));
}

function closeBar(start, end) {
  const span = end - start;
  const take = (reg) => onsets.filter((o) => o.reg === reg && o.time >= start - 0.04 && o.time < end - 0.04)
    .map((o) => Math.min(0.9999, Math.max(0, (o.time - start) / span)));
  bars.push({ start, end, beats: DANCE_BEATS, graves: take('graves'), agudos: take('agudos') });
  if (bars.length > 64) bars.shift();
  const b = bars[bars.length - 1];
  densities.push(b.graves.length + b.agudos.length);
  if (densities.length > 64) densities.shift();
  planNext();
}

function onHop(hop) {
  const { tracker, detector, grid } = L;
  const beat = tracker.pushHop(hop);
  const now = hops / 100;
  hops++;
  lastHopPerf = performance.now();
  if (tracker.silent) silentFor += 0.01;
  else {
    if (silentFor >= SILENCE_RESET_S) { tracker.dbn.reset(); grid.reset(); forgetBars(); }
    silentFor = 0;
  }
  for (const o of detector.pushHop(hop)) onsets.push(o);
  while (onsets.length && onsets[0].time < now - 30) onsets.shift();
  // el tracker a veces engancha el doble: arriba de HALF_ABOVE se baila a la mitad
  const f = grid.bpm > HALF_ABOVE ? 2 : grid.bpm && grid.bpm < HALF_BELOW ? 1 : factor;
  if (f !== factor || grid.songChanged) {
    const danceBpm = grid.bpm / f;
    if (!lastDanceBpm || Math.abs(danceBpm - lastDanceBpm) / lastDanceBpm > 0.15) forgetBars();
    factor = f;
    grid.meter = DANCE_BEATS * factor;
  }
  if (grid.period) lastDanceBpm = grid.bpm / factor;
  for (const b of grid.step(now, beat !== null, tracker.silent)) {
    // el compás arranca en el primer beat, sin esperar a que el conteo pase por un "1" (igual es arbitrario)
    if (barStart === null) grid.barOrigin = b.index;
    const inBar = grid.inBar(b.index);
    lastBeat = { time: b.time, inBar };
    beatLines.push({ t: b.time, strong: inBar === 1 });
    if (beatLines.length > 64) beatLines.shift();
    if (inBar === 1) {
      if (barStart !== null) pending.push([barStart, b.time]);
      barStart = b.time;
    }
  }
  while (pending.length && now >= pending[0][1] + FINALIZE_AFTER_S) closeBar(...pending.shift());
  $('#bpm-live').textContent = grid.period ? `${Math.round(grid.bpm / factor)} bpm` : '– bpm';
  $('#status').textContent = tracker.silent ? 'silencio' : grid.confidence > 0.58 ? 'enganchado' : 'escuchando…';
}

// los modelos se bajan apenas carga la página, así "en vivo" arranca sin esperar
let models = null;
let liveWanted = false;  // se pidió audio y todavía no se cortó
const loadModels = () => (models ??= Promise.all([
  fetch('/public/models/madmom-beats.json').then((r) => r.json()),
  fetch('/public/models/madmom-beats.bin').then((r) => r.arrayBuffer()),
  OnsetDetector.load(here('rt/')),
]));

async function startLive(getStream) {
  const s = await getStream();
  liveWanted = true;
  if (!L.tracker) {
    $('#status').textContent = 'cargando modelos…';
    const [manifest, bin, detector] = await loadModels();
    L.tracker = new BeatTracker(loadModel(manifest, bin), manifest, bin);
    L.detector = detector;
    L.ctx = new AudioContext({ sampleRate: 44100 });
    await L.ctx.audioWorklet.addModule('/worklet.js');
  }
  stopLive();
  liveWanted = true;
  L.stream = s;
  L.tracker.reset(); L.detector.reset(); L.grid.reset(); L.grid.meter = DANCE_BEATS;
  resetLive();
  choreo.reset();
  L.src = L.ctx.createMediaStreamSource(s);
  L.node = new AudioWorkletNode(L.ctx, 'hop-collector');
  L.node.port.onmessage = (ev) => onHop(ev.data);
  L.src.connect(L.node);
  L.node.connect(L.ctx.destination);
  await L.ctx.resume();
  $('#status').textContent = 'escuchando…';
}

function stopLive() {
  liveWanted = false;
  if (L.src) L.src.disconnect();
  if (L.node) { L.node.port.onmessage = null; L.node.disconnect(); }
  if (L.stream) for (const t of L.stream.getTracks()) t.stop();
  L.src = L.node = L.stream = null;
}

function selectLive() {
  audio.pause();
  setLoading(null);
  S.source = 'live';
  markSource('live');
  choreo.reset();
  dancer.classList.add('idle');
  setHint();
  renderPattern(null);
  $('#r-step').textContent = '–';
  $('#r-guide').textContent = '–';
}

const NO_PROC = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
const guard = (fn) => () => fn().catch((e) => {
  console.error(e);
  $('#status').textContent = e.name === 'NotAllowedError' ? 'permiso denegado' : `error: ${e.message}`;
});
$('#tab').onclick = guard(() => startLive(async () => {
  const s = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: NO_PROC });
  if (!s.getAudioTracks().length) {
    for (const t of s.getTracks()) t.stop();
    throw new Error('se compartió sin audio: tildá "compartir audio"');
  }
  for (const t of s.getVideoTracks()) t.stop();
  return s;
}));
$('#mic').onclick = guard(() => startLive(() => navigator.mediaDevices.getUserMedia({ audio: NO_PROC })));

function liveFrame() {
  const { tracker, grid } = L;
  const ready = tracker && L.stream && grid.period && lastBeat && barStart !== null;
  const live = ready && !tracker.silent;
  const perf = performance.now();
  if (live) notLiveSince = null; else if (notLiveSince === null) notLiveSince = perf;
  const dancing = ready && (live || perf - notLiveSince < IDLE_AFTER_S * 1000);
  dancer.classList.toggle('idle', !dancing);
  const st = $('#status').textContent;
  setLoading(liveWanted && !dancing ? (st.startsWith('cargando') ? 'cargando modelos…' : 'escuchando…') : null);
  // reloj del audio: los hops llegan en ráfagas, así que se interpola y nunca se deja retroceder
  if (L.stream) clock = Math.max(clock, hops / 100 + Math.min(0.05, (perf - lastHopPerf) / 1000));
  const barDur = grid.period ? grid.meter * grid.period : 2;
  if (dancing) {
    if (!plan) planNext();
    const start = barStart + Math.floor(Math.max(0, clock - barStart) / barDur) * barDur;
    choreo.update(plan, clock, start, barDur);
    dancer.classList.toggle('downbeat', clock - start < grid.period);
  }
  const rows = [
    { label: 'agudos', reg: 'a', hollow: false, hits: onsets.filter((o) => o.reg === 'agudos').map((o) => o.time) },
    { label: 'graves', reg: 'g', hollow: false, hits: onsets.filter((o) => o.reg === 'graves').map((o) => o.time) },
  ];
  const win = barDur * 2.2;
  drawRoll(rows, beatLines, clock - win, clock + 0.05 * win, clock);
}

// ---------------------------------------------------------------- loop

function frame() {
  if (S.source === 'clip') clipFrame(); else if (S.source === 'live') liveFrame();
  requestAnimationFrame(frame);
}
window.addEventListener('resize', () => { S.cur = -1; if (S.source === 'live') renderPattern(plan ? patternView(bars.slice(-2), DANCE_BEATS, plan) : null); });
requestAnimationFrame(frame);
init();
