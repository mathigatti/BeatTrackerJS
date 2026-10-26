// Coreografía del robot, compartida por baile.js (con archivo) y vivo.js (en vivo).
//
// - Cadencia: el vaivén es un péndulo que nunca se corta; la fase se corrige de a poco (±20% de velocidad
//   como mucho) para llegar al extremo en el ancla. El paso cambia poco (ver STEP_CONFIRM / STEP_MIN_BARS) y
//   la intensidad y la forma cambian con fundido.
// - Movimiento principal: ir de un lado al otro, una vez por compás como mucho. El cuerpo viaja a lo largo
//   del compás y llega al extremo justo en la nota "ancla": el grave confirmado (sonó en los dos compases
//   anteriores) que más se repitió en los últimos REPEAT_BARS compases; si no hay, el 1 del compás.
// - Textura: las demás notas bien establecidas (sonaron en TEX_MIN_REPEAT de los últimos REPEAT_BARS
//   compases; como mucho TEX_MAX por registro) no deciden el viaje, le dan forma. Pocas: viaje fluido y
//   directo. Algunas: ondulaciones redondeadas al ritmo de esas notas. Muchas: cortas y puntiagudas. La
//   reacción grande es de la cabeza (cabecea con los graves, se inclina con los agudos); brazos y piernas
//   acompañan apenas.
// Todo lo que decide el compás n sale de compases anteriores, como en vivo.

const REPEAT_BARS = 4;
const TEX_MIN_REPEAT = 3; // una nota de textura tiene que haber sonado en 3 de los últimos 4 compases
const TEX_MAX = 4;        // y como mucho 4 por registro (las más repetidas)
const POINTY_FROM = 6;    // notas de textura por compás desde las que el movimiento es puntiagudo
const WAVY_FROM = 2;      // desde cuántas hay ondulación (menos: fluido y directo)

const stepsOf = (pos, n) => new Set(pos.map((p) => Math.round(p * n)).filter((s) => s < n));
const inter = (a, b) => new Set([...a].filter((x) => b.has(x)));

/** Plan del próximo compás a partir de los compases ya escuchados ({beats, graves: [pos], agudos: [pos]}). */
export function planBar(history, beats, dist = []) {
  const n = beats * 4;
  const same = history.filter((b) => b.beats === beats);
  const last = same[same.length - 1], prev = same[same.length - 2];
  const sg = (b) => (b ? stepsOf(b.graves, n) : new Set()), sa = (b) => (b ? stepsOf(b.agudos, n) : new Set());
  const confG = last && prev ? inter(sg(last), sg(prev)) : new Set();
  const confA = last && prev ? inter(sa(last), sa(prev)) : new Set();
  // ancla: el grave confirmado más repetido; a igualdad, el que cae en un tiempo y el más temprano
  const counter = (f) => {
    const c = new Map();
    for (const b of same.slice(-REPEAT_BARS)) for (const s of f(b)) c.set(s, (c.get(s) || 0) + 1);
    return c;
  };
  const count = counter(sg), countA = counter(sa);
  let anchor = 0, by = 'el 1 del compás (sin patrón seguro)';
  if (confG.size) {
    anchor = [...confG].sort((a, b) => (count.get(b) - count.get(a)) || ((a % 4) - (b % 4)) || (a - b))[0];
    by = `grave más repetido (${count.get(anchor)} de ${Math.min(REPEAT_BARS, same.length)} compases)`;
  }
  const pickTex = (conf, c, skip) => [...conf].filter((s) => s !== skip && c.get(s) >= TEX_MIN_REPEAT)
    .sort((a, b) => (c.get(b) - c.get(a)) || ((a % 4) - (b % 4)) || (a - b)).slice(0, TEX_MAX);
  const texG = pickTex(confG, count, anchor), texA = pickTex(confA, countA, null);
  const tex = texG.length + texA.length;
  const shape = tex >= POINTY_FROM ? 'puntiagudo' : tex >= WAVY_FROM ? 'ondulado' : 'directo';
  const dens = last ? last.graves.length + last.agudos.length : 0;
  const sorted = [...dist].sort((a, b) => a - b), q = (f) => sorted[Math.floor(f * (sorted.length - 1))];
  const typical = Math.max(1, q(0.5) ?? 8);
  // tipo de momento, con los últimos 2 compases: pocos graves = transición; densidad entre el 20% más alto de lo
  // escuchado (y bien por encima de lo típico) = pico; si no, base
  const recent = history.slice(-2);
  const avg = (f) => recent.reduce((a, b) => a + f(b), 0) / Math.max(1, recent.length);
  const graves = avg((b) => b.graves.length), total = avg((b) => b.graves.length + b.agudos.length);
  let kind = 'base';
  if (!recent.length || graves <= 1) kind = 'transicion';
  else if (sorted.length >= 16 && total >= q(0.8) && total >= 1.25 * typical && graves >= 2) kind = 'pico';
  return {
    n, anchor, by, texG, texA, shape, kind,
    amp: Math.min(1.4, Math.max(0.5, 0.5 + 0.5 * dens / typical)),
    label: `${shape} (${tex} notas de textura)`,
  };
}

// Pasos: "transicion" (pocos graves: vaivén lento, cada 2 compases, brazos flotando), "pico" (máxima intensidad:
// brazos arriba y rebote en cada tiempo), y dos pasos de base que se alternan cada BASE_SWAP_BARS compases:
// "lateral" (paso de lado clásico) y "rebote" (rebota en los tiempos con los brazos marcando, cadera en twist).
// Todos comparten el mismo péndulo, así que pasar de uno a otro es un fundido sin cortar la cadencia.
// El paso cambia poco: una propuesta nueva tiene que repetirse STEP_CONFIRM compases seguidos y pasar al menos
// KIND_MIN_BARS compases desde el último cambio de paso (STEP_MIN_BARS si solo cambia ancla o forma).
const STEP_CONFIRM = 2;
const KIND_MIN_BARS = 4;
const STEP_MIN_BARS = 8;
const BASE_SWAP_BARS = 16;
const PLL_GAIN = 0.15;   // corrección de fase por compás (fracción del error)
const PLL_MAX = 0.2;     // como mucho ±20% de velocidad: se ajusta, nunca frena ni salta
const SMOOTH_S = 1.0;    // constante de tiempo para la intensidad y el fundido entre formas
const KIND_SMOOTH_S = 1.5;
const STEPS = ['lateral', 'rebote', 'transicion', 'pico'];
export const STEP_TEXT = { lateral: 'paso lateral', rebote: 'rebote', transicion: 'transición (pocos graves)', pico: 'pico de intensidad' };

/** Mueve el robot (variables CSS de estilo "flujo") según el plan y el tiempo. */
export class Choreo {
  constructor(el, onStep = null) {
    this.el = el;
    this.onStep = onStep;  // avisa cuando cambia el paso, para mostrarlo
    el.dataset.style = 'flujo';
    this.reset();
  }

  reset() {
    this.phase = null;   // fase del vaivén: extremos en múltiplos de π, medio compás de ida = π
    this.lastT = null;
    this.step = null;    // {kind, step, n, anchor, by, shape, label}
    this.cand = null; this.candN = 0; this.since = 0; this.baseBars = 0; this.nextBase = 'lateral';
    this.lastPlan = null;
    this.amp = null;
    this.mix = { ondulado: 0, puntiagudo: 0 };
    this.k = Object.fromEntries(STEPS.map((s) => [s, 0]));
  }

  newBar(plan) {
    this.since++;
    const prop = { kind: plan.kind, n: plan.n, anchor: plan.anchor, by: plan.by, shape: plan.shape, label: plan.label };
    const same = (a, b) => a && a.kind === b.kind && a.n === b.n && a.anchor === b.anchor && a.shape === b.shape;
    if (!this.step || this.step.n !== plan.n) { this.setStep(prop); return; }
    if (this.step.kind === 'base' && ++this.baseBars >= BASE_SWAP_BARS && this.since >= STEP_MIN_BARS) {
      this.setStep({ ...this.step, step: this.step.step === 'lateral' ? 'rebote' : 'lateral' });
      return;
    }
    if (same(this.step, prop)) { this.cand = null; this.candN = 0; return; }
    if (same(this.cand, prop)) this.candN++; else { this.cand = prop; this.candN = 1; }
    const minBars = prop.kind !== this.step.kind ? KIND_MIN_BARS : STEP_MIN_BARS;
    if (this.candN >= STEP_CONFIRM && this.since >= minBars) this.setStep(prop);
  }

  setStep(prop) {
    if (!prop.step) {
      if (prop.kind !== 'base') prop.step = prop.kind;
      else if (this.step?.kind === 'base') prop.step = this.step.step;
      else { prop.step = this.nextBase; this.nextBase = prop.step === 'lateral' ? 'rebote' : 'lateral'; }
    }
    if (prop.step !== this.step?.step) this.baseBars = 0;
    this.step = prop; this.cand = null; this.candN = 0; this.since = 0;
    if (this.onStep) this.onStep({ ...prop, text: STEP_TEXT[prop.step] });
  }

  /** t, barStart, barDur en segundos; raise 0-1 (subida). */
  update(plan, t, barStart, barDur, raise = 0) {
    if (!plan) return;
    if (plan !== this.lastPlan) { this.lastPlan = plan; this.newBar(plan); }
    const { n, anchor, shape, step } = this.step;
    // fase deseada: extremo justo en el ancla de cada compás (de un lado al otro una vez por compás)
    const want = Math.PI * ((t - barStart) / barDur - anchor / n);
    // el reloj puede retroceder unos ms (currentTime del audio, hops que llegan en ráfagas): eso no reinicia,
    // solo un salto grande (seek o pausa larga)
    let dt = this.lastT === null ? 0 : t - this.lastT;
    if (this.phase === null || dt < -0.5 || dt > 1) this.lastT = t;
    else if (dt < 0) dt = 0;
    else this.lastT = t;
    if (this.phase === null || dt < -0.5 || dt > 1) {
      this.phase = want;  // arranque o salto en el audio (seek)
      this.amp = plan.amp;
      for (const s of STEPS) this.k[s] = s === step ? 1 : 0;
    } else {
      // como un péndulo empujado: avanza siempre a la velocidad del compás y corrige la fase de a poco
      let err = (want - this.phase) % Math.PI;
      if (err > Math.PI / 2) err -= Math.PI; else if (err < -Math.PI / 2) err += Math.PI;
      const adj = Math.max(-PLL_MAX, Math.min(PLL_MAX, PLL_GAIN * err));
      this.phase += (Math.PI / barDur) * dt * (1 + adj);
      const k = 1 - Math.exp(-dt / SMOOTH_S), kk = 1 - Math.exp(-dt / KIND_SMOOTH_S);
      this.amp += (plan.amp - this.amp) * k;
      for (const s of ['ondulado', 'puntiagudo']) this.mix[s] += ((shape === s ? 1 : 0) - this.mix[s]) * k;
      for (const s of STEPS) this.k[s] += ((s === step ? 1 : 0) - this.k[s]) * kk;
    }
    const beats = n / 4;
    const sway = -Math.cos(this.phase);
    const sway2 = -Math.cos(this.phase / 2);                // un lado cada 2 compases
    const bob = Math.abs(Math.sin(this.phase * beats));     // 0 en cada tiempo (abajo), 1 entre tiempos
    const wob = (steps) => {
      let w = 0, p = 0;
      for (const back of [0, 1]) {
        for (const s of steps) {
          const d = t - (barStart - back * barDur + (s / plan.n) * barDur);
          if (d < 0) continue;
          p += Math.exp(-d / 0.05);
          if (d < barDur / 8) w += Math.sin((Math.PI * d) / (barDur / 8));
        }
      }
      return Math.min(1, this.mix.ondulado * w + this.mix.puntiagudo * p);
    };
    const st = this.el.style;
    st.setProperty('--sway', sway.toFixed(4));
    st.setProperty('--sway2', sway2.toFixed(4));
    st.setProperty('--bob', bob.toFixed(4));
    st.setProperty('--wobg', wob(plan.texG).toFixed(4));
    st.setProperty('--woba', wob(plan.texA).toFixed(4));
    st.setProperty('--amp', this.amp.toFixed(3));
    st.setProperty('--raise', raise.toFixed(3));
    st.setProperty('--kL', this.k.lateral.toFixed(3));
    st.setProperty('--kR', this.k.rebote.toFixed(3));
    st.setProperty('--kT', this.k.transicion.toFixed(3));
    st.setProperty('--kP', this.k.pico.toFixed(3));
  }
}
