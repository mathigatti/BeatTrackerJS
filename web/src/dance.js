// Dance moves for the robot.
//
// A move is two independent things, and keeping them apart is what lets the same
// look run at different rhythmic subdivisions:
//
//   style  which CSS transforms apply (the look)
//   arms / hips   how often each part changes, in beats
//
// So "Swing" and "Swing double" share every transform and differ only in that
// the arms toggle every beat instead of every half beat. `"bar"` means once per
// bar, which follows the detected meter: in 4/4 the hips move every 4 beats, in
// 3/4 every 3.
//
// Everything is driven from a continuous beat position rather than from beat
// events, because a subdivision below the beat (eighths) has no event to hang
// off — it falls between two of them.

export const MOVES = [
  { id: "swing", label: "Swing", style: "swing", arms: 1, hips: "bar" },
  { id: "swing2", label: "Swing · double", style: "swing", arms: 0.5, hips: "bar" },
  { id: "swingh", label: "Swing · half", style: "swing", arms: 2, hips: "bar2" },
  { id: "twist", label: "Twist", style: "twist", arms: 1, hips: 0.5 },
  { id: "robot", label: "Robot", style: "robot", arms: 1, hips: "bar" },
  { id: "robot2", label: "Robot · double", style: "robot", arms: 0.5, hips: "bar" },
  { id: "jump", label: "Jump", style: "jump", arms: 2, hips: 1 },
  { id: "sway", label: "Sway", style: "sway", arms: 2, hips: "bar" },
];

const BARS_PER_MOVE = 4;   // how long each move lasts in auto mode

export class Dancer {
  constructor(el) {
    this.el = el;
    this.move = MOVES[0];
    this.auto = false;
    this.intensity = 1;
    this.armTick = 0;
    this.hipTick = 0;
    this.autoAt = 0;         // bar position where the current move started
    this.apply();
  }

  /** id, or null for auto rotation. */
  setMove(id) {
    this.auto = id === null;
    const found = MOVES.find((m) => m.id === id);
    if (found) this.move = found;
    this.apply();
  }

  apply() {
    this.el.dataset.style = this.move.style;
  }

  /** Beats between changes for a part, resolving the meter-relative names. */
  every(part, meter) {
    const v = this.move[part];
    if (v === "bar") return meter;
    if (v === "bar2") return meter * 2;
    return v;
  }

  /** Called on real beats: the accent decides how hard the bounce lands. */
  onBeat(intensity) {
    this.intensity = intensity;
    this.el.style.setProperty("--hit", intensity.toFixed(3));
    this.el.classList.remove("idle");
  }

  /**
   * beatPos: continuous position in beats since the grid started.
   * barPos:  continuous position in bars, aligned to the detected downbeat.
   * Called every animation frame; it only touches the DOM when a tick changes.
   */
  update(beatPos, barPos, meter, periodMs) {
    if (this.auto) {
      const slot = Math.floor(barPos / BARS_PER_MOVE);
      if (slot !== this.autoAt) {
        this.autoAt = slot;
        // walk the list in order so the change is legible rather than random
        const i = ((slot % MOVES.length) + MOVES.length) % MOVES.length;
        this.move = MOVES[i];
        this.apply();
      }
    }

    const arms = Math.floor(beatPos / this.every("arms", meter));
    if (arms !== this.armTick) {
      this.armTick = arms;
      this.el.dataset.arms = String(((arms % 2) + 2) % 2);
      this.pulse(periodMs);
    }

    const hipEvery = this.every("hips", meter);
    const hips = Math.floor((barPos * meter) / hipEvery);
    if (hips !== this.hipTick) {
      this.hipTick = hips;
      this.el.dataset.hips = String(((hips % 2) + 2) % 2);
    }
  }

  /** The squash, restarted even when two ticks land close together. */
  pulse(periodMs) {
    const el = this.el;
    // the motion follows the tempo instead of running at a fixed speed
    el.style.setProperty("--beat-ms", `${Math.max(60, Math.round(periodMs * 0.45))}ms`);
    el.classList.remove("hit");
    void el.offsetWidth;
    el.classList.add("hit");
    window.clearTimeout(this._hitTimer);
    this._hitTimer = window.setTimeout(() => el.classList.remove("hit"), 90);
  }

  setDownbeat(on) {
    this.el.classList.toggle("downbeat", on);
  }

  idle() {
    this.el.classList.add("idle");
    this.el.classList.remove("downbeat");
  }
}
