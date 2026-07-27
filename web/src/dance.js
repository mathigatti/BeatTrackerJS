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
//
// A move can also declare a `flow`: a slow continuous motion spanning a whole
// bar or two, the way people roll their hips across a bar rather than snapping
// on each beat. That cannot be a 0/1 toggle, so the Dancer publishes `--flow`,
// a value sweeping -1..1 over the flow's period, and the CSS reads it directly.
// Anything driven by --flow must not carry a transition, or it would lag a value
// that already changes every frame.

export const MOVES = [
  { id: "swing", label: "Swing", style: "swing", arms: 1, hips: "bar" },
  { id: "swing2", label: "Swing · double", style: "swing", arms: 0.5, hips: "bar" },
  { id: "swingh", label: "Swing · half", style: "swing", arms: 2, hips: "bar2" },
  { id: "twist", label: "Twist", style: "twist", arms: 1, hips: 0.5 },
  { id: "robot", label: "Robot", style: "robot", arms: 1, hips: "bar" },
  { id: "robot2", label: "Robot · double", style: "robot", arms: 0.5, hips: "bar" },
  { id: "jump", label: "Jump", style: "jump", arms: 2, hips: 1 },
  // --- continuous ones: the body keeps moving between the beats
  { id: "sway", label: "Sway", style: "sway", arms: 2, hips: "bar", flow: "bar" },
  { id: "hips", label: "Hip roll", style: "hips", arms: 1, hips: "bar", flow: "bar" },
  { id: "hips2", label: "Hip roll · slow", style: "hips", arms: 2, hips: "bar2", flow: "bar2" },
  { id: "walk", label: "Walk back & forth", style: "walk", arms: 1, hips: "bar", flow: "walk" },
  { id: "lean", label: "Lean", style: "lean", arms: 2, hips: "bar2", flow: "walk" },
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
    if (!this.move.flow) this.el.style.setProperty("--flow", "0");
  }

  /**
   * The continuous channel, -1..1.
   *   bar   one full cycle per bar, crossing zero on the downbeat
   *   bar2  one full cycle every two bars
   *   walk  starts at one extreme on the downbeat and takes two bars to return,
   *         so the body travels one way for a whole bar and back the next
   */
  flowValue(barPos) {
    switch (this.move.flow) {
      case "bar": return Math.sin(2 * Math.PI * barPos);
      case "bar2": return Math.sin(Math.PI * barPos);
      case "walk": return -Math.cos(Math.PI * barPos);
      default: return 0;
    }
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

    if (this.move.flow) {
      this.el.style.setProperty("--flow", this.flowValue(barPos).toFixed(4));
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
    this.el.style.setProperty("--flow", "0");
  }
}
