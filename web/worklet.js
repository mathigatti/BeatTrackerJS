// Collects the 128-sample blocks the AudioWorklet delivers into hops of 441
// (what madmom expects at 44100 Hz and 100 fps) and posts them to the main thread.

const HOP = 441;

class HopCollector extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(HOP);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;

    let read = 0;
    while (read < channel.length) {
      const take = Math.min(HOP - this.filled, channel.length - read);
      this.buf.set(channel.subarray(read, read + take), this.filled);
      this.filled += take;
      read += take;
      if (this.filled === HOP) {
        // a copy is transferred: the buffer is reused on the next round
        const out = this.buf.slice();
        this.port.postMessage(out, [out.buffer]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("hop-collector", HopCollector);
