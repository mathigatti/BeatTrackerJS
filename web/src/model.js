// Loads the madmom dump (madmom-beats.bin + .json) into typed arrays.

export function loadModel(manifest, buffer) {
  const view = (name) => {
    const e = manifest.arrays[name];
    if (!e) throw new Error(`missing array ${name}`);
    return new Float32Array(buffer, e.offset, e.count);
  };
  const mat = (name) => {
    const e = manifest.arrays[name];
    return { data: view(name), rows: e.shape[0], cols: e.shape[1] ?? 1 };
  };

  const nets = manifest.nets.map((net, ni) =>
    net.layers.map((layer, li) => {
      const p = `net${ni}.layer${li}`;
      if (layer.type === "lstm") {
        const gate = (part) => ({
          w: mat(`${p}.${part}.weights`),
          r: mat(`${p}.${part}.recurrent_weights`),
          b: view(`${p}.${part}.bias`),
          peep: manifest.arrays[`${p}.${part}.peephole_weights`]
            ? view(`${p}.${part}.peephole_weights`)
            : null,
        });
        return {
          type: "lstm",
          size: layer.size,
          inputSize: layer.input_size,
          inputGate: gate("input_gate"),
          forgetGate: gate("forget_gate"),
          cell: gate("cell"),
          outputGate: gate("output_gate"),
          init: view(`${p}.init`),
          cellInit: view(`${p}.cell_init`),
        };
      }
      return {
        type: "dense",
        size: layer.size,
        inputSize: layer.input_size,
        w: mat(`${p}.weights`),
        b: view(`${p}.bias`),
        activation: layer.activation,
      };
    })
  );

  return {
    config: manifest,
    nets,
    filterbank: {
      weights: view("filterbank.weights"),
      starts: view("filterbank.starts"),
      lengths: view("filterbank.lengths"),
      centerFreqs: view("filterbank.center_freqs"),
      numBands: manifest.num_bands,
    },
    window: view("window"),
  };
}
