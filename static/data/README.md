# static/data

Every file here is generated; regenerate with the named script, never edit by hand.

- `mnist-train.png` — first 8000 MNIST train digits as a 2800×2240 grayscale spritesheet (100 tiles/row, 28×28 each); `scripts/build-mnist.mjs`.
- `mnist-test.png` — first 2000 MNIST test digits, 2800×560, same layout; `scripts/build-mnist.mjs`.
- `mnist-labels.bin` — Uint8, 8000 train labels then 2000 test labels; `scripts/build-mnist.mjs`.
- `mnist-meta.json` — `{side, cols, train, test}` spritesheet geometry; `scripts/build-mnist.mjs`.
- `text-tokens.bin` — Uint8 char-level token stream, ~1.5M chars of TinyStories-style text cut at a story boundary; `scripts/build-corpus.mjs`.
- `text-vocab.json` — `{chars, vocabSize}` sorted char vocabulary (69 chars) for `text-tokens.bin`; `scripts/build-corpus.mjs`.
- `rook-tokens.bin` — Uint16 random-legal chess pretraining stream (`<game>`=0, then UCI move ids), copied verbatim from llmvibes; `scripts/copy-rook-data.mjs`.
- `rook-vocab.json` — `{moves, vocabSize}` UCI move vocabulary (1931 ids incl. `<game>`), copied verbatim; `scripts/copy-rook-data.mjs`.
- `rook-probe.json` — linear-probe results (board state from residual stream, per layer/square), copied verbatim; `scripts/copy-rook-data.mjs`.
- `rook-sft-tokens.bin` — Uint16 SFT stream of greedy-material games (both sides take the biggest capture, else check half the time), same format as `rook-tokens.bin`; `scripts/build-chess-sft.mjs`.
- `timemachine/rook-manifest.json` — training time-machine manifest trimmed to 4 waypoints (steps 0, 303, 1269, 2600), each with a `file` field; `scripts/copy-rook-data.mjs`.
- `timemachine/rook-w{0,8,10,11}.i8` — int8-quantized weight snapshots for those waypoints (dequantize per leaf with the manifest scales); `scripts/copy-rook-data.mjs`.
