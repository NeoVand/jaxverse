# jaxverse tools

Small standalone pages that are not part of the site, run through Playwright by
the scripts in `../scripts`.

- `denoiser-bench` — the U-Net vs transformer timing the diffusion chapter quotes.
  `node scripts/bench-denoiser.mjs`
- `fashion-trainer` — trains the two shipped diffusion checkpoints in headless
  Chromium, so the weights the book ships come out of the same jax-js code a
  reader runs. `node scripts/train-fashion.mjs --objective flow --minutes 180`
- `checkpoint-proof` — draws proof sheets from those checkpoints (step budgets,
  class grids, guidance, morphs, and a nearest-neighbour check against the
  corpus) so the captions that make empirical claims can be checked against
  pictures. `node scripts/verify-fashion.mjs`
