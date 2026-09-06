# denoiser-bench

The measurement behind one sentence in Chapter 9: _"264 milliseconds a step
against the transformer's 88, at the same batch on the same laptop."_

A conv U-Net and a patch transformer, matched at roughly the parameter count
the chapter ships, timed end to end — forward, backward and optimizer — on
WebGPU. It also measures the two things that turned out to matter more than
the architecture choice: folding the Adam update inside the `jit` boundary
(151 ms → 88 ms), and batch size, which is nearly free because the step is
bound by kernel launches rather than by arithmetic.

```sh
node scripts/bench-denoiser.mjs
```

Numbers from the run the chapter quotes, on an M-series MacBook, batch 32,
32 × 32 × 4 images:

| model                       | params | ms/step |
| --------------------------- | ------ | ------- |
| U-Net (3×3 convs, full res) | 1.58M  | 264     |
| DiT (patch 4, 64 tokens)    | 2.53M  | 166     |
| DiT, Adam fused into jit    | 2.53M  | 88      |
