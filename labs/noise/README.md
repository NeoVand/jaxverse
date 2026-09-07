# Lab 9 · Out of the Static

A denoising diffusion model on 28 × 28 Fashion-MNIST: corrupt a garment by a random amount, learn to
name the noise that was added, then walk a fresh sheet of static back into a picture fifty steps at a
time. Sampling is unconditional DDIM; training still sees the class label (and drops it 10% of the
time, so the same weights know the unconditional field).

This is a standalone lab from [jaxverse](https://github.com/NeoVand/jaxverse). It has no build
requirements beyond Node:

```sh
npm install
npm run dev
```

then open the printed localhost URL. Everything of interest is in `src/main.ts` — small on
purpose, so you can change a number and see what happens. Try the schedule in `alphaBar`, the
step count in the `sample` call, or `LAYERS` and `DIM`.

Needs WebGPU for a usable step time; it will fall back to wasm and crawl.
