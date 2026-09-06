# Lab 10 · The Straight Path

Rectified flow on 32 × 32 emoji, with prompts. Interpolate along the straight line between a
picture and some noise, regress the velocity along it, and integrate back out with plain Euler —
then compose two prompts by adding their two guidance pushes and draw something that is in no
training set.

This is a standalone lab from [jaxverse](https://github.com/NeoVand/jaxverse). It has no build
requirements beyond Node:

```sh
npm install
npm run dev
```

then open the printed localhost URL. Everything of interest is in `src/main.ts` — small on
purpose, so you can change a number and see what happens. Try `PROMPT_A` and `PROMPT_B`, the two
guidance weights, or the step count (it stays legible far lower than the diffusion lab's does).

Needs WebGPU for a usable step time; it will fall back to wasm and crawl.
