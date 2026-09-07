# Lab 10 · The Straight Path

Rectified flow on 28 × 28 Fashion-MNIST. Interpolate along the straight line between a garment and
some noise, regress the velocity along it, and integrate back out with plain Euler — then pick two
class indices at the top of the file and guide (or mix the one-hot) toward a picture that is in no
training set.

This is a standalone lab from [jaxverse](https://github.com/NeoVand/jaxverse). It has no build
requirements beyond Node:

```sh
npm install
npm run dev
```

then open the printed localhost URL. Everything of interest is in `src/main.ts` — small on
purpose, so you can change a number and see what happens. Try `LABEL_A` and `LABEL_B` (0 T-shirt …
9 Ankle boot), the two guidance weights, `MIX`, or the step count (it stays legible far lower than
the diffusion lab's does).

Needs WebGPU for a usable step time; it will fall back to wasm and crawl.
