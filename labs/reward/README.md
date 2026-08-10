# Lab 6 · Learning from Reward

REINFORCE on a sea chart in plain TypeScript. A boat cannot sail within 35° of the wind, and the
harbour is upwind — so watch a policy that has never heard of tacking work out how to beat to
windward. No GPU, no tensors.

This is a standalone lab from [jaxverse](https://github.com/NeoVand/jaxverse). It has no build
requirements beyond Node:

```sh
npm install
npm run dev
```

then open the printed localhost URL. Everything of interest is in `src/main.ts` — small on
purpose, so you can change a number and see what happens.
