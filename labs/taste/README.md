# Lab 7 · Teaching Taste

Fit a Bradley–Terry judge to pairwise comparisons drawn from a taste it never sees, then turn an
optimizer loose on the judge and watch the two part company. Prints the over-optimization curve —
proxy up, gold over the top — and then the same run on a leash of length β.

This is a standalone lab from [jaxverse](https://github.com/NeoVand/jaxverse). It has no build
requirements beyond Node:

```sh
npm install
npm run dev
```

then open the printed localhost URL. Everything of interest is in `src/main.ts` — small on
purpose, so you can change a number and see what happens. Start with `TRUE_BEST`, the hidden
taste, or drop the comparison budget below thirty and watch the judge get worse in a way that
only the right-hand column can see.
