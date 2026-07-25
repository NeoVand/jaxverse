# jaxverse — authoring guide

The single source of truth for building chapters. Read this whole file before
writing any chapter code. Deviating from it produces visual noise; don't.

## What this site is

An interactive book: eight chapters that teach deep learning by training real
models in the browser (jax-js on WebGPU, in Web Workers). Editorial, minimal,
warm. Think "physics textbook designed by a gallery": prose in a serif reading
column, demos as numbered _plates_ (figures), math interleaved and beautiful.

## Voice

- Precise, warm, zero hype. Short paragraphs (2–5 sentences).
- Define every term in prose the first time it appears, in _italics_.
- Address the reader as "you"; the model is "it".
- Never say "simply", "just", "magic". The math IS the explanation.
- Section headings inside chapters: use `<h2 class="...">` sparingly (2–4 per
  chapter), sentence case, serif.

## Page anatomy (copy this skeleton)

```svelte
<script lang="ts">
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import Wide from '$lib/components/ui/Wide.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import MyDemo from '$lib/components/demos/MyDemo.svelte';
</script>

<ChapterShell slug="descent">
	<Prose>
		<p>Opening paragraphs…</p>
		<Math
			display
			tex={'\\theta_{t+1} = \\theta_t - \\gamma \\nabla_\\theta \\mathcal{L}(\\theta_t)'}
		/>
		<p>More prose…</p>
	</Prose>

	<Wide>
		<MyDemo />
	</Wide>

	<Prose>
		<p>Reflection on what the reader just saw…</p>
	</Prose>
</ChapterShell>
```

- `ChapterShell slug=` must match the route folder and `src/lib/data/chapters.ts`.
- Prose column is 42rem; `Wide` (demos) is 64rem. Alternate them.
- All interlinks: `import { resolve } from '$app/paths'; href={resolve('/space')}`.
- All static asset fetches: `import { base } from '$app/paths'; fetch(`${base}/data/foo.bin`)`.

## Components (already built — use, don't reinvent)

- `Math` — `<Math tex={'…'} />` inline, `<Math display tex={'…'} />` block.
  KaTeX. Color terms with `\htmlClass{eq-a}{\gamma}` → accent; `eq-w` warm,
  `eq-g` good, `eq-m` muted. Color only terms the surrounding prose or a demo
  refers to by that color.
- `Plate` — every demo's frame. `<Plate n={1} title="The landscape" caption="…">
{#snippet status()}<span>…</span>{/snippet} …stage… </Plate>`.
  `n` numbers plates _within a chapter_ starting at 1. The caption explains
  what the reader is looking at, italic, one or two sentences.
- `Slider` — `<Slider label="learning rate γ" bind:value min max step format={(v) => …} tone="accent|warm|ink" />`
- `Btn` — `<Btn kind="primary" onclick={…}>Train</Btn>`; ghost by default.
- `Prose`, `Wide`, `ChapterShell` as above.
- Icons: `lucide-svelte`, named imports, `size={13|14|16}`, colored via
  `style="color: var(--…)"`, `aria-hidden="true"` when decorative.

## Design tokens (never hardcode colors)

Surfaces: `--paper --surface --surface-2 --line --line-soft`
Ink: `--ink --ink-2 --ink-3`
Voices: `--accent` (class A / links / "live"), `--warm` (class B / contrast),
`--good --bad`, ten categoricals `--cat-0…--cat-9` (digit classes).
Fonts: `--font-serif --font-sans --font-mono`.

Tailwind maps them: `bg-paper text-ink-2 border-line font-serif` etc.
Utility classes: `.eyebrow` (tracked uppercase label), `.num` (tabular mono),
`.chapter-prose` (serif reading style — Prose applies it).

In canvas code, read tokens at draw time:
`getComputedStyle(canvas).getPropertyValue('--accent')` — then colors follow
theme switches. Redraw loops pick this up automatically.

## Canvas / SVG conventions

- Canvas: handle devicePixelRatio (cap at 2), size from clientWidth/Height,
  `ctx.setTransform(dpr, 0, 0, dpr, 0, 0)` each frame. See
  `src/lib/components/ui/DescentMark.svelte` for the pattern.
- Honor `matchMedia('(prefers-reduced-motion: reduce)')`: no autoplaying
  animation; render a meaningful static frame instead.
- SVG for charts (loss curves): fixed viewBox, `preserveAspectRatio="none"`,
  string-built paths, hairline gridlines in `--line-soft`, labels `.num` at
  10–11px, axes unobtrusive. Train curve in `--accent`; val in `--warm`.
- Scatter points: radius 2.5–3.5, class A `--accent`, class B `--warm`,
  no stroke on fills except selection.

## Demo (plate) behavior contract

- Demos AUTO-LOAD when scrolled near: wrap the stage root with
  `use:inview={boot}` (`$lib/components/ui/inview`) — it fires once, ~160px
  before the plate enters the viewport. No "Load" buttons anywhere; while
  booting show the quiet loading row; the first button the reader sees is the
  one that DOES something (Train, Play, Sample). Never boot on mount directly
  (prerender + reader may never scroll there).
- Training demos run at one house pace — steps per chunk, pacing delays,
  episodes per frame are fixed constants tuned for reading. No speed controls.
- Engine lifecycle: `let engine: MlpEngine | null = null` (NOT $state);
  a `phase` $state: 'idle' | 'loading' | 'ready' | 'training' | 'error' |
  'no-webgpu' drives the UI. `onDestroy(() => engine?.dispose())`.
- Train in chunks (25–50 steps) between eval calls so charts show honest
  held-out curves; keep a `stopFlag` so Stop works mid-run.
- Status snippet shows live numbers: `step 240 · loss 0.312 · 12 ms/step`
  in `.num`.
- Every control must do something the reader can _see_. If a slider wouldn't
  visibly change the demo, cut it.
- WebGPU: `MlpEngine` falls back to cpu/wasm automatically (small models are
  fine). For LM demos (worker `src/lib/llm/*`), check
  `detectWebGPU()` from `$lib/nn/engine` and show a graceful prose fallback.

## Engine API (chapters 1–4) — src/lib/nn/

```ts
import { MlpEngine } from '$lib/nn/mlp-engine';
const engine = new MlpEngine();
await engine.init(
	{ layers: [2, 16, 16, 2], activation: 'tanh', loss: 'xent', lr: 5e-3, batchSize: 64, seed: 7 },
	{ x: Float32Array /* n×din */, y: Int32Array /* labels */ | Float32Array /* n×dout */, n }
);
await engine.train(200, (m) => {
	/* {step, loss, stepMs} per step */
});
await engine.stop();
const logits = await engine.predict(xFlat, n); // n×dout
const { layers, widths } = await engine.activations(xFlat, n); // post-act per layer
const out = await engine.forwardFrom(k, hFlat, n); // run tail from layer k
const g = await engine.inputGrad(xRow, classIdx); // saliency, one row
const { loss, accuracy } = await engine.eval(); // held-out, deterministic
const layersW = await engine.weights(); // per-layer W/b
await engine.setData({ x, y, n }); // swap data, keep weights
await engine.reset(seed); // re-roll weights
await engine.dispose();
```

Loss 'xent' expects Int32Array labels and outDim = nClasses; 'mse' expects
Float32Array targets (autoencoder: pass y = x). 2-D toy datasets:
`makeDataset2d('spirals' | 'circles' | 'moons' | 'blobs', n, noise, seed)` in
`$lib/nn/datasets2d`.

## Code standards

- Svelte 5 runes only ($state, $derived, $effect, $props). Tabs, single
  quotes (prettier config is authoritative — run `npm run format`).
- TypeScript strict; no `any` outside worker/jax-js seams.
- Comments explain _constraints_ ("jit caches per shape, so pad to chunks"),
  never narration.
- Validate every component: `npx @sveltejs/mcp svelte-autofixer <file>` until
  clean, then `npm run check` and `npm run lint`.
- Never add dependencies without asking.

## Math notation (consistent across chapters)

- Parameters θ, learning rate γ, loss ℒ (\mathcal{L}), gradient ∇ℒ,
  dataset {(x⁽ⁱ⁾, y⁽ⁱ⁾)}, prediction ŷ = f(x; θ).
- The update rule is written the same way in every chapter:
  θ ← θ − γ∇ℒ.
- Chapter-specific: softmax σ, cross-entropy H(p,q), KL divergence, reward R,
  policy π_θ, return G_t.
