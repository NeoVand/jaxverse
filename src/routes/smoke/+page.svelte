<script lang="ts">
	// The design proof sheet — not linked from the book. Everything the visual
	// constitution promises, on one page: every token in both themes, every
	// equation voice on a real formula, the band at three viewport widths, and
	// every state of every control. If two things here disagree, the book is
	// about to disagree with itself somewhere too.
	//
	// The engine smoke test lives at the bottom, where it always did.
	// aliased: this page also does arithmetic, and the component would shadow it
	import Formula from '$lib/components/ui/Math.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { setChapter } from '$lib/data/plates';
	import { MlpEngine } from '$lib/nn/mlp-engine';
	import { makeDataset2d } from '$lib/nn/datasets2d';

	setChapter(() => 'smoke');

	// ── the palette ──────────────────────────────────────────────────────────
	const SURFACES = ['--paper', '--band', '--surface', '--surface-2', '--line', '--line-soft'];
	const INKS = ['--ink', '--ink-2', '--ink-3'];
	const VOICES = [
		'--accent',
		'--accent-strong',
		'--accent-soft',
		'--warm',
		'--warm-soft',
		'--good',
		'--bad'
	];
	const CATS = Array.from({ length: 10 }, (_, i) => `--cat-${i}`);

	// ── the equation voices, each on a formula the book actually prints ──────
	const EQ = [
		{
			cls: 'eq-model',
			token: '--accent',
			role: 'learned, slot 1 — θ, w, Q',
			tex: '\\htmlClass{eq-model}{\\theta} \\leftarrow \\htmlClass{eq-model}{\\theta} - \\htmlClass{eq-knob}{\\eta} \\htmlClass{eq-world}{\\nabla \\mathcal{L}}'
		},
		{
			cls: 'eq-model-2',
			token: '--cat-8',
			role: 'learned, slot 2 — b, K',
			tex: '\\htmlClass{eq-out}{a} = \\htmlClass{eq-op}{\\sigma}(\\htmlClass{eq-model}{w}x + \\htmlClass{eq-model-2}{b})'
		},
		{
			cls: 'eq-model-3',
			token: '--cat-6',
			role: 'learned, slot 3 — v, V',
			tex: '\\mathrm{attention}(\\htmlClass{eq-model}{Q}, \\htmlClass{eq-model-2}{K}, \\htmlClass{eq-model-3}{V}) = \\htmlClass{eq-op}{\\operatorname{softmax}}\\!\\left(\\frac{\\htmlClass{eq-model}{Q}\\htmlClass{eq-model-2}{K}^{\\top}}{\\sqrt{d_k}} + \\htmlClass{eq-mute}{M}\\right)\\htmlClass{eq-model-3}{V}'
		},
		{
			cls: 'eq-world',
			token: '--warm',
			role: 'supplied by the world — x, y, r, G, ∇L',
			tex: '\\mathcal{D} = \\{(\\htmlClass{eq-world}{x^{(i)}}, \\htmlClass{eq-world}{y^{(i)}})\\}_{i=1}^{N}'
		},
		{
			cls: 'eq-op',
			token: '--cat-2',
			role: 'fixed machinery — σ, tanh, softmax, KL',
			tex: '\\htmlClass{eq-op}{\\operatorname{softmax}}_c(z) = \\frac{e^{z_c}}{\\sum_j e^{z_j}}'
		},
		{
			cls: 'eq-knob',
			token: '--cat-1',
			role: 'set, not learned — η, γ, β, T',
			tex: '\\htmlClass{eq-world}{G_t} = \\htmlClass{eq-world}{r_t} + \\htmlClass{eq-knob}{\\gamma}\\,\\htmlClass{eq-world}{r_{t+1}} + \\htmlClass{eq-knob}{\\gamma}^2 \\htmlClass{eq-world}{r_{t+2}} + \\cdots'
		},
		{
			cls: 'eq-out',
			token: '--good',
			role: 'the answer, as a probability — p_y, P(x)',
			tex: '\\mathcal{L}(\\htmlClass{eq-model}{\\theta}) = -\\log \\htmlClass{eq-out}{p_{\\htmlClass{eq-world}{y}}}'
		},
		{
			cls: 'eq-mute',
			token: '--ink-3',
			role: 'masked, ignored, negative — M, u⁻',
			tex: '-\\log\\Big(1 - \\htmlClass{eq-op}{\\sigma}\\big(\\htmlClass{eq-model}{v}^{\\top}\\htmlClass{eq-mute}{u^{-}_{k}}\\big)\\Big)'
		}
	];

	// ── the band, drawn at the three widths the book is walked at ────────────
	// Each sample is a real page fragment laid out at its real viewport width,
	// then scaled down to fit this sheet, so the proportions stay honest.
	const WIDTHS = [390, 1024, 1920];
	let sheetW = $state(0);
	const sampleH = $state<Record<number, number>>({});

	let sliderA = $state(0.4);
	let sliderB = $state(-1.2);
	let sliderC = $state(6);
	let segOn = $state('one');
	let chipOn = $state(true);

	// ── engine smoke test ────────────────────────────────────────────────────
	let log = $state<string[]>([]);
	let running = $state(false);
	const say = (s: string) => {
		log = [...log, s];
		console.log('[smoke]', s);
	};

	async function run() {
		running = true;
		log = [];
		const engine = new MlpEngine();
		try {
			const data = makeDataset2d('spirals', 400, 0.05, 7);
			say('init…');
			const t0 = performance.now();
			await engine.init(
				{ layers: [2, 16, 16, 2, 2], activation: 'tanh', loss: 'xent', lr: 5e-3, batchSize: 64 },
				{ x: data.x, y: data.labels, n: data.n }
			);
			say(
				`device=${engine.device} params=${engine.paramCount} (${Math.round(performance.now() - t0)}ms)`
			);

			let first = 0;
			let last = 0;
			await engine.train(300, (m) => {
				if (m.step === 1) first = m.loss;
				last = m.loss;
				if (m.step % 100 === 0)
					say(`step ${m.step} loss=${m.loss.toFixed(4)} (${m.stepMs.toFixed(0)}ms/step)`);
			});
			say(
				`loss ${first.toFixed(3)} → ${last.toFixed(3)} ${last < first ? '✓ learning' : '✗ NOT learning'}`
			);

			const ev = await engine.eval();
			say(`eval loss=${ev.loss.toFixed(4)} acc=${((ev.accuracy ?? 0) * 100).toFixed(1)}%`);

			const grid = new Float32Array(2 * 100);
			for (let i = 0; i < 100; i++) {
				grid[2 * i] = (i % 10) / 5 - 1;
				grid[2 * i + 1] = Math.floor(i / 10) / 5 - 1;
			}
			const pred = await engine.predict(grid, 100);
			say(`predict ok: ${pred.length} values, sample=${pred[0].toFixed(3)},${pred[1].toFixed(3)}`);

			const acts = await engine.activations(grid, 100);
			say(`activations ok: ${acts.layers.length} layers, widths=${acts.widths.join(',')}`);

			const h = acts.layers[2].slice(0, 2 * 100); // the 2-wide layer
			const tail = await engine.forwardFrom(3, h, 100);
			say(`forwardFrom ok: ${tail.length} values`);

			const g = await engine.inputGrad(grid.slice(0, 2), 1);
			say(`inputGrad ok: [${g[0].toExponential(2)}, ${g[1].toExponential(2)}]`);

			const w = await engine.weights();
			say(`weights ok: ${w.map((l) => `${l.inDim}×${l.outDim}`).join(' ')}`);

			const ck = await engine.exportCheckpoint();
			say(`export ok: ${ck.byteLength} bytes`);

			await engine.reset(99);
			const ev2 = await engine.eval();
			say(`reset ok: acc back to ${((ev2.accuracy ?? 0) * 100).toFixed(1)}%`);
			say('ALL OK');
		} catch (err) {
			say(`ERROR: ${err instanceof Error ? err.message : String(err)}`);
		} finally {
			await engine.dispose();
			running = false;
		}
	}
</script>

<svelte:head>
	<title>Proof sheet · jaxverse</title>
	<meta name="robots" content="noindex" />
</svelte:head>

{#snippet swatches(names: string[])}
	<div class="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
		{#each names as name (name)}
			<div class="flex items-center gap-2.5">
				<span
					class="h-6 w-6 shrink-0 rounded-[var(--r-1)] border border-line"
					style="background: var({name});"
				></span>
				<span class="num text-[11px] text-ink-2">{name}</span>
			</div>
		{/each}
	</div>
{/snippet}

{#snippet palette()}
	<div class="space-y-5 p-5">
		<div>
			<p class="eyebrow mb-2">surfaces</p>
			{@render swatches(SURFACES)}
		</div>
		<div>
			<p class="eyebrow mb-2">ink</p>
			{@render swatches(INKS)}
		</div>
		<div>
			<p class="eyebrow mb-2">voices</p>
			{@render swatches(VOICES)}
		</div>
		<div>
			<p class="eyebrow mb-2">categories</p>
			{@render swatches(CATS)}
		</div>
		<div>
			<p class="eyebrow mb-2">type</p>
			<p class="font-serif text-[1.6rem]" style="font-variation-settings: 'opsz' 28;">
				Newsreader carries the prose
			</p>
			<p class="text-[13px] text-ink-2">Inter carries the interface</p>
			<p class="num text-[12px]">0123456789 · JetBrains Mono carries every number</p>
		</div>
	</div>
{/snippet}

<article class="pb-24">
	<header class="rail-prose pt-16 pb-8">
		<p class="eyebrow mb-4">Proof sheet · not part of the book</p>
		<h1
			class="font-serif tracking-tight"
			style="font-size: clamp(2.2rem, 5vw, 3rem); line-height: 1.05; font-weight: 480; font-variation-settings: 'opsz' 60;"
		>
			The visual constitution
		</h1>
		<p class="mt-4 font-serif text-[1.15rem] leading-[1.5] text-ink-2 italic">
			Cool hues are learned inside the machine; warm hues come from outside it — vermilion from the
			world, amber from your hand. Teal is fixed machinery, green is the verdict, grey is ignored.
		</p>
	</header>

	<!-- ── tokens, both themes at once ── -->
	<section class="rail">
		<h2 class="h2 h2-flush px-4">Tokens</h2>
		<div class="grid gap-px bg-line-soft sm:grid-cols-2">
			<div class="theme-light" style="background: var(--paper); color: var(--ink);">
				<p class="eyebrow px-5 pt-4">light</p>
				{@render palette()}
			</div>
			<div class="theme-dark" style="background: var(--paper); color: var(--ink);">
				<p class="eyebrow px-5 pt-4">dark</p>
				{@render palette()}
			</div>
		</div>
	</section>

	<!-- ── the equation voices ── -->
	<section class="rail mt-16">
		<h2 class="h2 h2-flush px-4">Equation voices</h2>
		<div class="px-4">
			{#each EQ as v (v.cls)}
				<div class="grid gap-2 border-t border-line-soft py-4 sm:grid-cols-[14rem_1fr]">
					<div>
						<p class="num text-[12px]" style="color: var({v.token});">.{v.cls}</p>
						<p class="mt-0.5 text-[12px] text-ink-3">{v.role}</p>
						<p class="num mt-0.5 text-[11px] text-ink-3">{v.token}</p>
					</div>
					<div class="min-w-0 overflow-x-auto">
						<Formula display tex={v.tex} />
					</div>
				</div>
			{/each}
		</div>
	</section>

	<!-- ── the band at three widths ── -->
	<section class="rail mt-16">
		<h2 class="h2 h2-flush px-4">The band, at the three widths the book is walked at</h2>
		<div class="space-y-8 px-4" bind:clientWidth={sheetW}>
			{#each WIDTHS as w (w)}
				{@const scale = Math.min(1, (sheetW || 900) / w)}
				<div>
					<p class="eyebrow mb-2">{w}px · shown at {(scale * 100).toFixed(0)}%</p>
					<!-- a real page fragment at a real viewport width, scaled to fit here -->
					<div
						class="overflow-hidden border border-line-soft"
						style="width: {w * scale}px; height: {(sampleH[w] ?? 320) * scale}px;"
					>
						<div
							bind:clientHeight={sampleH[w]}
							style="width: {w}px; transform: scale({scale}); transform-origin: top left;"
						>
							<div class="chapter-prose rail-prose">
								<p>
									The reading rail is 42rem wide, and the band beneath it goes edge to edge — the
									page steps onto different paper for the length of a figure.
								</p>
							</div>
							<div class="band">
								<figure class="rail">
									<div class="plate-head">
										<span class="flex items-baseline gap-2.5">
											<span class="eyebrow plate-label">Plate I</span>
											<span class="plate-title">A specimen figure</span>
											<span class="plate-live"></span>
										</span>
										<span class="num text-[11px] text-ink-3">loss 0.412</span>
									</div>
									<div
										class="flex h-24 items-center justify-center border border-line-soft bg-surface"
									>
										<span class="num text-[11px] text-ink-3">the demo lives here</span>
									</div>
									<figcaption class="plate-caption">
										The caption sits on the plate rail, left-aligned, and never centred — it is
										prose about the figure, not a label under a picture.
									</figcaption>
								</figure>
							</div>
						</div>
					</div>
				</div>
			{/each}
		</div>
	</section>

	<!-- ── controls, every state ── -->
	<section class="rail mt-16">
		<h2 class="h2 h2-flush px-4">Controls</h2>
		<div class="grid gap-8 px-4 sm:grid-cols-2">
			<div>
				<p class="eyebrow mb-3">chips</p>
				<div class="flex flex-wrap items-center gap-1.5">
					<button class="chip" onclick={() => (chipOn = !chipOn)}>default</button>
					<button class="chip" class:chip-on={chipOn} onclick={() => (chipOn = !chipOn)}
						>selected</button
					>
					<button class="chip" disabled>disabled</button>
				</div>

				<p class="eyebrow mt-6 mb-3">segmented</p>
				<div class="seg">
					{#each ['one', 'two', 'three'] as s (s)}
						<button class:on={segOn === s} onclick={() => (segOn = s)}>{s}</button>
					{/each}
				</div>

				<p class="eyebrow mt-6 mb-3">solid</p>
				<button class="btn-solid rounded-[var(--r-2)] px-4 py-2 text-[13px]">Train</button>

				<p class="eyebrow mt-6 mb-3">verdicts</p>
				<p class="num text-[12px]">
					<span style="color: var(--good);">✓ legal</span> ·
					<span style="color: var(--bad);">✗ illegal</span> ·
					<span style="color: var(--ink-3);">ignored</span>
				</p>
			</div>

			<div>
				<p class="eyebrow mb-3">sliders — one per voice</p>
				<div class="space-y-3">
					<Slider
						label="θ · learned, slot 1"
						bind:value={sliderA}
						min={-1}
						max={1}
						step={0.01}
						tone="model"
						format={(v) => v.toFixed(2)}
					/>
					<Slider
						label="b · learned, slot 2"
						bind:value={sliderB}
						min={-3}
						max={3}
						step={0.05}
						tone="model-2"
						format={(v) => v.toFixed(2)}
					/>
					<Slider
						label="v · learned, slot 3"
						bind:value={sliderC}
						min={-8}
						max={8}
						step={0.1}
						tone="model-3"
						format={(v) => v.toFixed(1)}
					/>
					<Slider
						label="γ · a knob you set"
						bind:value={sliderA}
						min={-1}
						max={1}
						step={0.01}
						tone="knob"
						format={(v) => v.toFixed(2)}
					/>
					<Slider
						label="x · from the world"
						bind:value={sliderB}
						min={-3}
						max={3}
						step={0.05}
						tone="world"
						format={(v) => v.toFixed(2)}
					/>
					<Slider
						label="disabled"
						bind:value={sliderC}
						min={-8}
						max={8}
						step={0.1}
						tone="ink"
						disabled
						format={(v) => v.toFixed(1)}
					/>
				</div>
				<p class="mt-3 text-[12px] text-ink-3">
					Tab into any control: one focus halo, at one radius, tinted by what the control already
					means.
				</p>
			</div>
		</div>
	</section>

	<!-- ── the engine smoke test, where it always was ── -->
	<section class="rail-prose mt-20">
		<h2 class="h2 h2-flush">Engine smoke test</h2>
		<p class="mt-2 text-[13px] text-ink-2">
			Trains a small MLP on spirals and prints every stage of the worker contract.
		</p>
		<button
			class="btn-solid mt-4 rounded-[var(--r-2)] px-4 py-2 text-[13px]"
			onclick={run}
			disabled={running}>run smoke test</button
		>
		<pre class="num mt-6 text-[12px] leading-relaxed">{log.join('\n')}</pre>
	</section>
</article>
