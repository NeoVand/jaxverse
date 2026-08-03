<script lang="ts">
	import { resolve } from '$app/paths';
	import { chapters, type ChapterSlug } from '$lib/data/chapters';
	import { progress } from '$lib/data/progress.svelte';
	import HeroField from '$lib/components/ui/HeroField.svelte';
	import {
		ArrowRight,
		BookOpen,
		Castle,
		ChevronRight,
		Gamepad2,
		Grid3x3,
		Map,
		Mountain,
		PenLine,
		Spline,
		Type
	} from 'lucide-svelte';

	// One glyph per chapter — presentation only, so the mapping lives here.
	const glyphs: Record<ChapterSlug, typeof Mountain> = {
		descent: Mountain,
		neuron: Spline,
		space: Grid3x3,
		digits: PenLine,
		latent: Map,
		language: Type,
		reward: Gamepad2,
		rook: Castle
	};

	// The contents rail: one smooth descent curve threading every chapter node.
	// Node i sits at row center (i + 0.5) / N in a unit viewBox stretched to
	// the list's real height; the curve wanders left–right between nodes.
	const N = chapters.length + 1; // + epilogue
	const nodeY = (i: number) => (i + 0.5) / N;
	const nodeX = (i: number) => 0.5 + 0.34 * Math.sin(i * 1.7 + 0.6);

	const railPath = (() => {
		let d = `M ${nodeX(0)} ${nodeY(0)}`;
		for (let i = 1; i < N; i++) {
			const x0 = nodeX(i - 1);
			const y0 = nodeY(i - 1);
			const x1 = nodeX(i);
			const y1 = nodeY(i);
			const my = (y0 + y1) / 2;
			d += ` C ${x0} ${my}, ${x1} ${my}, ${x1} ${y1}`;
		}
		return d;
	})();
</script>

<svelte:head>
	<title>jaxverse — a little universe of learning machines</title>
	<meta
		name="description"
		content="An interactive book about deep learning. Train real neural networks — from a single neuron to a language model — live in your browser, and watch what they learn."
	/>
</svelte:head>

<!-- ── hero ── -->
<section class="mx-auto max-w-2xl px-5 pt-20 pb-8 sm:pt-28">
	<p class="eyebrow mb-6">An interactive book, computed as you read</p>
	<h1
		class="font-serif tracking-tight text-balance"
		style="font-size: clamp(2.8rem, 7vw, 4.4rem); line-height: 1.03; font-weight: 460; font-variation-settings: 'opsz' 72;"
	>
		A little universe of learning&nbsp;machines
	</h1>
	<p
		class="mt-6 font-serif text-[1.3rem] leading-[1.55] text-ink-2 italic"
		style="font-variation-settings: 'opsz' 20;"
	>
		Every model in this book is real, and every one of them trains here, in your browser — from a
		single neuron bending a line to a language model learning chess.
	</p>

	<div class="mt-8 flex flex-wrap items-center gap-3">
		<a
			href={resolve('/descent')}
			class="btn-solid inline-flex items-center gap-2 rounded-md px-4 py-2 text-[11.5px] font-medium tracking-[0.1em] uppercase"
		>
			Begin the descent <ArrowRight size={13} aria-hidden="true" />
		</a>
		<a
			href="https://github.com/NeoVand/jaxverse"
			rel="external"
			class="cta-ghost inline-flex items-center gap-2 rounded-md px-4 py-2 text-[11.5px] font-medium tracking-[0.1em] uppercase"
		>
			<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
				<path
					d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
				/>
			</svg>
			View the source
		</a>
	</div>

	<div class="mt-8">
		<HeroField height={300} />
	</div>
	<p
		class="mt-2.5 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-[12px] text-ink-3"
	>
		<span class="inline-flex items-center gap-1.5">
			<span class="inline-block h-2 w-2 rounded-full" style="background: var(--ink-3);"></span>
			gradient descent
		</span>
		<span class="inline-flex items-center gap-1.5">
			<span class="inline-block h-2 w-2 rounded-full" style="background: var(--accent);"></span>
			with momentum
		</span>
		<span class="inline-flex items-center gap-1.5">
			<span class="inline-block h-2 w-2 rounded-full" style="background: var(--warm);"></span>
			Adam
		</span>
		<span class="font-serif italic">— live now; click the map to drop them somewhere new</span>
	</p>
</section>

<!-- ── manifesto ── -->
<section class="chapter-prose mx-auto max-w-2xl px-5 py-8">
	<p>
		Machine learning has a reputation for mystery it doesn’t deserve. Under every headline model —
		the ones that talk, draw, and play — sits one modest idea: build a smooth, adjustable function,
		measure how wrong it is, and nudge every knob a little downhill. Do this a few million times.
		That’s the whole trick. Everything else is architecture and appetite.
	</p>
	<p>
		This book makes that idea physical. It is arranged as a descent: eight chapters, each one lower
		on the loss surface than the last. You’ll fit curves, bend space until tangled spirals come
		apart, teach a network to read handwriting, watch a map of meaning assemble itself without a
		single label, and end by training a small language model three different ways — the same
		pipeline, in miniature, that produced the models everyone is talking about.
	</p>
	<p>
		Nothing here is a recording. The numbers move because a network is training on your GPU as you
		watch, courtesy of <a href="https://jax-js.com" rel="external">jax-js</a>. Pause anywhere. Break
		things. The models are small and do not mind.
	</p>
</section>

<!-- ── the hidden course ── -->
<section class="mx-auto max-w-2xl px-5 py-8">
	<h2 class="eyebrow mb-4">A second book, for the builders</h2>
	<div class="hood-ad rounded-lg px-4 py-3">
		<span class="flex flex-wrap items-center gap-x-3 gap-y-1">
			<ChevronRight size={14} aria-hidden="true" style="color: var(--accent);" />
			<span class="eyebrow" style="color: var(--accent);">Under the hood</span>
			<span class="font-serif text-[15px] italic" style="font-variation-settings: 'opsz' 14;">
				the code each plate runs
			</span>
			<span class="num ml-auto text-[10.5px] text-ink-3">collapsed, until you want it</span>
		</span>
	</div>
	<p class="chapter-prose mt-4">
		Beneath every plate that trains something sits a quiet line like the one above. Open it and you
		get the machinery that plate just ran — the real jax-js code from this repository, annotated,
		with the stagecraft on its own tab. Followed chapter by chapter, the blocks add up to a short
		course in <a href="https://jax-js.com" rel="external">jax-js</a> itself: pytrees,
		<code>valueAndGrad</code>, <code>jit</code>, devices, workers, a whole transformer. Each chapter
		also ends with a downloadable lab — a standalone npm project that runs the same model with
		nothing but <code>npm install</code>.
	</p>
</section>

<!-- ── contents ── -->
<section id="contents" class="mx-auto max-w-3xl px-5 py-14">
	<h2 class="eyebrow mb-8">Contents — the descent</h2>

	<div class="relative">
		<!-- the rail -->
		<svg
			class="pointer-events-none absolute top-0 left-0 hidden h-full w-14 sm:block"
			viewBox="0 0 1 1"
			preserveAspectRatio="none"
			aria-hidden="true"
		>
			<path
				d={railPath}
				fill="none"
				stroke="var(--line)"
				stroke-width="1.5"
				vector-effect="non-scaling-stroke"
			/>
		</svg>

		<ol class="relative">
			{#each chapters as c, i (c.slug)}
				{@const Glyph = glyphs[c.slug]}
				<li class="relative sm:pl-14">
					<!-- node -->
					<span
						class="absolute top-1/2 hidden h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 sm:block"
						style="left: calc({nodeX(i)} * 3.5rem); border-color: {progress.visited.has(c.slug)
							? 'var(--warm)'
							: 'var(--ink-3)'}; background: {progress.visited.has(c.slug)
							? 'var(--warm)'
							: 'var(--paper)'};"
						aria-hidden="true"
					></span>

					<a
						href={resolve(`/${c.slug}`)}
						class="group -mx-3 my-1 flex items-baseline gap-4 rounded-lg border border-line-soft border-transparent px-3 py-4 transition-colors hover:border-line hover:bg-surface"
					>
						<span class="num w-5 shrink-0 text-right text-[13px] text-ink-3">{c.n}</span>
						<span class="min-w-0">
							<span class="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
								<Glyph
									size={15}
									aria-hidden="true"
									class="relative top-[1px] shrink-0 self-center text-ink-3 transition-colors group-hover:text-[var(--accent)]"
								/>
								<span
									class="font-serif text-[1.35rem] tracking-tight"
									style="font-weight: 500; font-variation-settings: 'opsz' 24;"
								>
									{c.title}
								</span>
								<span class="eyebrow" style="color: var(--accent);">{c.kicker}</span>
							</span>
							<span class="mt-1 block max-w-xl text-[13.5px] leading-relaxed text-ink-2">
								{c.deck}
							</span>
						</span>
						<span class="num ml-auto shrink-0 text-[11px] text-ink-3">
							<span class="group-hover:hidden">{c.minutes}′</span>
							<ArrowRight
								size={13}
								aria-hidden="true"
								class="hidden text-[var(--accent)] group-hover:inline"
							/>
						</span>
					</a>
				</li>
			{/each}

			<!-- epilogue node -->
			<li class="relative sm:pl-14">
				<span
					class="absolute top-1/2 hidden h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 sm:block"
					style="left: calc({nodeX(
						chapters.length
					)} * 3.5rem); border-color: var(--ink-3); background: var(--paper);"
					aria-hidden="true"
				></span>
				<a
					href={resolve('/epilogue')}
					class="group -mx-3 my-1 flex items-baseline gap-4 rounded-lg border border-transparent px-3 py-4 transition-colors hover:border-line hover:bg-surface"
				>
					<span class="num w-5 shrink-0 text-right text-[13px] text-ink-3">∞</span>
					<span>
						<span class="flex items-center gap-3">
							<BookOpen
								size={15}
								aria-hidden="true"
								class="shrink-0 text-ink-3 transition-colors group-hover:text-[var(--accent)]"
							/>
							<span
								class="font-serif text-[1.35rem] tracking-tight"
								style="font-weight: 500; font-variation-settings: 'opsz' 24;"
							>
								Epilogue
							</span>
						</span>
						<span class="mt-1 block max-w-xl text-[13.5px] leading-relaxed text-ink-2">
							Where the ideas came from, and where to go next.
						</span>
					</span>
				</a>
			</li>
		</ol>
	</div>
</section>

<style>
	.cta-ghost {
		font-family: var(--font-sans);
		border: 1px solid var(--line);
		color: var(--ink-2);
		transition:
			border-color 100ms ease,
			color 100ms ease;
	}
	.cta-ghost:hover {
		border-color: var(--ink-3);
		color: var(--ink);
	}

	/* a facsimile of the under-the-hood bar, so readers recognize it later */
	.hood-ad {
		border: 1px dashed var(--line);
		background: color-mix(in srgb, var(--surface-2) 45%, var(--surface));
	}
	.chapter-prose code {
		font-family: var(--font-mono);
		font-size: 0.86em;
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0.05em 0.3em;
	}
</style>
