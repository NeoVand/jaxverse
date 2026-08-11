<script lang="ts">
	import { resolve } from '$app/paths';
	import { chapters } from '$lib/data/chapters';
	import { chapterGlyphs, epilogueGlyph as BookOpen } from '$lib/data/glyphs';
	import { progress } from '$lib/data/progress.svelte';
	import HeroField from '$lib/components/ui/HeroField.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import { setChapter } from '$lib/data/plates';
	import { OPTIMIZER_LEGEND } from '$lib/optim/optimizers';
	import { ArrowRight } from 'lucide-svelte';

	// The frontispiece is a plate too, so the reader meets the grammar before
	// the first chapter uses it.
	setChapter(() => 'home');

	// The contents rail: one thin thread wandering through every chapter node,
	// with a soft glow drifting down it. Node positions come from the *measured*
	// center of each row (rows have different heights, so equal division would
	// drift), and everything stays inside the 56px rail — the row hover box
	// starts at 44px and can never touch it.
	const N = chapters.length + 1; // + epilogue
	const RAIL_W = 56;
	let railBox: HTMLElement | undefined = $state();
	let rowEls: HTMLElement[] = $state([]);
	let railH = $state(0);
	let centers = $state<number[]>([]);
	const nodeX = (i: number) => 22 + 11 * Math.sin(i * 1.7 + 0.6); // 11…33px

	$effect(() => {
		if (!railBox) return;
		const box = railBox;
		const measure = () => {
			railH = box.clientHeight;
			const boxTop = box.getBoundingClientRect().top;
			centers = rowEls.filter(Boolean).map((el) => {
				// align each node with the row's chapter number, not the row's
				// geometric center — the number is the first span in the link
				const r = (el.querySelector('a > span') ?? el).getBoundingClientRect();
				return r.top - boxTop + r.height / 2;
			});
		};
		measure();
		const ro = new ResizeObserver(measure);
		ro.observe(box);
		return () => ro.disconnect();
	});

	const spine = $derived.by(() => {
		if (centers.length < N) return '';
		let d = `M ${nodeX(0)} ${centers[0]}`;
		for (let i = 1; i < N; i++) {
			const my = (centers[i - 1] + centers[i]) / 2;
			d += ` C ${nodeX(i - 1)} ${my}, ${nodeX(i)} ${my}, ${nodeX(i)} ${centers[i]}`;
		}
		return d;
	});
</script>

<svelte:head>
	<title>jaxverse — a little universe of learning machines</title>
	<meta
		name="description"
		content="An interactive book about deep learning. Train real neural networks — from a single neuron to a language model — live in your browser, and watch what they learn."
	/>
</svelte:head>

<!-- ── hero ── -->
<section class="rail-prose pt-20 pb-8 sm:pt-28">
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
</section>

<Plate
	id="field"
	title="The loss landscape, live"
	live
	caption="Three optimizers descending the same surface, right now, on your machine. Click anywhere on the map to drop them somewhere new and watch the routes diverge — the whole of the Prologue is in this picture."
>
	<HeroField height={300} />
	<p class="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1 px-4 text-[12px] text-ink-3">
		{#each OPTIMIZER_LEGEND as o (o.id)}
			<span class="inline-flex items-center gap-1.5">
				<span class="inline-block h-2 w-2 rounded-full" style="background: var({o.token});"></span>
				{o.label}
			</span>
		{/each}
	</p>
</Plate>

<section class="rail-prose pb-8">
	<div class="flex flex-wrap items-center justify-center gap-3">
		<a
			href={resolve('/descent')}
			class="btn-solid inline-flex items-center gap-2 rounded-md px-5 py-2.5 text-[11.5px] font-medium tracking-[0.1em] uppercase"
		>
			Begin the descent <ArrowRight size={13} aria-hidden="true" />
		</a>
		<a
			href="https://github.com/NeoVand/jaxverse"
			rel="external"
			class="cta-ghost inline-flex items-center gap-2 rounded-md px-5 py-2.5 text-[11.5px] font-medium tracking-[0.1em] uppercase"
		>
			<!-- Feather Icons “github” (MIT) — the standard outlined mark -->
			<svg
				width="13"
				height="13"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				stroke-width="2"
				stroke-linecap="round"
				stroke-linejoin="round"
				aria-hidden="true"
			>
				<path
					d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"
				/>
			</svg>
			View the source
		</a>
	</div>
</section>

<!-- ── manifesto ── -->
<section class="chapter-prose rail-prose py-8">
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
<section class="rail-prose pt-8">
	<h2 class="h2 h2-flush">A second book, for the builders</h2>
</section>
<!-- a real block, not a mock-up: it opens on the code behind the hero map -->
<UnderTheHood slug="home" block="hero" />
<section class="rail-prose pb-8">
	<p class="chapter-prose">
		Beneath every plate that trains something sits a quiet line like the one above — and the one
		above works: open it for the machinery behind the map at the top of this page. Every block holds
		the real jax-js code from this repository, annotated, with the stagecraft on its own tab.
		Followed chapter by chapter, the blocks add up to a short course in
		<a href="https://jax-js.com" rel="external">jax-js</a> itself: pytrees,
		<code>valueAndGrad</code>, <code>jit</code>, devices, workers, a whole transformer. Each chapter
		also ends with a downloadable lab — a standalone npm project that runs the same model with
		nothing but <code>npm install</code>.
	</p>
</section>

<!-- ── contents ── -->
<section id="contents" class="mx-auto max-w-3xl px-5 py-14">
	<h2 class="h2 h2-flush mb-8">Contents — the descent</h2>

	<div class="relative" bind:this={railBox}>
		<!-- the rail: one thread through every chapter, a soft glow drifting down it -->
		{#if spine}
			<svg
				class="pointer-events-none absolute top-0 left-0 hidden h-full w-14 sm:block"
				viewBox="0 0 {RAIL_W} {railH}"
				aria-hidden="true"
			>
				<defs>
					<filter id="rail-glow" x="-200%" y="-200%" width="500%" height="500%">
						<feGaussianBlur stdDeviation="4" />
					</filter>
				</defs>
				<path d={spine} fill="none" stroke="var(--line)" stroke-width="1.2" />
				<!-- the signal: a blurred halo and its small bright heart, drifting together -->
				<g class="rail-signal">
					<circle r="7" fill="var(--accent)" opacity="0.35" filter="url(#rail-glow)">
						<animateMotion dur="14s" repeatCount="indefinite" path={spine} />
					</circle>
					<circle r="1.6" fill="var(--accent)" opacity="0.8">
						<animateMotion dur="14s" repeatCount="indefinite" path={spine} />
					</circle>
				</g>
				<!-- the chapter nodes, at the exact measured center of each row -->
				{#each chapters as c, i (c.slug)}
					<circle
						cx={nodeX(i)}
						cy={centers[i]}
						r="4"
						fill={progress.visited.has(c.slug) ? 'var(--warm)' : 'var(--paper)'}
						stroke={progress.visited.has(c.slug) ? 'var(--warm)' : 'var(--ink-3)'}
						stroke-width="1.6"
					/>
				{/each}
				<circle
					cx={nodeX(chapters.length)}
					cy={centers[chapters.length]}
					r="4"
					fill="var(--paper)"
					stroke="var(--ink-3)"
					stroke-width="1.6"
				/>
			</svg>
		{/if}

		<ol>
			{#each chapters as c, i (c.slug)}
				{@const Glyph = chapterGlyphs[c.slug]}
				<li class="sm:pl-14" bind:this={rowEls[i]}>
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
			<li class="sm:pl-14" bind:this={rowEls[chapters.length]}>
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

	/* the glow drifting down the contents thread disappears for readers who
	   asked for reduced motion */
	@media (prefers-reduced-motion: reduce) {
		.rail-signal {
			display: none;
		}
	}

	.chapter-prose code {
		font-family: var(--font-mono);
		font-size: 0.86em;
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0.05em 0.3em;
	}
</style>
