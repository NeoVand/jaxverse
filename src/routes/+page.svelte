<script lang="ts">
	import { resolve } from '$app/paths';
	import { chapters } from '$lib/data/chapters';
	import { progress } from '$lib/data/progress.svelte';
	import DescentMark from '$lib/components/ui/DescentMark.svelte';

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

	<div class="mt-10 rounded-lg border border-line-soft">
		<DescentMark height={200} />
	</div>
	<p class="mt-2 text-center font-serif text-[13px] text-ink-3 italic">
		gradient descent, running now — the only trick this whole book needs
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
						<span class="num ml-auto shrink-0 text-[11px] text-ink-3">{c.minutes}′</span>
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
						<span
							class="font-serif text-[1.35rem] tracking-tight"
							style="font-weight: 500; font-variation-settings: 'opsz' 24;"
						>
							Epilogue
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
