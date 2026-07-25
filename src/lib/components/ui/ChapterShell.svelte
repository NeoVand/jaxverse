<script lang="ts">
	import type { Snippet } from 'svelte';
	import { resolve } from '$app/paths';
	import { ArrowLeft, ArrowRight } from 'lucide-svelte';
	import { chapterBySlug, neighbors, type Chapter } from '$lib/data/chapters';
	import { progress } from '$lib/data/progress.svelte';

	// The chapter page anatomy: ghost numeral, eyebrow, serif title, italic
	// deck — then the chapter body, then quiet prev/next cards.
	interface Props {
		slug: string;
		children: Snippet;
	}

	let { slug, children }: Props = $props();

	const chapter = $derived(chapterBySlug.get(slug) as Chapter);
	const nav = $derived(neighbors(slug));

	$effect(() => {
		progress.visit(slug);
	});
</script>

<svelte:head>
	<title>{chapter.title} · jaxverse</title>
	<meta name="description" content={chapter.deck} />
</svelte:head>

<article class="pb-10">
	<header class="relative mx-auto max-w-2xl px-5 pt-16 pb-10 sm:pt-24">
		<span
			aria-hidden="true"
			class="pointer-events-none absolute top-6 right-2 font-serif italic select-none sm:-right-16"
			style="font-size: clamp(7rem, 18vw, 13rem); line-height: 1; color: var(--ink); opacity: 0.055; font-variation-settings: 'opsz' 72;"
		>
			{chapter.n}
		</span>
		<p class="eyebrow mb-4">
			{chapter.n === 0 ? 'Prologue' : `Chapter ${chapter.n}`} · {chapter.kicker} · ≈{chapter.minutes}
			min
		</p>
		<h1
			class="font-serif tracking-tight text-balance"
			style="font-size: clamp(2.5rem, 6vw, 3.6rem); line-height: 1.05; font-weight: 480; font-variation-settings: 'opsz' 60;"
		>
			{chapter.title}
		</h1>
		<p
			class="mt-5 font-serif text-[1.28rem] leading-[1.5] text-ink-2 italic"
			style="font-variation-settings: 'opsz' 20;"
		>
			{chapter.deck}
		</p>
	</header>

	{@render children()}

	<nav class="mx-auto mt-20 max-w-5xl px-5" aria-label="Chapter navigation">
		<div class="grid grid-cols-1 gap-3 border-t border-line-soft pt-6 sm:grid-cols-2">
			{#if nav.prev}
				<a
					href={resolve(`/${nav.prev.slug}`)}
					class="group rounded-lg border border-line-soft p-4 transition-colors hover:border-line"
				>
					<span class="eyebrow flex items-center gap-1.5">
						<ArrowLeft size={11} />
						{nav.prev.n === 0 ? 'Prologue' : `Chapter ${nav.prev.n}`}
					</span>
					<span class="mt-1.5 block font-serif text-[17px] group-hover:text-ink"
						>{nav.prev.title}</span
					>
				</a>
			{:else}
				<span aria-hidden="true"></span>
			{/if}
			{#if nav.next}
				<a
					href={resolve(`/${nav.next.slug}`)}
					class="group rounded-lg border border-line-soft p-4 text-right transition-colors hover:border-line"
				>
					<span class="eyebrow flex items-center justify-end gap-1.5">
						{nav.next.n === 0 ? 'Prologue' : `Chapter ${nav.next.n}`}
						<ArrowRight size={11} />
					</span>
					<span class="mt-1.5 block font-serif text-[17px] group-hover:text-ink"
						>{nav.next.title}</span
					>
				</a>
			{:else}
				<a
					href={resolve('/epilogue')}
					class="group rounded-lg border border-line-soft p-4 text-right transition-colors hover:border-line"
				>
					<span class="eyebrow flex items-center justify-end gap-1.5">
						The end <ArrowRight size={11} />
					</span>
					<span class="mt-1.5 block font-serif text-[17px] group-hover:text-ink">Epilogue</span>
				</a>
			{/if}
		</div>
	</nav>
</article>
