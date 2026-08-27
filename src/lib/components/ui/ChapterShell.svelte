<script lang="ts">
	import type { Snippet } from 'svelte';
	import { resolve } from '$app/paths';
	import type { Pathname } from '$app/types';
	import { ArrowLeft, ArrowRight } from 'lucide-svelte';
	import References from '$lib/components/ui/References.svelte';
	import { chapterBySlug, chapters, neighbors, type Chapter } from '$lib/data/chapters';
	import { progress } from '$lib/data/progress.svelte';
	import { setChapter } from '$lib/data/plates';

	// The chapter page anatomy: ghost numeral, eyebrow, serif title, italic
	// deck — then the chapter body, then quiet prev/next cards.
	interface Props {
		slug: string;
		children: Snippet;
		/**
		 * Front matter for a page that is not in the chapter list — the
		 * epilogue. Without it the shell reads the chapter registry as usual.
		 */
		front?: { numeral: string; eyebrow: string; title: string; deck: string };
	}

	let { slug, children, front }: Props = $props();

	// Every plate below reads this to find its own numeral.
	setChapter(() => slug);

	const chapter = $derived(chapterBySlug.get(slug) as Chapter | undefined);
	const head = $derived(
		front ??
			({
				numeral: String(chapter!.n),
				eyebrow: `${chapter!.n === 0 ? 'Prologue' : `Chapter ${chapter!.n}`} · ${chapter!.kicker} · ≈${chapter!.minutes} min`,
				title: chapter!.title,
				deck: chapter!.deck
			} as const)
	);
	// One shape for both feet of the book: a chapter's neighbours, or the
	// epilogue's walk back to the last chapter and out to the beginning.
	interface Card {
		route: Pathname;
		kicker: string;
		title: string;
	}

	const card = (c: Chapter): Card => ({
		route: `/${c.slug}`,
		kicker: c.n === 0 ? 'Prologue' : `Chapter ${c.n}`,
		title: c.title
	});

	const prev = $derived.by((): Card | undefined => {
		if (!chapter) return card(chapters[chapters.length - 1]);
		const p = neighbors(slug).prev;
		return p && card(p);
	});

	const next = $derived.by((): Card | undefined => {
		if (!chapter) return { route: '/', kicker: 'Start over', title: 'The beginning' };
		const n = neighbors(slug).next;
		return n ? card(n) : { route: '/epilogue', kicker: 'The end', title: 'Epilogue' };
	});

	$effect(() => {
		progress.visit(slug);
	});
</script>

<svelte:head>
	<title>{head.title} · jaxverse</title>
	<meta name="description" content={head.deck} />
</svelte:head>

<article class="pb-10">
	<header class="rail-prose relative pt-16 pb-10 sm:pt-24">
		<span
			aria-hidden="true"
			class="pointer-events-none absolute top-6 right-2 font-serif italic select-none sm:-right-16"
			style="font-size: clamp(7rem, 18vw, 13rem); line-height: 1; color: var(--ink); opacity: 0.055; font-variation-settings: 'opsz' 72;"
		>
			{head.numeral}
		</span>
		<p class="eyebrow mb-4">{head.eyebrow}</p>
		<h1
			class="font-serif tracking-tight text-balance"
			style="font-size: clamp(2.5rem, 6vw, 3.6rem); line-height: 1.05; font-weight: 480; font-variation-settings: 'opsz' 60;"
		>
			{head.title}
		</h1>
		<p
			class="mt-5 font-serif text-[1.28rem] leading-[1.5] text-ink-2 italic"
			style="font-variation-settings: 'opsz' 20;"
		>
			{head.deck}
		</p>
	</header>

	{@render children()}

	<References {slug} />

	<nav class="rail-prose mt-20" aria-label="Chapter navigation">
		<div class="grid grid-cols-1 gap-3 border-t border-line-soft pt-6 sm:grid-cols-2">
			{#if prev}
				<a
					href={resolve(prev.route)}
					class="group rounded-lg border border-line-soft p-4 transition-colors hover:border-line"
				>
					<span class="eyebrow flex items-center gap-1.5">
						<ArrowLeft size={11} />
						{prev.kicker}
					</span>
					<span class="mt-1.5 block font-serif text-[17px] group-hover:text-ink">{prev.title}</span>
				</a>
			{:else}
				<span aria-hidden="true"></span>
			{/if}
			{#if next}
				<a
					href={resolve(next.route)}
					class="group rounded-lg border border-line-soft p-4 text-right transition-colors hover:border-line"
				>
					<span class="eyebrow flex items-center justify-end gap-1.5">
						{next.kicker}
						<ArrowRight size={11} />
					</span>
					<span class="mt-1.5 block font-serif text-[17px] group-hover:text-ink">{next.title}</span>
				</a>
			{/if}
		</div>
	</nav>
</article>
