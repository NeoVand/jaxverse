<script lang="ts">
	import { resolve } from '$app/paths';
	import type { Pathname } from '$app/types';
	import { chapterBySlug } from '$lib/data/chapters';

	// "Chapter 6", written by the registry rather than by hand, and clickable.
	// Prose never spells a chapter numeral, so inserting a chapter renumbers
	// the book and every sentence that points into it, at once, correctly —
	// the same contract <PlateRef> keeps for figures.
	interface Props {
		slug: string;
		/** Mid-sentence use wants "chapter 6", not "Chapter 6". */
		lower?: boolean;
		/** Append the chapter's title: "Chapter 6, Learning from Reward". */
		titled?: boolean;
	}

	let { slug, lower = false, titled = false }: Props = $props();

	const chapter = $derived(chapterBySlug.get(slug));
	// The prologue has no numeral to speak; it has a name.
	const label = $derived.by(() => {
		if (!chapter) return 'that chapter';
		const head = chapter.n === 0 ? 'the Prologue' : `Chapter ${chapter.n}`;
		return titled ? `${head}, ${chapter.title}` : head;
	});
</script>

<a href={resolve(`/${slug}` as Pathname)}>{lower ? label.replace(/^Chapter/, 'chapter') : label}</a>
