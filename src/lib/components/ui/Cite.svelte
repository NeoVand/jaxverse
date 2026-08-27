<script lang="ts">
	import { citationNumber, papers, refAnchor, sourceLabel, type PaperId } from '$lib/data/papers';
	import { getChapter } from '$lib/data/plates';

	// A citation is a small raised numeral that goes straight to the paper —
	// one click, no detour through a bibliography. The numeral itself comes
	// from the chapter's list in $lib/data/papers, never from a number typed
	// into a sentence, so inserting a source renumbers the chapter correctly.
	//
	// Where the reader has a pointer, hovering opens the full record instead of
	// making them leave the page to find out what they are about to open. On a
	// touch screen there is no hover to open it with, so the tap just follows
	// the link, and the chapter's own reference list at the foot of the page
	// carries the same record in full.
	interface Props {
		id: PaperId;
	}

	let { id }: Props = $props();

	const chapter = getChapter();
	const paper = $derived(papers[id]);
	const n = $derived(citationNumber(chapter(), id));
</script>

<span class="cite">
	<a
		class="cite-mark"
		href={paper.url}
		target="_blank"
		rel="external noopener noreferrer"
		aria-describedby={refAnchor(id)}
		title="{paper.authors} ({paper.year}), {paper.title}"
	>
		{n ?? '·'}
	</a>
	<span class="cite-card" aria-hidden="true">
		<span class="cite-card-authors">{paper.authors} · {paper.year}</span>
		<span class="cite-card-title">{paper.title}</span>
		<span class="cite-card-where">{paper.where} · {sourceLabel(paper.url)}</span>
	</span>
</span>
