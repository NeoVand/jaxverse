<script lang="ts">
	import { ArrowUpRight } from 'lucide-svelte';
	import { citationOrder, papers, refAnchor, sourceLabel } from '$lib/data/papers';

	// The foot of a chapter: the papers it stood on, numbered as the prose
	// numbered them, each one a link to somewhere the reader can actually read
	// it. The chapter shell renders this itself, so no page has to remember to.
	interface Props {
		slug: string;
	}

	let { slug }: Props = $props();

	const list = $derived(citationOrder[slug] ?? []);
</script>

{#if list.length}
	<section class="rail-prose mt-20" aria-labelledby="sources-{slug}">
		<h2 class="eyebrow mb-5 border-t border-line-soft pt-6" id="sources-{slug}">Sources</h2>
		<ol class="refs">
			{#each list as id, i (id)}
				{@const paper = papers[id]}
				<li class="ref" id={refAnchor(id)}>
					<span class="ref-n num">{i + 1}</span>
					<span class="ref-body">
						<a class="ref-title" href={paper.url} target="_blank" rel="noopener noreferrer">
							{paper.title}<ArrowUpRight class="ref-out" size={13} />
						</a>
						<span class="ref-meta">
							{paper.authors} · {paper.year} · {paper.where} ·
							<span class="ref-src">{sourceLabel(paper.url)}</span>
						</span>
						{#if paper.note}
							<span class="ref-note">{paper.note}</span>
						{/if}
					</span>
				</li>
			{/each}
		</ol>
	</section>
{/if}
