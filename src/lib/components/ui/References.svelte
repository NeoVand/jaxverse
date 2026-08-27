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
		<ol class="sources">
			{#each list as id, i (id)}
				{@const paper = papers[id]}
				<li class="source" id={refAnchor(id)}>
					<span class="source-n num">{i + 1}</span>
					<span class="source-body">
						<a class="source-title" href={paper.url} target="_blank" rel="noopener noreferrer">
							{paper.title}<ArrowUpRight class="source-out" size={13} />
						</a>
						<span class="source-meta">
							{paper.authors} · {paper.year} · {paper.where} ·
							<span class="source-where">{sourceLabel(paper.url)}</span>
						</span>
						{#if paper.note}
							<span class="source-note">{paper.note}</span>
						{/if}
					</span>
				</li>
			{/each}
		</ol>
	</section>
{/if}
