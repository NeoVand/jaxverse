<script lang="ts">
	// The hidden book inside the book. Every plate that trains something carries
	// one of these directly beneath it: closed, it is a single quiet line; open,
	// it is the machinery the plate just ran — the jax-js lesson, the real code
	// from this repository, and (on a chapter's last block) a lab you can
	// download and run with nothing but npm.
	import { slide } from 'svelte/transition';
	import { ChevronRight, Download } from 'lucide-svelte';
	import { base } from '$app/paths';
	import { highlight } from '$lib/hood/highlight';
	import { hood } from '$lib/hood';

	interface Props {
		slug: string;
		block: string;
	}
	let { slug, block }: Props = $props();

	const entry = $derived(hood[slug]?.blocks.find((b) => b.id === block));
	let open = $state(false);
	let tab = $state<'ml' | 'ui'>('ml');

	const hasUi = $derived(!!entry && entry.ui.length > 0);
	const sections = $derived(entry ? (tab === 'ml' && entry.ml.length ? entry.ml : entry.ui) : []);
</script>

{#if entry}
	<!-- Self-centering at the reading-column width, so closed it reads as one
	     quiet line of the text, not a stray plate hugging the left margin. -->
	<div class="mx-auto max-w-2xl px-5">
		<section class="hood my-8 overflow-hidden rounded-lg" class:hood-open={open}>
			<button
				class="hood-toggle flex w-full flex-wrap items-center gap-x-3 gap-y-1 px-4 py-3 text-left"
				aria-expanded={open}
				onclick={() => (open = !open)}
			>
				<ChevronRight
					size={14}
					aria-hidden="true"
					class="shrink-0 transition-transform duration-200 {open ? 'rotate-90' : ''}"
					style="color: var(--accent);"
				/>
				<span class="eyebrow" style="color: var(--accent);">Under the hood</span>
				<span class="font-serif text-[15px] italic" style="font-variation-settings: 'opsz' 14;">
					{entry.lesson}
				</span>
				{#if !open}
					<span class="num ml-auto text-[10.5px] text-ink-3">the code this plate runs</span>
				{/if}
			</button>

			{#if open}
				<div transition:slide={{ duration: 220 }}>
					{#if entry.lede}
						<div class="border-t border-line-soft px-4 pt-4 pb-2 sm:px-6">
							<p class="hood-lede font-serif text-[15px] leading-relaxed text-ink-2">
								<!-- eslint-disable-next-line svelte/no-at-html-tags -- our own content module -->
								{@html entry.lede}
							</p>
						</div>
					{/if}

					{#if hasUi}
						<div
							class="flex items-center gap-1.5 px-4 pt-2 pb-3 sm:px-6 {entry.lede
								? ''
								: 'border-t border-line-soft pt-4'}"
							role="tablist"
						>
							<button
								class="chip"
								class:chip-on={tab === 'ml'}
								role="tab"
								aria-selected={tab === 'ml'}
								onclick={() => (tab = 'ml')}
							>
								the machine learning
							</button>
							<button
								class="chip"
								class:chip-on={tab === 'ui'}
								role="tab"
								aria-selected={tab === 'ui'}
								onclick={() => (tab = 'ui')}
							>
								the stagecraft
							</button>
						</div>
					{:else if !entry.lede}
						<div class="border-t border-line-soft pt-4"></div>
					{/if}

					<div class="flex flex-col gap-6 px-4 pt-1 pb-5 sm:px-6">
						{#each sections as s (s.title)}
							<div>
								<h3
									class="mb-1.5 font-serif text-[17px] tracking-tight"
									style="font-weight: 540; font-variation-settings: 'opsz' 18;"
								>
									{s.title}
								</h3>
								<p class="hood-body font-serif text-[14.5px] leading-relaxed text-ink-2">
									<!-- eslint-disable-next-line svelte/no-at-html-tags -- our own content module -->
									{@html s.body}
								</p>
								{#if s.code}
									<div class="code mt-3 overflow-hidden rounded-md">
										{#if s.code.file}
											<div
												class="num border-b border-line-soft px-3 py-1.5 text-[10px] tracking-wide text-ink-3"
											>
												{s.code.file}
											</div>
										{/if}
										<!-- eslint-disable svelte/no-at-html-tags -- the highlighter escapes its input first -->
										<pre class="num overflow-x-auto px-3 py-2.5 text-[11.5px] leading-relaxed"><code
												>{@html highlight(s.code.code)}</code
											></pre>
										<!-- eslint-enable svelte/no-at-html-tags -->
									</div>
								{/if}
							</div>
						{/each}
					</div>

					<div
						class="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-line-soft px-4 py-3 sm:px-6"
					>
						{#if entry.lab}
							<!-- eslint-disable svelte/no-navigation-without-resolve -- static asset, not a route -->
							<a
								class="btn-solid inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[11px] font-medium tracking-[0.1em] uppercase"
								href="{base}/labs/{entry.lab.file}"
								download
							>
								<Download size={12} aria-hidden="true" /> Download the lab
							</a>
							<!-- eslint-enable svelte/no-navigation-without-resolve -->
							<span class="font-serif text-[13px] text-ink-3 italic">
								{entry.lab.note} — unzip, <span class="num not-italic">npm install</span>,
								<span class="num not-italic">npm run dev</span>.
							</span>
						{/if}
						<a
							class="num ml-auto text-[10.5px] text-ink-3 underline decoration-dotted underline-offset-2 hover:text-ink"
							href="https://github.com/NeoVand/jaxverse"
							rel="external"
						>
							the whole book on GitHub
						</a>
					</div>
				</div>
			{/if}
		</section>
	</div>
{/if}

<style>
	.hood {
		border: 1px dashed var(--line);
		background: color-mix(in srgb, var(--surface-2) 45%, var(--surface));
		transition: border-color 150ms ease;
	}
	.hood-open {
		border-style: solid;
		background: var(--surface);
	}
	.hood-toggle:hover {
		background: color-mix(in srgb, var(--surface-2) 60%, transparent);
	}

	.code {
		border: 1px solid var(--line-soft);
		background: color-mix(in srgb, var(--surface-2) 55%, var(--surface));
	}
	.code pre {
		font-family: var(--font-mono);
		tab-size: 2;
	}

	/* the five voices of the highlighter, in the site's own ink */
	.code :global(.hl-kw) {
		color: var(--accent);
	}
	.code :global(.hl-str) {
		color: var(--good);
	}
	.code :global(.hl-com) {
		color: var(--ink-3);
		font-style: italic;
	}
	.code :global(.hl-num) {
		color: var(--warm);
	}
	.code :global(.hl-fn) {
		color: var(--ink);
	}

	.hood-lede :global(em),
	.hood-body :global(em) {
		font-style: italic;
	}
	.hood-lede :global(code),
	.hood-body :global(code) {
		font-family: var(--font-mono);
		font-size: 0.86em;
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0.05em 0.3em;
	}
</style>
