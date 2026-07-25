<script lang="ts">
	// KaTeX, rendered at build time too (renderToString is pure, so prerendered
	// pages ship their math as HTML — no client-side flash).
	import katex from 'katex';
	import 'katex/dist/katex.min.css';

	interface Props {
		tex: string;
		display?: boolean;
	}

	let { tex, display = false }: Props = $props();

	// \htmlClass is allowed so equations can color-code their terms with the
	// site's own tokens: eq-a (accent), eq-w (warm), eq-g (good), eq-m (muted).
	const html = $derived(
		katex.renderToString(tex, {
			displayMode: display,
			throwOnError: false,
			strict: false,
			trust: (ctx) => ctx.command === '\\htmlClass'
		})
	);
</script>

{#if display}
	<!-- eslint-disable-next-line svelte/no-at-html-tags -- KaTeX output of author-written TeX -->
	<div class="math-display my-5 overflow-x-auto py-1">{@html html}</div>
{:else}
	<!-- eslint-disable-next-line svelte/no-at-html-tags -- KaTeX output of author-written TeX -->
	<span class="math-inline">{@html html}</span>
{/if}

<style>
	.math-display :global(.katex-display) {
		margin: 0;
	}
	.math-display :global(.katex) {
		font-size: 1.16em;
		color: var(--ink);
	}
	.math-inline :global(.katex) {
		font-size: 1.02em;
		color: var(--ink);
	}
</style>
