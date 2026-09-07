<script lang="ts">
	// A grid of model output, drawn crisp and composited onto the plate.
	// Shared by both diffusion chapters — nine plates draw pictures this way,
	// and the theme-repaint dance should only be written once.
	import { paintTiles } from '$lib/viz/fashion-paint';
	import { readTokens, themePulse, watchTheme } from '$lib/viz/tokens.svelte';

	interface Props {
		/** [count · res · res], ink coverage in [-1, 1]. */
		pixels: Float32Array | null;
		count: number;
		columns: number;
		res?: number;
		gap?: number;
		/** Reveal only the first n tiles, for progressive fills. */
		limit?: number;
		/** Extra classes for the canvas; height usually comes from here. */
		class?: string;
		label: string;
	}
	let {
		pixels,
		count,
		columns,
		res = 28,
		gap = 3,
		limit,
		class: klass = '',
		label
	}: Props = $props();

	let canvas: HTMLCanvasElement | undefined = $state();

	$effect(() => watchTheme());

	$effect(() => {
		void themePulse.tick;
		void pixels;
		void limit;
		void count;
		void columns;
		if (!canvas) return;
		const tk = readTokens(canvas);
		if (!pixels) {
			// empty slots, so the plate has its final height before the first draw
			const blank = new Float32Array(count * res * res).fill(-1);
			paintTiles(canvas, blank, count, res, { columns, background: tk.band, ink: tk.ink, gap });
			return;
		}
		paintTiles(canvas, pixels, count, res, {
			columns,
			background: tk.band,
			ink: tk.ink,
			gap,
			limit
		});
	});
</script>

<canvas bind:this={canvas} class="block w-full {klass}" aria-label={label}></canvas>
