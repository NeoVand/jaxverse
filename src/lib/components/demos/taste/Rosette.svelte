<script lang="ts">
	// One ornament, drawn from one gene. Pure presentation — every number it
	// needs comes out of rosette.ts, and it holds no state of its own, so the
	// same component serves the comparator, the specimen grid on the taste map,
	// and the before/after pairs on the over-optimization plate.
	//
	// One performance note worth keeping, because it is not obvious. Every
	// colour here is a `color-mix` over CSS variables, so the browser has to
	// resolve it during style recalculation. Writing those expressions onto the
	// forty-odd paths of a rosette costs forty resolutions per ornament, and a
	// page carrying sixty ornaments spends most of a second on colour alone.
	// So the mixes are resolved once, into six custom properties on the <svg>,
	// and every path just names one. Same picture, one twentieth of the work.
	import { inkOf, rosette, R_MAX, VIEW } from './rosette';
	import type { Gene } from '$lib/optim-rl/preference';

	interface Props {
		gene: Gene;
		size?: number;
		/** Alt text — the plates supply something specific where it matters. */
		label?: string;
		/** The medallion rule is the specimen sheet's furniture; tiny thumbnails
		 * drop it, because at 24px it is the only thing you would see. */
		frame?: boolean;
	}
	let { gene, size = 160, label, frame = true }: Props = $props();

	const shape = $derived(rosette(gene));
	const innerAlpha = $derived(shape.inner?.alpha ?? 1);
	const vars = $derived(
		[
			`--rz:${inkOf(shape.mix)}`,
			`--rz-fill:${inkOf(shape.mix, shape.fill)}`,
			`--rz-in:${inkOf(shape.mix, innerAlpha)}`,
			`--rz-in-fill:${inkOf(shape.mix, shape.fill * innerAlpha)}`,
			`--rz-ring:${inkOf(shape.mix, 0.55)}`,
			`--rz-frame:${inkOf(shape.mix, 0.3)}`
		].join(';')
	);
</script>

<svg
	width={size}
	height={size}
	viewBox="0 0 {VIEW} {VIEW}"
	style={vars}
	role={label ? 'img' : 'presentation'}
	aria-label={label}
	aria-hidden={label ? undefined : 'true'}
>
	{#if frame}
		<circle
			cx={VIEW / 2}
			cy={VIEW / 2}
			r={R_MAX}
			fill="none"
			stroke="var(--rz-frame)"
			stroke-width="0.6"
		/>
	{/if}
	{#each shape.rings as r, i (i)}
		<circle
			cx={VIEW / 2}
			cy={VIEW / 2}
			{r}
			fill="none"
			stroke="var(--rz-ring)"
			stroke-width={Math.min(1.1, shape.stroke * 0.4)}
		/>
	{/each}
	{#if shape.inner}
		{#each shape.inner.petals as d, i (i)}
			<path
				{d}
				fill="var(--rz-in-fill)"
				stroke="var(--rz-in)"
				stroke-width={shape.stroke * 0.72}
				stroke-linejoin="round"
			/>
		{/each}
	{/if}
	{#each shape.outer.petals as d, i (i)}
		<path
			{d}
			fill="var(--rz-fill)"
			stroke="var(--rz)"
			stroke-width={shape.stroke}
			stroke-linejoin="round"
		/>
	{/each}
	<circle cx={VIEW / 2} cy={VIEW / 2} r={shape.hub} fill="var(--rz)" />
</svg>
