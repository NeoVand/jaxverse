<script lang="ts">
	// The neuron as a circuit: the signal flows left to right along ultramarine
	// edges — input x and the constant +1 into the weighted sum Σ, through the
	// activation σ, out as a. The parameters (vermilion labels) annotate live:
	// the w edge thickens with |w|, and both labels show their current value.
	interface Props {
		w: number;
		b: number;
	}

	let { w, b }: Props = $props();

	const wWidth = $derived(1 + (1.6 * Math.abs(w)) / 6);
	const minus = (s: string) => s.replace('-', '−');
</script>

<svg
	viewBox="0 0 300 170"
	class="block w-full max-w-[300px]"
	role="img"
	aria-label="Diagram of one neuron: the input x is scaled by the weight w, the bias b is added at the sum node, the result passes through the activation sigma and comes out as a"
>
	<!-- edges first; node fills cover the joins -->
	<line x1="46" y1="52" x2="150" y2="87" stroke="var(--accent)" stroke-width={wWidth} />
	<line x1="46" y1="122" x2="150" y2="87" stroke="var(--accent)" stroke-width="1" />
	<line x1="150" y1="87" x2="228" y2="87" stroke="var(--accent)" stroke-width="1.4" />
	<line x1="228" y1="87" x2="277" y2="87" stroke="var(--accent)" stroke-width="1.4" />
	<polygon points="283,87 275,83.2 275,90.8" fill="var(--accent)" />

	<!-- input x -->
	<circle cx="46" cy="52" r="16" fill="var(--surface)" stroke="var(--line)" />
	<text
		x="46"
		y="57"
		text-anchor="middle"
		font-size="15"
		fill="var(--ink)"
		style="font-family: var(--font-serif); font-style: italic;">x</text
	>

	<!-- the bias input, always 1 -->
	<circle cx="46" cy="122" r="12" fill="var(--surface)" stroke="var(--line)" />
	<text x="46" y="126" text-anchor="middle" class="num" font-size="10" fill="var(--ink-3)">+1</text>

	<!-- the weighted sum -->
	<circle cx="150" cy="87" r="17" fill="var(--surface)" stroke="var(--line)" />
	<text
		x="150"
		y="93"
		text-anchor="middle"
		font-size="16"
		fill="var(--ink)"
		style="font-family: var(--font-serif);">Σ</text
	>

	<!-- the activation: a little tanh glyph -->
	<circle cx="228" cy="87" r="17" fill="var(--surface)" stroke="var(--line)" />
	<path
		d="M217 94C226 94 230 80 239 80"
		fill="none"
		stroke="var(--accent)"
		stroke-width="1.6"
		stroke-linecap="round"
	/>
	<text
		x="228"
		y="118"
		text-anchor="middle"
		font-size="11"
		fill="var(--ink-3)"
		style="font-family: var(--font-serif); font-style: italic;">σ</text
	>

	<!-- output -->
	<text
		x="290"
		y="92"
		font-size="15"
		fill="var(--ink)"
		style="font-family: var(--font-serif); font-style: italic;">a</text
	>

	<!-- parameter labels, live -->
	<text x="92" y="56" text-anchor="middle">
		<tspan
			font-size="13.5"
			fill="var(--warm)"
			style="font-family: var(--font-serif); font-style: italic;">w</tspan
		>
		<tspan dx="3" class="num" font-size="10" fill="var(--warm)">{minus(w.toFixed(1))}</tspan>
	</text>
	<text x="92" y="126" text-anchor="middle">
		<tspan
			font-size="13.5"
			fill="var(--warm)"
			style="font-family: var(--font-serif); font-style: italic;">b</tspan
		>
		<tspan dx="3" class="num" font-size="10" fill="var(--warm)">{minus(b.toFixed(2))}</tspan>
	</text>
</svg>
