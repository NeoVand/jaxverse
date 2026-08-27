<script lang="ts">
	// The house range input: label left, mono value right, hairline track with
	// an ink fill. No native styling survives.
	interface Props {
		label: string;
		value: number;
		min: number;
		max: number;
		step?: number;
		/** Format the value readout; defaults to the raw number. */
		format?: (v: number) => string;
		disabled?: boolean;
		/**
		 * Which voice of the color constitution the slider speaks in. The
		 * semantic names match the .eq-* classes, so a slider can be painted the
		 * same color as the symbol it moves.
		 */
		tone?: keyof typeof TONES;
	}

	const TONES = {
		accent: 'var(--accent)',
		warm: 'var(--warm)',
		teal: 'var(--cat-2)',
		ink: 'var(--ink-2)',
		model: 'var(--accent)',
		'model-2': 'var(--cat-8)',
		'model-3': 'var(--cat-6)',
		world: 'var(--warm)',
		op: 'var(--cat-2)',
		knob: 'var(--cat-1)',
		out: 'var(--good)'
	} as const;

	let {
		label,
		value = $bindable(),
		min,
		max,
		step = 1,
		format = (v) => String(v),
		disabled = false,
		tone = 'accent'
	}: Props = $props();

	const pct = $derived(((value - min) / (max - min)) * 100);
	const toneColor = $derived(TONES[tone]);

	// The label is set in the eyebrow voice, which is uppercase — and a slider
	// is very often named for a Greek letter. Uppercasing one turns ρ into Ρ
	// and β into Β, glyphs a reader cannot tell from Latin P and B, so the
	// control ends up labelled with a different symbol than the equation it
	// moves. Split the label and let the symbols keep their own case.
	const parts = $derived(splitSymbols(label));

	function splitSymbols(text: string): Array<{ text: string; sym: boolean }> {
		const runs: Array<{ text: string; sym: boolean }> = [];
		for (const ch of text) {
			const sym = /[\u0370-\u03ff\u1f00-\u1fff]/.test(ch);
			const last = runs[runs.length - 1];
			if (last && last.sym === sym) last.text += ch;
			else runs.push({ text: ch, sym });
		}
		return runs;
	}

	let uid = $props.id();
</script>

<div class="slider-root" class:opacity-50={disabled}>
	<div class="mb-1 flex items-baseline justify-between gap-3">
		<label for={uid} class="eyebrow cursor-pointer select-none"
			>{#each parts as part, i (i)}{#if part.sym}<span class="sym">{part.text}</span
					>{:else}{part.text}{/if}{/each}</label
		>
		<span class="num text-[12px] text-ink">{format(value)}</span>
	</div>
	<input
		id={uid}
		type="range"
		bind:value
		{min}
		{max}
		{step}
		{disabled}
		style="--p: {pct}%; --tone: {toneColor}; --focus-tone: {toneColor};"
	/>
</div>

<style>
	input[type='range'] {
		-webkit-appearance: none;
		appearance: none;
		width: 100%;
		height: 16px;
		background: transparent;
		cursor: pointer;
		padding: 0;
		border: none;
	}
	input[type='range']:disabled {
		cursor: default;
	}
	input[type='range']::-webkit-slider-runnable-track {
		height: 2px;
		border-radius: 1px;
		background: linear-gradient(to right, var(--tone) var(--p), var(--line) var(--p));
	}
	input[type='range']::-moz-range-track {
		height: 2px;
		border-radius: 1px;
		background: linear-gradient(to right, var(--tone) var(--p), var(--line) var(--p));
	}
	input[type='range']::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		margin-top: -5px;
		height: 12px;
		width: 12px;
		border-radius: 50%;
		background: var(--paper);
		border: 2px solid var(--tone);
		transition: transform 100ms ease;
	}
	input[type='range']::-moz-range-thumb {
		height: 12px;
		width: 12px;
		border-radius: 50%;
		background: var(--paper);
		border: 2px solid var(--tone);
		transition: transform 100ms ease;
	}
	input[type='range']:hover::-webkit-slider-thumb {
		transform: scale(1.18);
	}
	input[type='range']:focus-visible {
		outline: none;
	}
	input[type='range']:focus-visible::-webkit-slider-thumb {
		box-shadow: var(--focus-ring);
	}
	input[type='range']:focus-visible::-moz-range-thumb {
		box-shadow: var(--focus-ring);
	}
</style>
