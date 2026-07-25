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
		tone?: 'accent' | 'warm' | 'ink';
	}

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
	const toneColor = $derived(
		tone === 'warm' ? 'var(--warm)' : tone === 'ink' ? 'var(--ink-2)' : 'var(--accent)'
	);

	let uid = $props.id();
</script>

<div class="slider-root" class:opacity-50={disabled}>
	<div class="mb-1 flex items-baseline justify-between gap-3">
		<label for={uid} class="eyebrow cursor-pointer select-none">{label}</label>
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
		style="--p: {pct}%; --tone: {toneColor};"
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
		box-shadow: 0 0 0 4px color-mix(in srgb, var(--tone) 25%, transparent);
	}
	input[type='range']:focus-visible::-moz-range-thumb {
		box-shadow: 0 0 0 4px color-mix(in srgb, var(--tone) 25%, transparent);
	}
</style>
