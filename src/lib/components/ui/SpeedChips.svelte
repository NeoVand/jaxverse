<script lang="ts">
	// The house speed control for training demos: a multiplier the demo applies
	// to its own pacing (steps per chunk, pacing delays, episodes per frame).
	// 'max' (value 0) means: as fast as the machine allows, no display pacing.
	interface Props {
		value: number;
		/** Multipliers offered; 0 = max. */
		options?: number[];
	}

	let { value = $bindable(), options = [1, 3, 0] }: Props = $props();

	const label = (v: number) => (v === 0 ? 'max' : `×${v}`);
</script>

<span class="flex items-center gap-1" role="group" aria-label="Speed">
	<span class="eyebrow mr-1">speed</span>
	{#each options as v (v)}
		<button class="chip" class:chip-on={value === v} onclick={() => (value = v)}>
			{label(v)}
		</button>
	{/each}
</span>

<style>
	.chip {
		font-family: var(--font-sans);
		font-size: 11px;
		font-weight: 500;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		padding: 3px 9px;
		border-radius: 5px;
		border: 1px solid var(--line);
		color: var(--ink-2);
		background: var(--surface);
		transition: all 100ms ease;
	}
	.chip:hover {
		border-color: var(--ink-3);
		color: var(--ink);
	}
	.chip-on {
		background: var(--ink);
		border-color: var(--ink);
		color: var(--paper);
	}
</style>
