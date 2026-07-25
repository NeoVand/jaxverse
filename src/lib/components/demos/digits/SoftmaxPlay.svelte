<script lang="ts">
	// The softmax, by hand: six scores the reader controls and the belief they
	// become. Pure arithmetic — no engine, no data, nothing to load.
	import Slider from '$lib/components/ui/Slider.svelte';
	import { softmax } from './common';

	let logits = $state([2, 0.5, 1, -1, 0, -0.5]);
	let logT = $state(0); // temperature slider on a log scale, T = 10^logT
	const T = $derived(10 ** logT);
	const probs = $derived(
		softmax(
			logits.map((z) => z / T),
			0,
			logits.length
		)
	);

	const SUBS = ['z₁', 'z₂', 'z₃', 'z₄', 'z₅', 'z₆'];
</script>

<div class="grid grid-cols-1 gap-x-10 gap-y-5 p-4 sm:grid-cols-[1fr_1.1fr] sm:p-5">
	<!-- the scores -->
	<div>
		<span class="eyebrow">the scores · logits z</span>
		<div class="mt-2 flex flex-col gap-1">
			{#each SUBS as label, i (label)}
				<Slider
					{label}
					bind:value={logits[i]}
					min={-4}
					max={4}
					step={0.1}
					format={(v) => v.toFixed(1)}
				/>
			{/each}
		</div>
	</div>

	<!-- the belief -->
	<div class="flex flex-col gap-3">
		<span class="eyebrow">the belief · softmax(z / T)</span>
		<div class="flex items-end gap-2">
			{#each probs as p, i (i)}
				<div class="flex min-w-0 flex-1 flex-col items-center gap-1">
					<div class="relative h-32 w-full">
						<span
							class="num absolute inset-x-0 text-center text-[10px] text-ink-2"
							style="bottom: calc({Math.min(88, Math.max(1.5, p * 100))}% + 2px);"
						>
							{(p * 100).toFixed(0)}%
						</span>
						<div
							class="absolute inset-x-0 bottom-0 rounded-t-[3px] transition-[height] duration-150"
							style="height: {Math.max(1.5, p * 100)}%; background: var(--cat-{i}); opacity: 0.85;"
						></div>
					</div>
					<span class="num text-[11px] text-ink-3">{SUBS[i]}</span>
				</div>
			{/each}
		</div>
		<Slider
			label="temperature T"
			bind:value={logT}
			min={-0.6}
			max={0.6}
			step={0.02}
			format={() => T.toFixed(2)}
			tone="warm"
		/>
		<p class="m-0 text-[11px] leading-snug text-ink-3">
			<span class="num">Σp = 1.00</span> always — softmax spends one unit of belief, however the scores
			move. Lower T sharpens the outcome; raise it and the belief flattens toward indifference.
		</p>
	</div>
</div>
