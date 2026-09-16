<script lang="ts">
	import type { WorldEvaluation } from '$lib/world/engine';
	import Sensor from './Sensor.svelte';

	interface Props {
		evaluation: WorldEvaluation | null;
		initial: WorldEvaluation | null;
		training: boolean;
		interval?: number;
	}
	let { evaluation, initial, training, interval = 0.24 }: Props = $props();
	let selectedId = $state<number | null>(null);
	let view = $state<'before' | 'now' | 'copy'>('now');
	const probe = $derived(evaluation?.futureMatching ?? initial?.futureMatching ?? null);
	const before = $derived(initial?.futureMatching ?? null);
	const examples = $derived(probe?.examples ?? []);
	// Keep the chosen held-out moment fixed when a new checkpoint arrives.
	const example = $derived(examples.find((item) => item.id === selectedId) ?? examples[0] ?? null);
	const beforeExample = $derived(before?.examples.find((item) => item.id === example?.id) ?? null);
	const selected = $derived(
		view === 'before'
			? (beforeExample?.selectedIndex ?? null)
			: view === 'copy'
				? (example?.persistenceIndex ?? null)
				: (example?.selectedIndex ?? null)
	);
	const measured = $derived(view === 'before' ? beforeExample !== null : example !== null);
	const choiceLabel = $derived(view === 'before' ? 'Before' : view === 'copy' ? 'Copy' : 'Now');
	const choices = $derived([
		{
			id: 'before' as const,
			label: 'Before learning',
			correct: before?.correct,
			total: before?.total,
			ties: before?.ties
		},
		{
			id: 'now' as const,
			label: 'Now',
			correct: probe?.correct,
			total: probe?.total,
			ties: probe?.ties
		},
		{
			id: 'copy' as const,
			label: 'Copy the present',
			correct: probe?.persistenceCorrect,
			total: probe?.total,
			ties: probe?.persistenceTies
		}
	]);
	const letters = ['A', 'B', 'C', 'D', 'E', 'F'];
	function torque(value: number): string {
		return `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(2)}`;
	}
</script>

<section class="evidence" aria-label="Visual evidence of learning">
	<div class="evidence-controls">
		<div class="seg" role="group" aria-label="Compare predictions before and after learning">
			{#each choices as choice (choice.id)}
				<button
					type="button"
					class={{ on: view === choice.id }}
					aria-pressed={view === choice.id}
					onclick={() => (view = choice.id)}>{choice.label}</button
				>
			{/each}
		</div>
		{#if example}
			<div class="case-nav">
				<span class="eyebrow">Moment</span>
				<div class="cases" role="group" aria-label="Held-out examples">
					{#each examples as item, index (item.id)}
						<button
							type="button"
							class={['case', { active: example.id === item.id }]}
							aria-label={`Example ${index + 1}`}
							aria-pressed={example.id === item.id}
							onclick={() => (selectedId = item.id)}>{index + 1}</button
						>
					{/each}
				</div>
			</div>
		{/if}
	</div>

	<p class="question">Two pictures, their motor commands. What comes next?</p>
	{#if example}
		<div class="observations">
			<figure class="observation">
				<div class="sensor">
					<Sensor pixels={example.previous} size={Math.sqrt(example.previous.length)} />
				</div>
				<figcaption>Previous</figcaption>
			</figure>
			<div
				class="action"
				aria-label={`Previous torque: base ${torque(example.previousAction[0])}, elbow ${torque(example.previousAction[1])}`}
			>
				<span class="action-label">Torques</span>
				<span class="torques"
					><span>{torque(example.previousAction[0])}</span><span
						>{torque(example.previousAction[1])}</span
					></span
				>
				<span class="action-arrow" aria-hidden="true">⟶</span>
			</div>
			<figure class="observation">
				<div class="sensor">
					<Sensor pixels={example.current} size={Math.sqrt(example.current.length)} />
				</div>
				<figcaption>Present</figcaption>
			</figure>
			<div
				class="action"
				aria-label={`Next torque: base ${torque(example.action[0])}, elbow ${torque(example.action[1])}`}
			>
				<span class="action-label">Torques</span>
				<span class="torques"
					><span>{torque(example.action[0])}</span><span>{torque(example.action[1])}</span></span
				>
				<span class="action-arrow" aria-hidden="true">⟶</span>
			</div>
			<div class="next-moment">
				<span aria-hidden="true">?</span><span>+{interval.toFixed(2)} s</span>
			</div>
		</div>
		<p class="action-key">Turning forces at the base and elbow, shown from top to bottom.</p>

		<div class="answer-heading">
			<span
				>{view === 'copy'
					? 'Nearest picture to the present embedding'
					: 'Nearest picture to the predicted embedding'}</span
			>
			<span class="actual-key"><span aria-hidden="true">✓</span> Actual outcome</span>
		</div>
		<div
			class="candidates"
			aria-label={`${example.candidates.length} recorded candidate observations`}
		>
			{#each example.candidates as pixels, index (letters[index])}
				<figure
					class={[
						'candidate',
						{
							chosen: selected === index,
							actual: example.correctIndex === index,
							before: view === 'before',
							copy: view === 'copy'
						}
					]}
				>
					<div class="candidate-top">
						<span>{letters[index]}</span>{#if selected === index}<span
								class="choice-symbol"
								aria-label={`${choiceLabel} selection`}>● {choiceLabel}</span
							>{/if}
					</div>
					<div class="sensor"><Sensor {pixels} size={Math.sqrt(pixels.length)} /></div>
					<figcaption>
						<span class="actual-caption"
							>{#if example.correctIndex === index}<span aria-hidden="true">✓</span> Actual outcome{:else}<span
									aria-hidden="true">&nbsp;</span
								>{/if}</span
						>
					</figcaption>
				</figure>
			{/each}
		</div>
		<p class="result" aria-live="polite">
			{#if !measured}The initial measurement is being prepared.
			{:else if selected === null}Several pictures tie. This checkpoint has no unique nearest match.
			{:else if selected === example.correctIndex}<span class="match">A match.</span>
				{choiceLabel === 'Copy'
					? 'Copying the present'
					: view === 'before'
						? 'The untrained model'
						: 'The current model'} selects the recorded next frame.
			{:else}<span class="miss">A different picture.</span>
				{choiceLabel} selects {letters[selected]}; the recorded next frame is {letters[
					example.correctIndex
				]}.
			{/if}
		</p>
	{:else}
		<div class="empty-evidence">
			<span class="empty-symbol" aria-hidden="true">? → ?</span>
			<p>Two observations, a torque, and six possible next pictures.</p>
			<span
				>The first measurement will show the untrained model’s choices. These same moments stay here
				as it learns.</span
			>
		</div>
	{/if}

	<div class="matching-results">
		<div class="score-heading">
			<span class="eyebrow">Correct on {probe?.total ?? 64} unseen moments</span>
			<span class="checkpoint"
				>{#if evaluation}Measured at step {evaluation.step.toLocaleString()}{:else}Before training{/if}{#if training}
					· updating as it learns{/if}</span
			>
		</div>
		<dl class="scores">
			{#each choices as choice (choice.id)}
				<div>
					<dt>{choice.label}</dt>
					<dd class="score-value">{choice.correct ?? '—'}<span> / {choice.total ?? '—'}</span></dd>
				</div>
			{/each}
		</dl>
	</div>
	<p class="method-note">
		{#if view === 'copy'}Copy the present assumes nothing changes: it matches the current embedding
			to the six pictures, without using the predictor or actions.
		{:else}The model predicts a vector. Its marked picture is the closest match in embedding space;
			the green check identifies what actually happened.{/if}
		{probe?.candidatesPerExample ?? 6} recorded pictures per moment; ties count as misses.
	</p>
</section>

<style>
	.evidence {
		max-width: 852px;
		margin: 18px auto 20px;
		padding-inline: 16px;
	}
	.evidence-controls {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 14px;
	}
	.question {
		margin: 16px 0;
		font: 400 18px/1.45 var(--font-serif);
		color: var(--ink-2);
		text-align: center;
	}
	.checkpoint {
		font: 10px/1.5 var(--font-mono);
		color: var(--ink-3);
	}
	.case-nav {
		display: flex;
		align-items: center;
		justify-content: center;
		flex-wrap: wrap;
		gap: 10px;
	}
	.matching-results {
		margin-top: 10px;
		padding-top: 10px;
		border-top: 1px solid var(--line-soft);
	}
	.score-heading {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 5px 16px;
	}
	.scores {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 28px;
		margin: 10px 0 0;
	}
	.scores > div {
		display: flex;
		align-items: baseline;
		gap: 10px;
	}
	.scores dt {
		font: 11px/1.5 var(--font-sans);
		color: var(--ink-2);
	}
	.score-value {
		margin: 0;
		font: 14px/1.5 var(--font-mono);
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}
	.score-value > span {
		font-size: 11px;
		color: var(--ink-3);
	}
	.cases {
		display: flex;
		gap: 5px;
	}
	.case {
		width: 28px;
		height: 28px;
		border: 1px solid var(--line);
		border-radius: 50%;
		background: transparent;
		color: var(--ink-2);
		font: 11px var(--font-mono);
	}
	.case:hover {
		border-color: var(--accent);
	}
	.case.active {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--surface);
	}
	button:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 3px;
	}
	.observations {
		display: grid;
		grid-template-columns: minmax(0, 116px) 64px minmax(0, 116px) 64px 58px;
		gap: 14px;
		align-items: center;
		justify-content: center;
	}
	.observation {
		margin: 0;
		min-width: 0;
	}
	.sensor {
		line-height: 0;
	}
	.sensor :global(canvas) {
		display: block;
		width: 100%;
		height: auto;
		aspect-ratio: 1;
	}
	.observation figcaption {
		margin-top: 6px;
		color: var(--ink-2);
		font: 10px var(--font-sans);
		text-align: center;
	}
	.action {
		display: grid;
		justify-items: center;
		gap: 4px;
		color: var(--warm);
		padding-bottom: 15px;
	}
	.action-label {
		font: 9px var(--font-sans);
	}
	.torques {
		display: grid;
		gap: 2px;
		font: 10px/1.2 var(--font-mono);
	}
	.action-arrow {
		font: 28px/0.8 var(--font-sans);
	}
	.next-moment {
		display: grid;
		justify-items: center;
		gap: 10px;
		padding-bottom: 15px;
		color: var(--ink-2);
	}
	.next-moment > span:first-child {
		font: 42px/1 var(--font-serif);
		color: var(--accent);
	}
	.next-moment > span:last-child {
		font: 9px var(--font-mono);
		white-space: nowrap;
	}
	.action-key {
		margin: 4px 0 12px;
		text-align: center;
		color: var(--ink-3);
		font: 10px/1.5 var(--font-sans);
	}
	.answer-heading {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		color: var(--ink-2);
		font: 11px/1.4 var(--font-sans);
		margin-bottom: 10px;
	}
	.actual-key {
		white-space: nowrap;
		color: var(--good);
	}
	.candidates {
		display: grid;
		grid-template-columns: repeat(6, minmax(0, 1fr));
		gap: 10px;
	}
	.candidate {
		margin: 0;
		padding: 6px;
		min-width: 0;
		border: 1px solid var(--line);
		background: var(--surface);
	}
	.candidate.chosen {
		border-color: var(--accent);
		outline: 1px solid var(--accent);
		outline-offset: -1px;
	}
	.candidate.chosen.before {
		border-style: dashed;
	}
	.candidate.chosen.copy {
		border-style: double;
	}
	.candidate-top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		height: 18px;
		margin-bottom: 4px;
		color: var(--ink-2);
		font: 10px var(--font-mono);
	}
	.choice-symbol {
		color: var(--accent);
		font-size: 10px;
	}
	.candidate :global(canvas) {
		border: 0;
	}
	.candidate figcaption {
		display: grid;
		gap: 3px;
		margin-top: 7px;
		font: 9px/1.4 var(--font-sans);
	}
	.actual-caption {
		color: var(--good);
	}
	.candidate.actual .actual-caption {
		font-weight: 600;
	}
	.result {
		margin: 10px 0 0;
		color: var(--ink-2);
		font: 12px/1.6 var(--font-sans);
		text-align: center;
	}
	.match {
		color: var(--good);
	}
	.miss {
		color: var(--warm);
	}
	.method-note {
		margin: 12px auto 0;
		color: var(--ink-2);
		font: 11px/1.65 var(--font-sans);
	}
	.empty-evidence {
		min-height: 350px;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 32px;
		text-align: center;
	}
	.empty-symbol {
		color: var(--accent);
		font: 36px var(--font-serif);
	}
	.empty-evidence p {
		color: var(--ink-2);
		font: 16px/1.5 var(--font-serif);
		margin: 18px 0 8px;
	}
	.empty-evidence > span:last-child {
		color: var(--ink-3);
		font: 12px/1.7 var(--font-sans);
		max-width: 360px;
	}
	@media (max-width: 600px) {
		.evidence-controls {
			justify-content: center;
		}
		.evidence-controls .seg > button {
			padding-inline: 9px;
			font-size: 10px;
		}
		.scores {
			gap: 8px 20px;
		}
		.question {
			font-size: 17px;
		}
		.observations {
			grid-template-columns: minmax(0, 1fr) 42px minmax(0, 1fr) 42px 42px;
			gap: 7px;
		}
		.action-label,
		.next-moment > span:last-child {
			font-size: 8px;
		}
		.torques {
			font-size: 9px;
		}
		.next-moment > span:first-child {
			font-size: 34px;
		}
		.case-nav {
			gap: 10px;
		}
		.candidates {
			grid-template-columns: repeat(3, minmax(0, 1fr));
			gap: 9px;
		}
		.answer-heading {
			font-size: 10px;
			align-items: start;
		}
		.answer-heading > span:first-child {
			max-width: 190px;
		}
		.candidate figcaption {
			font-size: 9px;
		}
		.empty-evidence {
			min-height: 420px;
			padding: 28px 16px;
		}
	}
	@media (pointer: coarse) {
		.case {
			width: 44px;
			height: 44px;
		}
		.cases {
			gap: 4px;
		}
	}
</style>
