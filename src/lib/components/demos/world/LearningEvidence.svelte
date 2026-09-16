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
	<div class="evidence-heading">
		<div>
			<span class="eyebrow">An unseen moment</span>
			<h3>Which picture comes next?</h3>
		</div>
		<span class="checkpoint">
			{#if evaluation}
				Measured at {evaluation.step.toLocaleString()} updates
				{#if training}<span class="measuring">Learning continues</span>{/if}
			{:else}
				Pictures held out of training
			{/if}
		</span>
	</div>

	<div class="score-views" role="group" aria-label="Compare predictions before and after learning">
		{#each choices as choice (choice.id)}
			<button
				type="button"
				class={['score-view', { active: view === choice.id }]}
				aria-pressed={view === choice.id}
				onclick={() => (view = choice.id)}
			>
				<span class="score-label">{choice.label}</span>
				<span class="score-value">{choice.correct ?? '—'}<span> / {choice.total ?? '—'}</span></span
				>
				<span class="score-note">correct matches{choice.ties ? ` · ${choice.ties} tied` : ''}</span>
			</button>
		{/each}
	</div>

	{#if example}
		<div class="case-nav">
			<span class="eyebrow">Look closely</span>
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
		<p class="action-key">Motor commands: base above, elbow below.</p>

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
								aria-label={`${choiceLabel} selection`}
								>{view === 'before' ? '◇' : view === 'copy' ? '□' : '●'}</span
							>{/if}
					</div>
					<div class="sensor"><Sensor {pixels} size={Math.sqrt(pixels.length)} /></div>
					<figcaption>
						<span class="chosen-caption"
							>{#if selected === index}{choiceLabel} chooses {letters[index]}{:else}<span
									aria-hidden="true">&nbsp;</span
								>{/if}</span
						>
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

	<p class="method-note">
		{probe?.total ?? 64} fixed held-out moments, {probe?.candidatesPerExample ?? 6} choices each. Ties
		count as misses. The pictures are recorded observations; the model predicts an embedding, not an image.
	</p>
</section>

<style>
	.evidence {
		max-width: 860px;
		margin: 22px auto 28px;
		padding-inline: 16px;
	}
	.evidence-heading {
		display: flex;
		justify-content: space-between;
		align-items: end;
		gap: 20px;
		margin-bottom: 22px;
	}
	h3 {
		margin: 5px 0 0;
		font: 400 clamp(25px, 3vw, 32px)/1.12 var(--font-serif);
		letter-spacing: -0.025em;
		color: var(--ink);
	}
	.checkpoint {
		display: grid;
		gap: 4px;
		text-align: right;
		font: 10px/1.5 var(--font-mono);
		color: var(--ink-2);
	}
	.measuring {
		color: var(--accent);
	}
	.score-views {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		border-top: 1px solid var(--line);
		border-bottom: 1px solid var(--line);
	}
	.score-view {
		display: grid;
		align-content: start;
		gap: 7px;
		min-width: 0;
		padding: 14px 16px;
		text-align: left;
		color: var(--ink-2);
		background: transparent;
		border-bottom: 2px solid transparent;
	}
	.score-view + .score-view {
		border-left: 1px solid var(--line);
	}
	.score-view:hover {
		background: var(--surface-2);
	}
	.score-view.active {
		background: var(--accent-soft);
		border-bottom-color: var(--accent);
		color: var(--accent);
	}
	.score-label {
		font: 11px/1.4 var(--font-sans);
	}
	.score-value {
		font: 26px/1 var(--font-mono);
		font-variant-numeric: tabular-nums;
	}
	.score-value > span {
		font-size: 13px;
		color: var(--ink-3);
	}
	.score-note {
		display: block;
		font: 10px/1.4 var(--font-sans);
		color: var(--ink-2);
	}
	.case-nav {
		display: flex;
		justify-content: center;
		align-items: center;
		flex-wrap: wrap;
		gap: 14px;
		margin: 22px 0 18px;
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
		margin: 10px 0 24px;
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
		box-shadow: 0 0 0 1px var(--accent);
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
		font-size: 13px;
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
	.chosen-caption {
		color: var(--accent);
	}
	.actual-caption {
		color: var(--good);
	}
	.candidate.actual .actual-caption {
		font-weight: 600;
	}
	.result {
		margin: 14px 0 0;
		min-height: 34px;
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
		max-width: 650px;
		color: var(--ink-3);
		font: 10px/1.6 var(--font-sans);
		text-align: center;
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
		.evidence-heading {
			flex-direction: column;
			align-items: start;
			gap: 10px;
		}
		.checkpoint {
			display: flex;
			flex-wrap: wrap;
			gap: 4px 12px;
			text-align: left;
			font-size: 10px;
		}
		.score-view {
			padding: 12px 9px;
		}
		.score-label {
			min-height: 30px;
			font-size: 10px;
		}
		.score-value {
			font-size: 22px;
		}
		.score-note {
			font-size: 10px;
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
