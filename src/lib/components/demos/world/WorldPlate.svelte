<script lang="ts">
	import { onDestroy } from 'svelte';
	import { fromAction } from 'svelte/attachments';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { type WorldLab } from '$lib/world/lab.svelte';
	import {
		initialArm,
		armPoints,
		stepArm,
		poseError,
		DEFAULT_DAMPING,
		type ArmState,
		type Action
	} from '$lib/world/simulator';
	import { renderSensor } from '$lib/world/sensor';
	import type { WorldPlan, WorldForecast } from '$lib/world/engine';
	import { rescoreFutures } from '$lib/world/future-choices';
	import {
		createForecastStart,
		forecastCommands,
		type ForecastProgram
	} from '$lib/world/forecast-scenario';
	import HistoryPlate from './HistoryPlate.svelte';
	import Instrument from './Instrument.svelte';
	import Projection from './Projection.svelte';
	import LearningCurve from './LearningCurve.svelte';
	import LearningEvidence from './LearningEvidence.svelte';
	import CandidateFutures from './CandidateFutures.svelte';

	type Mode = 'history' | 'train' | 'forecast' | 'collapse' | 'plan' | 'transfer';
	let { mode, lab }: { mode: Mode; lab: WorldLab } = $props();
	const titles: Record<Mode, string> = {
		history: 'The missing moment',
		train: 'Learn the consequences',
		forecast: 'Let a prediction continue',
		collapse: 'When everything looks the same',
		plan: 'Before the move',
		transfer: 'The same knowledge, a new intention'
	};
	const captions: Record<Mode, string> = {
		history: '',
		train:
			'Keep the test fixed while the model changes. Learning is useful when it predicts what happens next more often than simply assuming the present will persist.',
		forecast:
			'Ink is the simulator; blue is a diagnostic readout of predicted embeddings. The same torque sequence drives both. Longer rollouts feed predictions back into the model.',
		collapse:
			'Watch the prediction-only representation as it learns. Your trained model stays fixed on the left. At the end, both runs have the same initialization, examples, and training budget; only the distribution term differs.',
		plan: 'Rehearse considers actions without moving. Step tests the first action in the world; its predicted pose stays visible. Run repeats this loop with fresh observations and frozen weights.',
		transfer:
			'Rehearse once, then change the request: the same three predicted futures receive new scores. Step tests the preferred first action. Rehearse again to search beyond this small gallery.'
	};
	const goals: { name: string; pose: ArmState }[] = [
		{ name: 'Unfold', pose: { q1: -0.9, q2: 1.0, v1: 0, v2: 0 } },
		{ name: 'Return', pose: { q1: -1.8, q2: 1.8, v1: 0, v2: 0 } },
		{ name: 'Curl', pose: { q1: -1.05, q2: 2.25, v1: 0, v2: 0 } }
	];
	let physical = $state.raw<ArmState>(initialArm());
	let previous = initialArm();
	let previousAction: Action = [0, 0];
	let trail = $state.raw<ArmState[]>([]);
	let goalIndex = $state(0);
	let horizon = $state(6);
	let forecastHorizon = $state(3);
	let forecastProgram = $state<ForecastProgram>('push');
	const forecastPrograms: { id: ForecastProgram; name: string }[] = [
		{ id: 'push', name: 'Keep pushing' },
		{ id: 'reverse', name: 'Reverse torques' },
		{ id: 'release', name: 'Release' }
	];
	let hold = $state(true);
	let damping = $state(DEFAULT_DAMPING);
	let rehearsed = $state.raw<WorldPlan | null>(null);
	let rehearsedStart = $state.raw<ArmState>(initialArm());
	let forecast = $state.raw<WorldForecast | null>(null);
	let ghost = $state.raw<ArmState | null>(null);
	let actionPrediction = $state.raw<ArmState | null>(null);
	let actionPredictionStep = $state(-1);
	let actionPredictionCheckpoint = $state(-1);
	let actionReadoutError = $state(0);
	let forecastActual = $state.raw<ArmState[]>([]);
	let forecastError = $state<number | null>(null);
	let previewIndex = $state(0);
	let previewPlaying = $state(false);
	let running = $state(false);
	let executing = $state(false);
	let time = $state(0);
	let dwell = $state(0);
	let localNotice = $state('');
	let previewTimer: ReturnType<typeof setTimeout> | null = null;
	let animation = 0;
	let finishAnimation: (() => void) | null = null;
	let rehearsedStep = $state(-1);
	let forecastStep = $state(-1);
	let rehearsedCheckpoint = $state(-1);
	let rehearsedGoal = $state(0);
	let rehearsedHold = $state(true);
	let forecastCheckpoint = $state(-1);
	let generation = 0;
	let gone = false;
	const goal = $derived(goals[goalIndex].pose);
	const error = $derived(poseError(physical, goal));
	const preview = $derived(lab.info?.preview);
	const interval = $derived(lab.info?.dt ?? 0.24);
	const forecastStart = $derived(createForecastStart(interval));
	const canInfer = $derived(
		lab.step > 0 && lab.phase === 'ready' && !executing && !running && !lab.motionOwner
	);
	const validRehearsal = $derived(
		lab.checkpoint === rehearsedCheckpoint && lab.step === rehearsedStep && !lab.training
			? rehearsed
			: null
	);
	const gallery = $derived(
		validRehearsal && mode === 'transfer'
			? rescoreFutures(validRehearsal, goalIndex, hold)
			: { plan: validRehearsal, selectedIndex: 0 }
	);
	const visiblePlan = $derived(gallery.plan);
	const rescored = $derived(
		mode === 'transfer' && (goalIndex !== rehearsedGoal || hold !== rehearsedHold)
	);
	const visibleActionPrediction = $derived(
		lab.checkpoint === actionPredictionCheckpoint &&
			lab.step === actionPredictionStep &&
			!lab.training
			? actionPrediction
			: null
	);
	const visibleForecast = $derived(
		lab.checkpoint === forecastCheckpoint && lab.step === forecastStep && !lab.training
			? forecast
			: null
	);
	const displayed = $derived(
		mode === 'train'
			? (preview?.states[previewIndex] ?? initialArm())
			: mode === 'forecast' && !visibleForecast
				? forecastStart.current
				: physical
	);
	const pixels = $derived(
		mode === 'train' && preview
			? preview.frames.slice(previewIndex * 1024, (previewIndex + 1) * 1024)
			: renderSensor(displayed)
	);
	const forecasts = $derived(
		mode === 'forecast'
			? visibleForecast
				? [visibleForecast.poses]
				: []
			: visiblePlan
				? [
						visiblePlan.poses,
						...visiblePlan.candidates
							.filter((_, i) => i !== gallery.selectedIndex)
							.map((c) => c.poses)
					]
				: []
	);
	const frozen = $derived(mode === 'forecast' || mode === 'plan' || mode === 'transfer');
	const activeError = $derived(lab.error || localNotice);
	const comparisonReference = $derived(lab.comparison?.regularized ?? lab.evaluation);
	const comparisonTarget = $derived(lab.comparison?.steps ?? lab.step);
	const comparisonProgress = $derived(Math.min(comparisonTarget, lab.comparisonStep));
	const comparisonDescription = $derived(
		!lab.comparison
			? 'Compare restarts the same initial weights without the distribution term. The right-hand distribution will update as it learns.'
			: lab.comparison.status === 'complete'
				? `Both runs: ${lab.comparison.steps.toLocaleString()} updates. One fixed embedding coordinate is shown; spread measures all coordinates.`
				: lab.comparison.status === 'paused'
					? `Paused after ${lab.comparison.unregularizedStep.toLocaleString()} of ${comparisonTarget.toLocaleString()} updates. The last measured distributions remain visible. Restart begins again from the same initial weights.`
					: `Your model is frozen at ${comparisonTarget.toLocaleString()} updates. Prediction alone is catching up; compare their final results at the same training budget.`
	);

	function poseGlyph(pose: ArmState) {
		const points = armPoints(pose);
		return [points.base, points.elbow, points.tip]
			.map((p, i) => `${i ? 'L' : 'M'}${20 + p.x * 21},${20 - p.y * 21}`)
			.join(' ');
	}
	function formatMetric(value: number | undefined, digits: number) {
		if (value === undefined) return '—';
		return value !== 0 && Math.abs(value) < 10 ** -digits
			? value.toExponential(1).replace('e-', 'e−')
			: value.toFixed(digits);
	}
	function boot() {
		if (mode !== 'history') void lab.boot();
	}
	function stopLocal() {
		running = false;
		executing = false;
		generation++;
		if (animation) cancelAnimationFrame(animation);
		finishAnimation?.();
		finishAnimation = null;
		if (lab.motionOwner === mode) lab.motionOwner = null;
	}
	function resetInstrument() {
		stopLocal();
		physical = mode === 'forecast' ? forecastStart.current : initialArm();
		previous = mode === 'forecast' ? forecastStart.previous : initialArm();
		previousAction = mode === 'forecast' ? forecastStart.previousAction : [0, 0];
		trail = [];
		rehearsed = null;
		forecast = null;
		ghost = null;
		actionPrediction = null;
		time = 0;
		dwell = 0;
		forecastError = null;
		localNotice = '';
	}
	function changedGoal() {
		stopLocal();
		if (mode !== 'transfer') rehearsed = null;
		dwell = 0;
		localNotice = '';
	}
	function togglePreview() {
		if (previewPlaying) {
			previewPlaying = false;
			if (previewTimer) clearTimeout(previewTimer);
			return;
		}
		if (!preview) return;
		if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
			previewIndex = (previewIndex + 1) % preview.states.length;
			return;
		}
		previewPlaying = true;
		previewIndex = 0;
		const next = () => {
			if (gone || !previewPlaying || !preview) return;
			if (previewIndex === preview.states.length - 1) {
				previewPlaying = false;
				return;
			}
			previewIndex++;
			previewTimer = setTimeout(next, interval * 1000);
		};
		previewTimer = setTimeout(next, interval * 1000);
	}
	function observations(): Float32Array {
		const frames = new Float32Array(2048);
		frames.set(renderSensor(previous));
		frames.set(renderSensor(physical), 1024);
		return frames;
	}
	async function rehearse() {
		localNotice = '';
		const token = generation;
		const start = physical;
		const result = await lab.plan({
			observations: observations(),
			previousAction,
			goal: renderSensor(goal),
			horizon,
			hold: mode === 'plan' ? false : hold,
			seed: 300 + Math.round(time / interval),
			comparisonGoals: mode === 'transfer' ? goals.map((g) => renderSensor(g.pose)) : undefined
		});
		if (gone || token !== generation) return null;
		if (result) {
			rehearsed = result;
			rehearsedStart = start;
			rehearsedStep = lab.step;
			rehearsedCheckpoint = lab.checkpoint;
			rehearsedGoal = goalIndex;
			rehearsedHold = hold;
		}
		return result;
	}
	async function execute(action: Action, prediction: ArmState, readoutError: number) {
		actionPrediction = prediction;
		actionPredictionStep = lab.step;
		actionPredictionCheckpoint = lab.checkpoint;
		actionReadoutError = readoutError;
		executing = true;
		const start = physical;
		const token = generation;
		const frames: ArmState[] = [];
		let s = start;
		for (let i = 0; i < 8; i++) {
			s = stepArm(s, action, { dt: interval / 8, damping });
			frames.push(s);
		}
		if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
			physical = frames[7];
			trail = [...trail, ...frames].slice(-180);
		} else
			await new Promise<void>((resolve) => {
				finishAnimation = resolve;
				let index = 0;
				const draw = () => {
					if (gone || generation !== token) {
						resolve();
						return;
					}
					physical = frames[index++];
					trail = [...trail, physical].slice(-180);
					if (index < frames.length) animation = requestAnimationFrame(draw);
					else {
						finishAnimation = null;
						resolve();
					}
				};
				animation = requestAnimationFrame(draw);
			});
		if (gone || token !== generation) return;
		previous = start;
		previousAction = action;
		time += interval;
		dwell = poseError(physical, goal) <= 0.15 ? dwell + interval : 0;
		rehearsed = null;
		executing = false;
	}
	async function step() {
		lab.motionOwner = mode;
		const token = generation;
		const selected = visiblePlan ?? (await rehearse());
		if (selected && !gone && token === generation) {
			await execute(selected.action, selected.poses[0], selected.readoutError);
			rehearsed = null;
		}
		if (lab.motionOwner === mode) lab.motionOwner = null;
	}
	async function toggleRun() {
		if (running) {
			running = false;
			localNotice = 'Pausing after the current action.';
			return;
		}
		running = true;
		lab.motionOwner = mode;
		localNotice = '';
		const token = generation;
		for (let i = 0; i < 60 && running && token === generation && !gone; i++) {
			const selected = await rehearse();
			if (!selected || !running || token !== generation) break;
			await execute(selected.action, selected.poses[0], selected.readoutError);
			if (dwell >= 5 * interval - 1e-8) {
				localNotice = 'Within 0.15 rad for five consecutive observations.';
				break;
			}
		}
		if (token === generation) {
			running = false;
			executing = false;
			if (lab.motionOwner === mode) lab.motionOwner = null;
		}
	}
	async function predict() {
		resetInstrument();
		lab.motionOwner = mode;
		const actions = forecastCommands(forecastProgram, forecastHorizon);
		const token = generation;
		const result = await lab.forecast({ observations: observations(), previousAction, actions });
		if (!result || gone || token !== generation) {
			if (lab.motionOwner === mode) lab.motionOwner = null;
			return;
		}
		forecast = result;
		forecastStep = lab.step;
		forecastCheckpoint = lab.checkpoint;
		let s = physical;
		forecastActual = [];
		for (let i = 0; i < forecastHorizon; i++) {
			s = stepArm(s, [actions[i * 2], actions[i * 2 + 1]], { dt: interval });
			forecastActual = [...forecastActual, s];
		}
		forecastError = Math.sqrt(
			forecastActual.reduce((sum, s, i) => sum + poseError(s, result.poses[i]) ** 2, 0) /
				forecastHorizon
		);
		trail = [physical, ...forecastActual];
		if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
			physical = s;
			ghost = result.poses.at(-1)!;
			lab.motionOwner = null;
			return;
		}
		executing = true;
		let index = 0;
		let last: number | null = null;
		const draw = (now: number) => {
			if (gone || token !== generation) return;
			last ??= now;
			if (now - last >= interval * 1000) {
				physical = forecastActual[index];
				ghost = result.poses[index];
				index++;
				last = now;
			}
			if (index < forecastHorizon) animation = requestAnimationFrame(draw);
			else {
				executing = false;
				if (lab.motionOwner === mode) lab.motionOwner = null;
			}
		};
		animation = requestAnimationFrame(draw);
	}
	onDestroy(() => {
		gone = true;
		stopLocal();
		if (previewTimer) clearTimeout(previewTimer);
	});
</script>

{#if mode === 'history'}
	<HistoryPlate />
{:else}
	<Plate id={mode} title={titles[mode]} caption={captions[mode]} live={mode === 'train'}>
		{#snippet status()}
			{#if lab.phase === 'loading'}<span>Collecting local experience…</span>
			{:else if lab.phase === 'forecasting' || lab.phase === 'planning'}<span
					>Considering consequences…</span
				>
			{:else if lab.phase === 'comparing'}<span
					>comparison · step {lab.comparisonStep} / {comparisonTarget}</span
				>
			{:else if lab.training}<span>step {lab.step} · {lab.metrics?.stepMs.toFixed(0)} ms/step</span>
			{:else if lab.step}<span>{frozen ? 'weights frozen · ' : ''}step {lab.step}</span>
			{:else}<span>from random weights</span>{/if}
			{#if mode === 'train' && lab.metrics}
				<span>· loss {lab.metrics.loss.toFixed(4)}</span>
			{/if}
		{/snippet}
		{#snippet actions()}
			{#if mode === 'train'}
				<Btn
					kind={lab.training ? 'ghost' : 'primary'}
					onclick={() => void lab.toggleTrain()}
					disabled={!lab.info ||
						lab.phase === 'error' ||
						(lab.busy && !lab.training) ||
						!!lab.motionOwner}>{lab.training ? 'Pause' : lab.step ? 'Train more' : 'Train'}</Btn
				>
				<Btn
					onclick={() => {
						resetInstrument();
						void lab.reset();
					}}
					disabled={lab.busy || !!lab.motionOwner}>Reset</Btn
				>
			{:else if mode === 'collapse'}
				<Btn
					kind={lab.phase === 'comparing' ? 'ghost' : 'primary'}
					disabled={!lab.step || !!lab.motionOwner || (lab.busy && lab.phase !== 'comparing')}
					onclick={() => void (lab.phase === 'comparing' ? lab.stop() : lab.compare())}
					>{lab.phase === 'comparing'
						? 'Pause'
						: lab.comparison?.status === 'paused'
							? 'Restart comparison'
							: lab.comparison?.status === 'complete'
								? 'Compare again'
								: 'Compare'}</Btn
				>
			{:else if mode === 'forecast'}
				<Btn kind="primary" disabled={!canInfer} onclick={() => void predict()}
					>Forecast & replay</Btn
				>
				<Btn onclick={resetInstrument} disabled={lab.busy}>Reset scene</Btn>
			{:else}
				<Btn
					kind={!visiblePlan && !running ? 'primary' : 'ghost'}
					disabled={!canInfer}
					onclick={() => void rehearse()}>Rehearse</Btn
				>
				<Btn disabled={!canInfer} onclick={() => void step()}>Step</Btn>
				<Btn
					kind={visiblePlan && !running ? 'primary' : 'ghost'}
					disabled={!running && !canInfer}
					onclick={() => void toggleRun()}>{running ? 'Pause' : 'Run'}</Btn
				>
				<Btn onclick={resetInstrument} disabled={lab.busy}>Reset scene</Btn>
			{/if}
		{/snippet}
		<div {@attach fromAction(inview, () => boot)}>
			{#if mode === 'collapse'}
				<div class="comparison-grid">
					{#each [{ key: 'regularized', name: 'Prediction + distribution', warm: false }, { key: 'unregularized', name: 'Prediction alone', warm: true }] as column (column.key)}
						{@const result =
							column.key === 'regularized' ? comparisonReference : lab.comparison?.unregularized}
						<div class="comparison-column">
							<h3>{column.name}</h3>
							<p class="checkpoint-label">
								{#if column.key === 'regularized'}
									{result
										? `${lab.comparison ? 'Frozen' : 'Current model'} · ${result.step.toLocaleString()} updates`
										: 'Your trained model'}
								{:else if result}
									{lab.comparison?.status === 'complete' ? 'Matched budget' : 'Measured so far'} · {result.step.toLocaleString()}
									updates
								{:else}
									{lab.phase === 'comparing'
										? 'Preparing the same starting weights…'
										: 'Same starting weights · prediction loss only'}
								{/if}
							</p>
							<Projection
								values={result?.projection ?? []}
								label={column.name}
								warm={column.warm}
								emptyLabel={column.key === 'regularized'
									? 'Train the model above'
									: lab.phase === 'comparing'
										? 'First measurement on its way…'
										: 'Compare to watch this model learn'}
							/>
							<dl>
								<div>
									<dt>Embedding spread</dt>
									<dd>{formatMetric(result?.spread, 3)}</dd>
								</div>
								<div>
									<dt>Prediction loss</dt>
									<dd>{formatMetric(result?.predictionLoss, 4)}</dd>
								</div>
								<div>
									<dt>Matched unseen futures</dt>
									<dd>
										{result
											? `${result.futureMatching.correct} / ${result.futureMatching.total}`
											: '—'}
									</dd>
								</div>
							</dl>
						</div>
					{/each}
				</div>
				{#if lab.comparison || lab.phase === 'comparing'}
					<div class="comparison-progress">
						<div class="progress-caption">
							<span
								>{lab.phase === 'comparing'
									? 'Training prediction alone'
									: lab.comparison?.status === 'complete'
										? 'Comparison complete'
										: 'Comparison paused'}</span
							>
							<span class="num"
								>{comparisonProgress.toLocaleString()} / {comparisonTarget.toLocaleString()} updates</span
							>
						</div>
						<progress
							max={Math.max(1, comparisonTarget)}
							value={comparisonProgress}
							aria-label="Prediction-only training progress"
						></progress>
					</div>
				{/if}
				<p class="quiet comparison-note">{comparisonDescription}</p>
				<p class="quiet">
					The matching test is the same one used above. A smaller loss can accompany a loss of
					useful distinctions.
				</p>
			{:else}
				{#if mode === 'train'}
					<LearningEvidence
						evaluation={lab.evaluation}
						initial={lab.initialEvaluation}
						training={lab.training}
						{interval}
					/>
				{:else}
					<Instrument
						state={displayed}
						compact={mode === 'plan' || mode === 'transfer'}
						goal={mode === 'plan' || mode === 'transfer' ? goal : null}
						trail={mode === 'forecast' && !visibleForecast ? [] : trail}
						{forecasts}
						past={mode === 'forecast' && !visibleForecast ? [forecastStart.previous] : []}
						ghost={visibleForecast ? ghost : visibleActionPrediction}
						{pixels}
						label="Actual instrument with diagnostic forecasts and a desired pose"
					/>
				{/if}
				{#if mode === 'train'}
					<div class="training-telemetry">
						<LearningCurve
							history={lab.history}
							regularization={lab.info?.config.regularization ?? 0.01}
						/>
						<p class="loss-guide">
							Training minimizes the sum of these two terms: predict the next embedding, while
							keeping different observations distinguishable. The unseen-picture test above checks
							what that buys us.
						</p>
					</div>
					<div class="training-strip">
						<span>5,000 updates per training run</span>
						<span
							>{lab.info
								? `${lab.info.transitions.toLocaleString()} collected transitions`
								: 'Generating pictures and torques locally'}</span
						><span
							>{lab.info
								? `${lab.info.parameters.toLocaleString()} parameters · ${lab.backend}`
								: 'No downloaded weights'}</span
						>
					</div>
					<details class="training-detail">
						<summary>Replay the experience used for training</summary>
						<Instrument
							state={displayed}
							{pixels}
							label="A recorded exploratory movement from the training corpus"
						/>
						<div class="experience-actions">
							<Btn onclick={togglePreview} disabled={!preview}
								>{previewPlaying ? 'Pause experience' : 'Replay experience'}</Btn
							>
						</div>
					</details>
				{:else if mode === 'forecast'}
					<div class="controls forecast-controls">
						<div class="choice-group" role="group" aria-label="Proposed actions">
							<span class="eyebrow">Same moving start, different action</span>
							<div class="seg program-choices">
								{#each forecastPrograms as program (program.id)}<button
										class={{ on: forecastProgram === program.id }}
										aria-pressed={forecastProgram === program.id}
										disabled={lab.busy || executing}
										onclick={() => {
											forecastProgram = program.id;
											resetInstrument();
										}}>{program.name}</button
									>{/each}
							</div>
						</div>
						<div class="choice-group" role="group" aria-label="Forecast length">
							<span class="eyebrow">Forecast length</span>
							<div class="choice-row">
								<div class="seg">
									{#each [3, 6, 12, 20] as n (n)}
										<button
											class={{ on: forecastHorizon === n }}
											aria-pressed={forecastHorizon === n}
											disabled={lab.busy || executing}
											title={`${n} observations · ${(n * interval).toFixed(2)} seconds`}
											onclick={() => {
												forecastHorizon = n;
												resetInstrument();
											}}>{n}</button
										>
									{/each}
								</div>
								<span class="choice-detail"
									>observations · {(forecastHorizon * interval).toFixed(2)} s</span
								>
							</div>
						</div>
						<span class="legend"
							><i class="actual"></i>Observed <i class="predicted"></i>Predicted readout</span
						>
					</div>
					<p class="quiet">
						All three choices begin after the same short push. Release turns the motors off;
						existing motion continues and gradually slows. The faint pose shows the preceding
						observation.
					</p>
					{#if visibleForecast && forecastError !== null}<p class="quiet">
							Rollout ghost error <span class="num">{forecastError.toFixed(2)} rad</span> · readout
							error on real held-out frames
							<span class="num">{forecast?.readoutError.toFixed(2)} rad</span>
						</p>{/if}
				{:else}
					<div class="controls planning-controls">
						<div class="choice-group" role="group" aria-label="Desired pose">
							<span class="eyebrow"
								>{mode === 'transfer' ? 'Change the request · destination' : 'Desired pose'}</span
							>
							<div class="choice-row pose-choices">
								{#each goals as g, i (g.name)}
									<button
										class={['chip', 'pose-choice', { 'chip-on': goalIndex === i }]}
										aria-pressed={goalIndex === i}
										disabled={running || executing || lab.busy}
										onclick={() => {
											goalIndex = i;
											changedGoal();
										}}
									>
										<svg viewBox="0 0 40 40" aria-hidden="true"
											><path d={poseGlyph(g.pose)} /><circle cx="20" cy="20" r="2" /></svg
										>
										{g.name}
									</button>
								{/each}
							</div>
						</div>
						<div class="choice-group" role="group" aria-label="Foresight">
							<span class="eyebrow">Foresight</span>
							<div class="choice-row">
								<div class="seg">
									{#each [6, 12, 20] as n (n)}
										<button
											class={{ on: horizon === n }}
											aria-pressed={horizon === n}
											disabled={running || executing || lab.busy}
											title={`${n} observations · ${(n * interval).toFixed(2)} seconds`}
											onclick={() => {
												horizon = n;
												rehearsed = null;
											}}>{n}</button
										>
									{/each}
								</div>
								<span class="choice-detail">observations · {(horizon * interval).toFixed(2)} s</span
								>
							</div>
						</div>
						{#if mode === 'transfer'}
							<div class="choice-group" role="group" aria-label="Intention">
								<span class="eyebrow">Change the request · intention</span>
								<div class="seg">
									{#each [{ value: false, name: 'Match the pose' }, { value: true, name: 'Arrive and remain' }] as choice (choice.name)}
										<button
											class={{ on: hold === choice.value }}
											aria-pressed={hold === choice.value}
											disabled={running || executing || lab.busy}
											onclick={() => {
												hold = choice.value;
												dwell = 0;
											}}>{choice.name}</button
										>
									{/each}
								</div>
							</div>
							<div class="choice-group" role="group" aria-label="Joint damping">
								<span class="eyebrow">Change the world · damping</span>
								<div class="seg">
									{#each [{ value: DEFAULT_DAMPING * 0.25, name: 'Less drag' }, { value: DEFAULT_DAMPING, name: 'Familiar' }, { value: DEFAULT_DAMPING * 2.5, name: 'More drag' }] as choice (choice.name)}
										<button
											class={{ on: damping === choice.value }}
											aria-pressed={damping === choice.value}
											disabled={running || executing || lab.busy}
											onclick={() => {
												damping = choice.value;
												dwell = 0;
											}}>{choice.name}</button
										>
									{/each}
								</div>
							</div>
						{/if}
					</div>
					<div class="plan-readout">
						<span class="num">pose error {error.toFixed(2)} rad · {time.toFixed(1)} s</span><span
							>{visiblePlan
								? `plan cost ${visiblePlan.cost.toFixed(4)} · ${rescored ? 'cached futures rescored' : `${visiblePlan.ms.toFixed(0)} ms`}`
								: 'Observe → rehearse → act'}</span
						>
					</div>
					{#if visiblePlan}
						<CandidateFutures
							candidates={visiblePlan.candidates}
							start={rehearsedStart}
							{goal}
							selectedIndex={gallery.selectedIndex}
							label={mode === 'transfer'
								? 'Same predictions · a different preference'
								: 'Three action sequences, three predicted futures'}
						/>
						<p class="quiet">
							{mode === 'transfer'
								? `Only the scores change when you change the request. ${hold ? 'Last four moments' : 'Final moment'} scored against the goal, plus motor effort. Step uses the best of these three; Run searches afresh.`
								: 'Step executes only the first action of the selected future. The rest is reconsidered after another observation.'}
							Drawings use a diagnostic readout ({visiblePlan.readoutError.toFixed(2)} rad held-out error).
						</p>
					{:else if mode === 'transfer'}
						<p class="quiet">
							Rehearse to collect three possible futures. Then change the destination or intention
							and watch their ranking.
						</p>
					{/if}
					{#if visibleActionPrediction && !executing}
						<p class="quiet action-evidence">
							<span class="legend"
								><i class="actual"></i>What happened <i class="predicted"></i>What was predicted</span
							><br />After one action: {poseError(physical, visibleActionPrediction).toFixed(2)} rad apart.
							This includes display-readout error ({actionReadoutError.toFixed(2)} rad on held-out pictures).
						</p>
					{/if}
				{/if}
			{/if}
			{#if mode !== 'train' && lab.step === 0}<p class="training-needed">
					Train the shared model in “Learn the consequences” above, then return to this experiment.
				</p>{/if}
			{#if activeError}<p class="message" role={lab.error ? 'alert' : 'status'}>
					{activeError}
				</p>{/if}
			{#if lab.phase === 'error'}<div class="retry">
					<Btn onclick={() => void lab.retry()}>Restart experiment</Btn>
				</div>{/if}
		</div>
	</Plate>
{/if}

<style>
	.training-telemetry {
		max-width: 852px;
		margin: 0 auto;
		padding: 16px 16px 0;
		border-top: 1px solid var(--line-soft);
	}
	.loss-guide {
		margin: 10px 0 0;
		font: 11px/1.65 var(--font-sans);
		color: var(--ink-2);
	}
	.program-choices {
		flex-wrap: wrap;
	}
	.training-detail {
		max-width: 820px;
		margin: 24px auto 0;
		border-top: 1px solid var(--line-soft);
		font: 12px/1.5 var(--font-sans);
		color: var(--ink-2);
	}
	.training-detail summary {
		cursor: pointer;
		padding: 16px 0;
	}
	.training-detail summary:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 4px;
	}
	.experience-actions {
		display: flex;
		justify-content: center;
		margin-bottom: 16px;
	}
	.action-evidence {
		padding-top: 16px;
		border-top: 1px solid var(--line-soft);
	}
	.controls {
		border-top: 1px solid var(--line-soft);
		padding: 18px 16px 0;
		max-width: 820px;
		margin: 0 auto;
	}
	.planning-controls {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 18px 32px;
	}
	.forecast-controls {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 16px 24px;
	}
	.choice-group {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 8px;
		min-width: 0;
	}
	.choice-group > .eyebrow {
		color: var(--ink-3);
	}
	.choice-row {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px;
	}
	.choice-detail {
		font: 10px var(--font-mono);
		color: var(--ink-3);
	}
	.pose-choices {
		gap: 6px;
	}
	.pose-choice {
		padding: 2px 9px 2px 4px;
	}
	.pose-choice svg {
		width: 30px;
		height: 30px;
		fill: none;
		stroke: var(--warm);
		stroke-width: 1.25;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.pose-choice circle {
		fill: var(--surface);
	}
	.controls button:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 3px;
	}
	@media (max-width: 680px) {
		.planning-controls {
			grid-template-columns: 1fr;
			gap: 18px;
		}
		.controls {
			padding-inline: 16px;
		}
	}
	.training-strip,
	.plan-readout {
		display: flex;
		justify-content: center;
		flex-wrap: wrap;
		gap: 10px 28px;
		font: 11px/1.5 var(--font-sans);
		color: var(--ink-3);
		margin: 12px 0 18px;
	}
	.plan-readout {
		margin-bottom: 0;
	}
	.quiet,
	.training-needed,
	.message {
		margin: 16px auto 0;
		max-width: 640px;
		text-align: center;
		color: var(--ink-3);
		font: 12px/1.6 var(--font-sans);
	}
	.training-needed {
		color: var(--ink-2);
	}
	.message {
		color: var(--warm);
	}
	.retry {
		display: flex;
		justify-content: center;
		margin-top: 12px;
	}
	.legend {
		display: inline-flex;
		gap: 6px;
		align-items: center;
		font: 11px var(--font-sans);
		color: var(--ink-3);
	}
	.legend i {
		width: 16px;
		border-top: 1.5px solid var(--ink-2);
	}
	.legend .predicted {
		border-color: var(--accent);
		margin-left: 8px;
	}
	.comparison-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 32px;
		padding: 28px 0 8px;
		max-width: 840px;
		margin: auto;
	}
	.comparison-column + .comparison-column {
		border-left: 1px solid var(--line-soft);
		padding-left: 32px;
	}
	h3 {
		text-align: center;
		margin: 0 0 5px;
		font: italic 20px var(--font-serif);
		color: var(--ink);
	}
	.checkpoint-label {
		min-height: 2.8em;
		margin: 0 0 12px;
		text-align: center;
		font: 10px/1.4 var(--font-mono);
		color: var(--ink-3);
	}
	.comparison-progress {
		max-width: 800px;
		padding: 0 16px;
		margin: 22px auto 0;
	}
	.progress-caption {
		display: flex;
		justify-content: space-between;
		gap: 16px;
		font: 11px/1.5 var(--font-sans);
		color: var(--ink-2);
		margin-bottom: 8px;
	}
	progress {
		display: block;
		width: 100%;
		height: 3px;
		border: 0;
		appearance: none;
		background: var(--line-soft);
		color: var(--warm);
	}
	progress::-webkit-progress-bar {
		background: var(--line-soft);
	}
	progress::-webkit-progress-value {
		background: var(--warm);
	}
	progress::-moz-progress-bar {
		background: var(--warm);
	}
	.comparison-note {
		padding: 0 16px;
	}
	dl {
		display: grid;
		gap: 8px;
		margin: 18px auto 0;
		max-width: 270px;
		font: 11px var(--font-sans);
	}
	dl div {
		display: flex;
		justify-content: space-between;
		gap: 12px;
	}
	dt {
		color: var(--ink-2);
	}
	dd {
		margin: 0;
		font-family: var(--font-mono);
		color: var(--ink);
	}
	@media (max-width: 580px) {
		.comparison-grid {
			gap: 18px;
			grid-template-columns: 1fr;
		}
		.comparison-column + .comparison-column {
			border-left: none;
			padding: 20px 0 0;
			border-top: 1px solid var(--line-soft);
		}
		.controls {
			gap: 12px;
		}
	}
	@media (pointer: coarse) {
		.controls button {
			min-height: 44px;
		}
	}
</style>
