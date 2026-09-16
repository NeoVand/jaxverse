<script lang="ts">
	import { onDestroy } from 'svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Instrument from './Instrument.svelte';
	import { stepArm, type ArmState, type Action } from '$lib/world/simulator';

	const action: Action = [0.2, 0.08];
	const coast: Action = [0, 0];
	const dt = 1 / 60;
	const pushSteps = 120;
	const totalSteps = 420;
	const start: ArmState = { q1: -1.9, q2: 1.45, v1: 0.45, v2: -0.55 };
	const opposite: ArmState = { ...start, v1: -start.v1, v2: -start.v2 };
	let left = $state.raw<ArmState>(start);
	let right = $state.raw<ArmState>(opposite);
	let leftTrail = $state.raw<ArmState[]>([]);
	let rightTrail = $state.raw<ArmState[]>([]);
	let revealed = $state(false);
	let running = $state(false);
	let tick = $state(0);
	let stepping = $state(false);
	const elapsed = $derived(tick * dt);
	const complete = $derived(tick >= totalSteps);
	const driving = $derived(tick < pushSteps && (running || tick > 0));
	const command = $derived(driving ? action : null);
	const phase = $derived(
		complete
			? 'The same push. Two different outcomes.'
			: tick >= pushSteps
				? 'Motors off. The instruments coast.'
				: driving
					? 'Both instruments receive the same motor commands.'
					: 'Two seconds of torque, then five seconds of coasting.'
	);
	let frame = 0;
	let previous = 0;
	let accumulator = 0;
	function past(s: ArmState): ArmState[] {
		// Undriven, undamped mechanics are time-reversible. Integrating with
		// reversed velocities gives real preceding poses rather than a sketch.
		let reversed = { ...s, v1: -s.v1, v2: -s.v2 };
		const frames: ArmState[] = [{ ...s }];
		for (let i = 0; i < 36; i++) {
			reversed = stepArm(reversed, coast, { dt, damping: 0 });
			frames.unshift({ ...reversed, v1: -reversed.v1, v2: -reversed.v2 });
		}
		return frames;
	}
	const leftPast = past(start);
	const rightPast = past(opposite);
	const leftHistory = $derived(
		revealed && tick === 0 ? [leftPast[0], leftPast[12], leftPast[24]] : []
	);
	const rightHistory = $derived(
		revealed && tick === 0 ? [rightPast[0], rightPast[12], rightPast[24]] : []
	);
	function reset() {
		cancelAnimationFrame(frame);
		running = false;
		previous = 0;
		accumulator = 0;
		tick = 0;
		left = { ...start };
		right = { ...opposite };
		leftTrail = revealed ? leftPast : [];
		rightTrail = revealed ? rightPast : [];
	}
	function reveal() {
		revealed = !revealed;
		reset();
	}
	function advance() {
		const torque = tick < pushSteps ? action : coast;
		left = stepArm(left, torque, { dt, damping: 0 });
		right = stepArm(right, torque, { dt, damping: 0 });
		tick++;
		leftTrail = [...leftTrail, left];
		rightTrail = [...rightTrail, right];
	}
	function animate(now: number) {
		if (!running) return;
		if (previous) accumulator += Math.min((now - previous) / 1000, 0.05);
		previous = now;
		while (accumulator >= dt && !complete) {
			advance();
			accumulator -= dt;
		}
		if (complete) {
			running = false;
			return;
		}
		frame = requestAnimationFrame(animate);
	}
	function play() {
		if (running) {
			running = false;
			cancelAnimationFrame(frame);
			previous = 0;
			accumulator = 0;
			return;
		}
		if (complete) reset();
		stepping = matchMedia('(prefers-reduced-motion: reduce)').matches;
		if (stepping) {
			for (let i = 0; i < 30 && !complete; i++) advance();
			return;
		}
		previous = 0;
		accumulator = 0;
		running = true;
		frame = requestAnimationFrame(animate);
	}
	onDestroy(() => {
		if (frame) cancelAnimationFrame(frame);
	});
</script>

<Plate
	id="history"
	title="The missing moment"
	caption="Identical torques for two seconds, then five seconds of coasting. This undamped simulation starts with opposite angular velocities, before any learning."
>
	{#snippet status()}<span>simulator · {elapsed.toFixed(1)} / 7.0 s</span>{/snippet}
	{#snippet actions()}<Btn onclick={reveal} pressed={revealed}>History</Btn><Btn
			kind={running ? 'ghost' : 'primary'}
			onclick={play}
			>{running
				? 'Pause'
				: complete
					? 'Replay'
					: tick > 0
						? stepping
							? 'Next moment'
							: 'Resume'
						: 'Apply torques'}</Btn
		><Btn onclick={reset}>Reset</Btn>{/snippet}
	<div class="pair">
		<div>
			<Instrument
				state={left}
				trail={leftTrail}
				past={leftHistory}
				action={command}
				compact
				label="First instrument, initially turning in one direction"
			/>
			<p>{revealed ? 'One starting motion' : tick > 0 ? 'First instrument' : 'The first pose'}</p>
			{#if revealed}<div class="velocity num">initial joint motion +0.45, −0.55 rad/s</div>{/if}
		</div>
		<div>
			<Instrument
				state={right}
				trail={rightTrail}
				past={rightHistory}
				action={command}
				compact
				label="Second instrument, initially turning in the opposite direction"
			/>
			<p>{revealed ? 'The opposite motion' : tick > 0 ? 'Second instrument' : 'The same pose'}</p>
			{#if revealed}<div class="velocity num">initial joint motion −0.45, +0.55 rad/s</div>{/if}
		</div>
	</div>
	<div
		class="sequence"
		aria-label="The shared sequence: two seconds of torque, five seconds coasting"
	>
		<div class="sequence-labels">
			<span class:active={driving}>Same torque · 2 s</span><span class:active={tick >= pushSteps}
				>Coast · 5 s</span
			>
		</div>
		<div class="sequence-track"><span style:width={`${(tick / totalSteps) * 100}%`}></span></div>
		<div class="phase" aria-live="polite">{phase}</div>
		<div class="commands num" class:muted={!driving}>
			motor 1 {driving ? '+0.20' : '0.00'} · motor 2 {driving ? '+0.08' : '0.00'}
		</div>
	</div>
	<div class="history-note">
		{revealed
			? tick === 0
				? 'The faint poses show the preceding 0.6 seconds. Both instruments will receive the same commands.'
				: 'Their initial joint velocities were equal and opposite. Both followed the same sequence of commands.'
			: tick === 0
				? 'What can you predict from these two pictures?'
				: 'Starting velocity changes what happens next. Reveal the history to see what the first pictures left out.'}
	</div>
</Plate>

<style>
	.pair {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 24px;
	}
	.pair > div + div {
		border-left: 1px solid var(--line-soft);
	}
	p {
		text-align: center;
		color: var(--ink-2);
		font: italic 17px var(--font-serif);
		margin: 0;
	}
	.velocity {
		margin-top: 5px;
		text-align: center;
		color: var(--ink-3);
		font-size: 10px;
	}
	.sequence {
		max-width: 440px;
		margin: 26px auto 0;
		padding-inline: 12px;
	}
	.sequence-labels {
		display: grid;
		grid-template-columns: 2fr 5fr;
		color: var(--ink-3);
		font: 10px var(--font-sans);
	}
	.sequence-labels > span + span {
		padding-left: 9px;
	}
	.sequence-labels .active {
		color: var(--warm);
	}
	.sequence-track {
		height: 3px;
		margin-block: 8px 14px;
		background: var(--line);
		position: relative;
	}
	.sequence-track::after {
		content: '';
		position: absolute;
		left: 28.5714%;
		top: -3px;
		height: 9px;
		border-left: 1px solid var(--ink-3);
	}
	.sequence-track span {
		display: block;
		height: 100%;
		background: var(--warm);
	}
	.phase {
		min-height: 1.5em;
		text-align: center;
		color: var(--ink-2);
		font: 12px/1.5 var(--font-sans);
	}
	.commands {
		text-align: center;
		margin-top: 4px;
		color: var(--warm);
		font-size: 10px;
	}
	.commands.muted {
		color: var(--ink-3);
	}
	.history-note {
		text-align: center;
		margin-top: 12px;
		padding-inline: 12px;
		color: var(--ink-3);
		font: 12px/1.5 var(--font-sans);
	}
	@media (max-width: 500px) {
		.pair {
			gap: 6px;
		}
		.velocity {
			font-size: 9px;
			max-width: 150px;
			margin-inline: auto;
		}
	}
</style>
