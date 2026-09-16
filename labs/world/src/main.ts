// Lab 11 — Before the Move
import { WorldEngine } from './world/worker-engine';
import type { WorldPlan } from './world/engine';
import {
	armPoints,
	initialArm,
	poseError,
	stepArm,
	type Action,
	type ArmState
} from './world/simulator';
import { renderSensor, SENSOR_SIZE } from './world/sensor';

// These are the experiment, not a pretrained checkpoint.
const SEED = 17;
const INITIAL_STEPS = 5000;
const TRAIN_STEPS = 5000;
const HORIZON = 6;
const engine = new WorldEngine();
const stage = document.querySelector<HTMLCanvasElement>('#stage')!;
const ctx = stage.getContext('2d')!;
const out = document.querySelector<HTMLPreElement>('#out')!;
const train = document.querySelector<HTMLButtonElement>('#train')!;
const plan = document.querySelector<HTMLButtonElement>('#plan')!;
const apply = document.querySelector<HTMLButtonElement>('#apply')!;
const reset = document.querySelector<HTMLButtonElement>('#reset')!;
const goalSelect = document.querySelector<HTMLSelectElement>('#goal')!;
const hold = document.querySelector<HTMLInputElement>('#hold')!;
const goals: ArmState[] = [
	{ q1: -0.9, q2: 1.0, v1: 0, v2: 0 },
	{ q1: -1.8, q2: 1.8, v1: 0, v2: 0 },
	{ q1: -1.05, q2: 2.25, v1: 0, v2: 0 }
];
let state = initialArm();
let previous = { ...state };
let previousAction: Action = [0, 0];
let candidate: WorldPlan | null = null;
let busy = true;
let step = 0;
let header = '';
let dt = 0;
let time = 0;

function controls() {
	train.disabled = busy;
	plan.disabled = busy || step === 0;
	apply.disabled = busy || !candidate;
	reset.disabled = busy;
	goalSelect.disabled = busy;
	hold.disabled = busy;
}

function goal() {
	return goals[Number(goalSelect.value)];
}

function drawArm(pose: ArmState, color: string, alpha = 1, dashed = false) {
	const p = armPoints(pose);
	const scale = stage.height * 0.44;
	const x = (v: number) => stage.width / 2 + v * scale;
	const y = (v: number) => stage.height / 2 - v * scale;
	ctx.save();
	ctx.globalAlpha = alpha;
	ctx.strokeStyle = color;
	ctx.lineWidth = dashed ? 3 : 7;
	ctx.lineCap = 'round';
	ctx.lineJoin = 'round';
	if (dashed) ctx.setLineDash([8, 7]);
	ctx.beginPath();
	ctx.moveTo(x(p.base.x), y(p.base.y));
	ctx.lineTo(x(p.elbow.x), y(p.elbow.y));
	ctx.lineTo(x(p.tip.x), y(p.tip.y));
	ctx.stroke();
	ctx.setLineDash([]);
	ctx.fillStyle = color;
	for (const joint of [p.base, p.elbow, p.tip]) {
		ctx.beginPath();
		ctx.arc(x(joint.x), y(joint.y), 5, 0, Math.PI * 2);
		ctx.fill();
	}
	ctx.restore();
}

function draw() {
	const css = getComputedStyle(document.documentElement);
	const color = (name: string) => css.getPropertyValue(name).trim();
	ctx.fillStyle = color('--paper');
	ctx.fillRect(0, 0, stage.width, stage.height);
	ctx.strokeStyle = color('--line');
	ctx.lineWidth = 1;
	ctx.beginPath();
	ctx.arc(stage.width / 2, stage.height / 2, stage.height * 0.38, 0, Math.PI * 2);
	ctx.stroke();
	if (candidate)
		candidate.poses.forEach((pose, i) =>
			drawArm(pose, color('--accent'), 0.07 + (0.3 * (i + 1)) / candidate!.poses.length)
		);
	drawArm(goal(), color('--warm'), 0.85, true);
	drawArm(state, color('--ink'));
}

function observations() {
	const pixels = new Float32Array(2 * SENSOR_SIZE ** 2);
	pixels.set(renderSensor(previous));
	pixels.set(renderSensor(state), SENSOR_SIZE ** 2);
	return pixels;
}

async function task(fn: () => Promise<void>) {
	busy = true;
	controls();
	try {
		await fn();
	} catch (error) {
		out.textContent = `${header}\nerror: ${error instanceof Error ? error.message : String(error)}`;
	} finally {
		busy = false;
		controls();
		draw();
	}
}

async function learn(steps: number) {
	candidate = null;
	await engine.train(steps, (metrics) => {
		step = metrics.step;
		out.textContent = `${header}\nstep ${step} · prediction ${metrics.predictionLoss.toFixed(5)} · SIGReg ${metrics.regularizer.toFixed(3)}\n${metrics.stepMs.toFixed(1)} ms / step · ${(metrics.trainingMs / 1000).toFixed(1)} s training`;
	});
	const evaluation = await engine.evaluate();
	out.textContent += `\nHeld-out prediction ${evaluation.predictionLoss.toFixed(5)} · persistence ${evaluation.persistenceLoss.toFixed(5)}\nEmbedding spread ${evaluation.spread.toFixed(3)} · covariance participation ratio ${evaluation.effectiveRank.toFixed(2)}\nWeights are now fixed. Consider a move, or continue training.`;
}

train.onclick = () => void task(() => learn(TRAIN_STEPS));
plan.onclick = () =>
	void task(async () => {
		out.textContent = `${header}\nstep ${step} · weights fixed\nFitting the diagnostic readout and considering torque sequences…`;
		candidate = await engine.plan({
			observations: observations(),
			previousAction,
			goal: renderSensor(goal()),
			horizon: HORIZON,
			hold: hold.checked,
			seed: 300 + Math.round(time / dt)
		});
		out.textContent = `${header}\nstep ${step} · weights fixed\nPredicted cost ${candidate.cost.toFixed(5)} · search ${candidate.ms.toFixed(0)} ms\nSeparate readout test error ${candidate.readoutError.toFixed(3)} rad\nActual joint error ${poseError(state, goal()).toFixed(3)} rad\nFirst command [${candidate.action.map((v) => v.toFixed(3)).join(', ')}]. Apply it, then observe and plan again.`;
	});
apply.onclick = () => {
	if (!candidate || busy) return;
	previous = { ...state };
	previousAction = candidate.action;
	state = stepArm(state, previousAction, { dt });
	time += dt;
	candidate = null;
	out.textContent = `${header}\nstep ${step} · weights fixed\nFirst action applied. Actual joint error ${poseError(state, goal()).toFixed(3)} rad.\nConsider a new move from the fresh pair of observations.`;
	controls();
	draw();
};
reset.onclick = () => {
	state = initialArm();
	previous = { ...state };
	previousAction = [0, 0];
	time = 0;
	candidate = null;
	controls();
	draw();
};
goalSelect.onchange = hold.onchange = () => {
	candidate = null;
	controls();
	draw();
};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
window.addEventListener('pagehide', () => void engine.dispose());
draw();
void task(async () => {
	const info = await engine.init({ seed: SEED });
	dt = info.dt;
	header = `${info.backend} · ${info.parameters.toLocaleString()} parameters · ${info.transitions.toLocaleString()} local training transitions`;
	await learn(INITIAL_STEPS);
});
