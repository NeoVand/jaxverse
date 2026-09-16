/** Browser-only research harness. Import from the Vite origin and call runWorldBench(). */
import { init, defaultDevice } from '@jax-js/jax';
import { WorldCore, WORLD_TRAINING_STEPS, type WorldInit } from '../src/lib/world/engine';
import { renderSensor } from '../src/lib/world/sensor';
import { stepArm, poseError, initialArm, type ArmState } from '../src/lib/world/simulator';

export async function runWorldBench(
	options: WorldInit & {
		steps?: number;
		control?: boolean;
		horizon?: number;
		suite?: boolean;
		holdHorizon?: number;
		gentleCurl?: boolean;
		effort?: number;
	} = {}
) {
	const devices = await init();
	const backend = devices.includes('webgpu') ? 'webgpu' : 'wasm';
	defaultDevice(backend);
	const model = new WorldCore(options);
	const initStart = performance.now();
	const info = await model.init();
	const initMs = performance.now() - initStart;
	const resolution = info.config.resolution;
	const pixels = resolution ** 2;
	console.log(
		'WORLD info',
		JSON.stringify({
			backend,
			parameters: info.parameters,
			transitions: info.transitions,
			resolution,
			initMs
		})
	);
	const before = await model.evaluate();
	console.log('WORLD before', JSON.stringify({ ...before, projection: undefined }));
	const trainStart = performance.now();
	const training = await model.train(options.steps ?? WORLD_TRAINING_STEPS, (m) => {
		if (m.step % 1000 === 0 || m.step === 1) console.log('WORLD training', JSON.stringify(m));
	});
	const trainWallMs = performance.now() - trainStart;
	const readoutStart = performance.now();
	const readout = await model.fitReadout();
	const readoutMs = performance.now() - readoutStart;
	const after = await model.evaluate();
	const rollouts = await model.evaluateRollouts();
	console.log('WORLD after', JSON.stringify({ ...after, projection: undefined }));
	console.log('WORLD rollouts', JSON.stringify(rollouts));
	const controls = [];
	if (options.control !== false)
		for (let trial = 0; trial < (options.suite ? (options.gentleCurl ? 7 : 6) : 7); trial++) {
			let current: ArmState = { q1: -1 + trial * 0.65, q2: 0.8 - trial * 0.5, v1: 0, v2: 0 };
			if (options.suite || trial >= 4) current = initialArm();
			let previous = current;
			let previousAction: [number, number] = [0, 0];
			let goal: ArmState = { q1: current.q1 + 0.5, q2: current.q2 - 0.6, v1: 0, v2: 0 };
			if (options.suite || trial >= 4)
				goal = {
					...current,
					...[
						{ q1: -0.9, q2: 1 },
						{ q1: -1.8, q2: 1.8 },
						{ q1: -0.7, q2: 2.65 }
					][options.suite ? trial % 3 : trial - 4]
				};
			if (options.suite && options.gentleCurl && trial % 3 === 2)
				goal = { ...current, q1: -1.05, q2: 2.25 };
			if (options.suite && options.gentleCurl && trial === 6)
				goal = { ...current, q1: -0.7, q2: 2.65 };
			const initial = poseError(current, goal);
			const horizon = options.suite
				? trial < 3
					? 6
					: (options.holdHorizon ?? 12)
				: (options.horizon ?? 12);
			const hold = options.suite ? trial >= 3 : true;
			const errors = [],
				times = [];
			for (let t = 0; t < 40; t++) {
				const observations = new Float32Array(2 * pixels);
				observations.set(renderSensor(previous, resolution));
				observations.set(renderSensor(current, resolution), pixels);
				const plan = await model.plan({
					observations,
					previousAction,
					goal: renderSensor(goal, resolution),
					horizon,
					hold,
					seed: 300 + t,
					effort: options.effort
				});
				previous = current;
				current = stepArm(current, plan.action, { dt: info.dt });
				previousAction = plan.action;
				errors.push(poseError(current, goal));
				times.push(plan.ms);
			}
			const result = {
				trial,
				horizon,
				hold,
				initial,
				dwell: errors.some((_, i) => i >= 4 && errors.slice(i - 4, i + 1).every((e) => e <= 0.15)),
				final: errors.at(-1),
				best: Math.min(...errors),
				lastFive: errors.slice(-5),
				medianMs: times.toSorted((a, b) => a - b)[Math.floor(times.length / 2)]
			};
			console.log('WORLD control', JSON.stringify(result));
			controls.push(result);
		}
	model.dispose();
	return {
		backend,
		info: {
			parameters: info.parameters,
			config: info.config,
			transitions: info.transitions,
			validationTransitions: info.validationTransitions,
			dt: info.dt
		},
		timing: { initMs, trainWallMs, readoutMs },
		training,
		before,
		after,
		readout,
		rollouts,
		controls
	};
}
