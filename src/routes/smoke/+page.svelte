<script lang="ts">
	// Hidden engine smoke test (not linked anywhere): trains a small MLP on
	// spirals and prints every stage of the worker contract. Dev tool only.
	import { MlpEngine } from '$lib/nn/mlp-engine';
	import { makeDataset2d } from '$lib/nn/datasets2d';

	let log = $state<string[]>([]);
	let running = $state(false);
	const say = (s: string) => {
		log = [...log, s];
		console.log('[smoke]', s);
	};

	async function run() {
		running = true;
		log = [];
		const engine = new MlpEngine();
		try {
			const data = makeDataset2d('spirals', 400, 0.05, 7);
			say('init…');
			const t0 = performance.now();
			await engine.init(
				{ layers: [2, 16, 16, 2, 2], activation: 'tanh', loss: 'xent', lr: 5e-3, batchSize: 64 },
				{ x: data.x, y: data.labels, n: data.n }
			);
			say(
				`device=${engine.device} params=${engine.paramCount} (${Math.round(performance.now() - t0)}ms)`
			);

			let first = 0;
			let last = 0;
			await engine.train(300, (m) => {
				if (m.step === 1) first = m.loss;
				last = m.loss;
				if (m.step % 100 === 0)
					say(`step ${m.step} loss=${m.loss.toFixed(4)} (${m.stepMs.toFixed(0)}ms/step)`);
			});
			say(
				`loss ${first.toFixed(3)} → ${last.toFixed(3)} ${last < first ? '✓ learning' : '✗ NOT learning'}`
			);

			const ev = await engine.eval();
			say(`eval loss=${ev.loss.toFixed(4)} acc=${((ev.accuracy ?? 0) * 100).toFixed(1)}%`);

			const grid = new Float32Array(2 * 100);
			for (let i = 0; i < 100; i++) {
				grid[2 * i] = (i % 10) / 5 - 1;
				grid[2 * i + 1] = Math.floor(i / 10) / 5 - 1;
			}
			const pred = await engine.predict(grid, 100);
			say(`predict ok: ${pred.length} values, sample=${pred[0].toFixed(3)},${pred[1].toFixed(3)}`);

			const acts = await engine.activations(grid, 100);
			say(`activations ok: ${acts.layers.length} layers, widths=${acts.widths.join(',')}`);

			const h = acts.layers[2].slice(0, 2 * 100); // the 2-wide layer
			const tail = await engine.forwardFrom(3, h, 100);
			say(`forwardFrom ok: ${tail.length} values`);

			const g = await engine.inputGrad(grid.slice(0, 2), 1);
			say(`inputGrad ok: [${g[0].toExponential(2)}, ${g[1].toExponential(2)}]`);

			const w = await engine.weights();
			say(`weights ok: ${w.map((l) => `${l.inDim}×${l.outDim}`).join(' ')}`);

			const ck = await engine.exportCheckpoint();
			say(`export ok: ${ck.byteLength} bytes`);

			await engine.reset(99);
			const ev2 = await engine.eval();
			say(`reset ok: acc back to ${((ev2.accuracy ?? 0) * 100).toFixed(1)}%`);
			say('ALL OK');
		} catch (err) {
			say(`ERROR: ${err instanceof Error ? err.message : String(err)}`);
		} finally {
			await engine.dispose();
			running = false;
		}
	}
</script>

<div class="mx-auto max-w-2xl px-5 py-16">
	<button class="rounded border border-line px-4 py-2" onclick={run} disabled={running}
		>run smoke test</button
	>
	<pre class="num mt-6 text-[12px] leading-relaxed">{log.join('\n')}</pre>
</div>
