<script lang="ts">
	import { base } from '$app/paths';
	import Plate from '$lib/components/ui/Plate.svelte';

	// Does a model that has only ever seen move strings know where the pieces
	// are? This plate reports the experiment that answers it, and the answer is
	// a picture: the board, read out of the model's own activations, layer by
	// layer, getting better as it goes.
	//
	// The results are precomputed, and they have to be — the probe is fitted on
	// a hundred thousand games against a Rook eight times the size of the one
	// this page trains, which is not a browser tab's afternoon. The caption
	// says so. What the page can do honestly is show the finding whole,
	// including the control, which is the half that makes it mean anything: the
	// same probe fitted to the same architecture with random weights.

	interface Example {
		ply: number;
		turn: string;
		/** 64 squares, oriented for the side to move: 0 empty, 1 mine, 2 theirs. */
		trueOriented: number[];
		/** The probe's reading of those 64 squares, per layer. */
		predByLayer: number[][];
	}
	interface ProbeData {
		config: { nLayer: number; nEmbd: number; params: number; games: number; valLoss: number };
		/** Accuracy of always answering "empty" — the number to beat. */
		baseline: number;
		layers: { trained: number; control: number }[];
		examples: Example[];
	}

	let data = $state<ProbeData | null>(null);
	let failed = $state(false);
	let pick = $state(0);

	$effect(() => {
		let live = true;
		fetch(`${base}/data/rook-probe.json`)
			.then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
			.then((d: ProbeData) => live && (data = d))
			.catch(() => live && (failed = true));
		return () => {
			live = false;
		};
	});

	const example = $derived(data?.examples[pick % (data?.examples.length || 1)] ?? null);

	/** Two runs of positions to iterate — the block numbers along the chart's
	 *  foot, and the chips that switch positions — neither of which needs the
	 *  value sitting at the index. */
	const blockNumbers = $derived(data?.layers.map((_, i) => i) ?? []);
	const exampleNumbers = $derived(data?.examples.map((_, i) => i) ?? []);

	// ── the boards, along the top ──
	const CELL = 9;
	const BOARD = CELL * 8;
	const BY = 28;
	const GAP = 12;
	/** Where the truth sits, and where the run of one-board-per-block starts. */
	const TRUTH_X = 10;
	const RUN_X = TRUTH_X + BOARD + 28;

	/** Empty, the side to move's own men, and the opponent's. The two occupied
	 *  classes take the book's two-class pair, as they do everywhere else. */
	const FILL = ['var(--surface-2)', 'var(--accent)', 'var(--warm)'] as const;

	// ── the chart, underneath them, on the same axis of depth ──
	const CX = RUN_X + BOARD / 2;
	const CY = 158;
	const CW = 5 * (BOARD + GAP);
	const CH = 104;
	const LO = 0.55;
	const HI = 0.83;

	/** Block k lines up under block k's board — the chart is the same journey
	 *  across the page, measured instead of drawn. */
	const chartX = (layer: number, n: number) => CX + (n === 1 ? CW / 2 : (layer / (n - 1)) * CW);
	const chartY = (a: number) => CY + ((HI - a) / (HI - LO)) * CH;

	function line(values: number[]): string {
		return values
			.map((v, i) => `${i === 0 ? 'M' : 'L'} ${chartX(i, values.length)} ${chartY(v).toFixed(2)}`)
			.join(' ');
	}

	const fmtPct = (v: number) => `${(100 * v).toFixed(0)}%`;
</script>

<Plate
	id="probe"
	title="Where the pieces are"
	caption="A measurement rather than a training run, and made offline: the probe is fitted on a hundred thousand games against a Rook of the same design but eight times the size. Fit the simplest readout there is — one linear layer, no bend — to the numbers running through the model at each block, and ask it for the occupancy of all sixty-four squares. Above: one position, then the board that readout returns from each block in turn, with a dot on every square it gets wrong. Below, on the same left-to-right axis of depth: how often it is right, against the two numbers that decide whether that means anything. Always answering 'empty' scores the dashed line. The same probe, fitted the same way to the same architecture with its weights left random, scores the grey one — a network that has learned nothing still carries some trace of the moves it was fed, and an honest claim has to clear that rather than merely the dashed line. What training adds is the gap between the grey line and the ultramarine one, and it widens with depth."
>
	<div class="px-4">
		{#if failed}
			<p class="num py-10 text-center text-[12px] text-ink-3">
				the probe results could not be loaded
			</p>
		{:else if !data || !example}
			<p class="num py-10 text-center text-[12px] text-ink-3">reading the probe results…</p>
		{:else}
			<svg
				viewBox="0 0 720 296"
				class="mx-auto block w-full max-w-[900px]"
				role="img"
				aria-label="Top: a chess position shown as an eight by eight grid of occupancy — empty, the side to move's pieces, and the opponent's — followed by six more grids, one per block of the model, each the board recovered from that block's activations by a linear probe, with a dot marking every square it read wrongly. Below, on the same left-to-right axis of depth: a chart of probe accuracy by block. The trained model's line rises from about 67 to about 79 per cent; the same probe on a random-weight control stays flat near 65; always answering empty scores 60."
			>
				<!-- ── the truth, then one board per block ── -->
				<text x={TRUTH_X} y="18" class="cap dim">the position</text>
				<text x={RUN_X} y="18" class="cap dim">read out of block …</text>

				{#each [{ squares: example.trueOriented, x: TRUTH_X, truth: true, label: 'truth' }, ...example.predByLayer.map( (sq, i) => ({ squares: sq, x: RUN_X + i * (BOARD + GAP), truth: false, label: String(i + 1) }) )] as board (board.x)}
					{#each board.squares as cls, i (i)}
						<rect
							x={board.x + (i % 8) * CELL}
							y={BY + Math.floor(i / 8) * CELL}
							width={CELL}
							height={CELL}
							fill={FILL[cls]}
						/>
						{#if !board.truth && cls !== example.trueOriented[i]}
							<circle
								cx={board.x + (i % 8) * CELL + CELL / 2}
								cy={BY + Math.floor(i / 8) * CELL + CELL / 2}
								r="1.5"
								fill="var(--ink)"
								opacity="0.62"
							/>
						{/if}
					{/each}
					<rect
						x={board.x}
						y={BY}
						width={BOARD}
						height={BOARD}
						fill="none"
						stroke="var(--line)"
						stroke-width="1"
					/>
					<text
						x={board.x + BOARD / 2}
						y={BY + BOARD + 13}
						text-anchor="middle"
						class="cap {board.truth ? '' : 'dim'}">{board.label}</text
					>
				{/each}

				<!-- ── the same journey, measured ── -->
				<text x={TRUTH_X} y={CY - 12} class="cap dim">how often it is right</text>

				<line
					x1={CX - 18}
					y1={chartY(data.baseline)}
					x2={CX + CW + 8}
					y2={chartY(data.baseline)}
					stroke="var(--ink-3)"
					stroke-width="1"
					stroke-dasharray="3 3"
					opacity="0.75"
				/>
				<path
					d={line(data.layers.map((l) => l.control))}
					fill="none"
					stroke="var(--ink-3)"
					stroke-width="1.4"
				/>
				<path
					d={line(data.layers.map((l) => l.trained))}
					fill="none"
					stroke="var(--accent)"
					stroke-width="1.8"
					stroke-linejoin="round"
				/>
				{#each data.layers as l, i (i)}
					<circle
						cx={chartX(i, data.layers.length)}
						cy={chartY(l.trained)}
						r="2.4"
						fill="var(--accent)"
					/>
				{/each}

				{#each [{ y: data.layers[data.layers.length - 1].trained, t: 'trained Rook', tone: '' }, { y: data.layers[data.layers.length - 1].control, t: 'random weights', tone: 'dim' }, { y: data.baseline, t: 'always “empty”', tone: 'dim' }] as k (k.t)}
					<text x={CX + CW + 14} y={chartY(k.y) + 3} class="cap {k.tone}"
						>{fmtPct(k.y)} · {k.t}</text
					>
				{/each}

				{#each blockNumbers as i (i)}
					<text
						x={chartX(i, data.layers.length)}
						y={CY + CH + 14}
						text-anchor="middle"
						class="cap dim">{i + 1}</text
					>
				{/each}
			</svg>

			<div class="mt-3 flex flex-wrap items-center justify-between gap-x-5 gap-y-2 px-1">
				<span class="flex flex-wrap items-center gap-x-4 gap-y-1">
					{#each [{ c: FILL[1], t: 'to move' }, { c: FILL[2], t: 'opponent' }, { c: FILL[0], t: 'empty' }] as k (k.t)}
						<span class="key">
							<span class="swatch" style="background: {k.c};"></span>{k.t}
						</span>
					{/each}
					<span class="key">
						<span class="swatch dot"></span>read wrongly
					</span>
				</span>

				<span class="flex items-center gap-1.5">
					<span class="eyebrow">position</span>
					{#each exampleNumbers as i (i)}
						<button
							type="button"
							class="chip"
							class:on={i === pick}
							onclick={() => (pick = i)}
							aria-pressed={i === pick}>{i + 1}</button
						>
					{/each}
				</span>
			</div>

			<p class="num mt-2.5 px-1 text-[10.5px] text-ink-3">
				probed model · {data.config.nLayer} blocks · {data.config.nEmbd} wide · {(
					data.config.params / 1e6
				).toFixed(1)}M parameters · {(data.config.games / 1000).toFixed(0)}k games · val loss {data.config.valLoss.toFixed(
					2
				)}
			</p>
		{/if}
	</div>
</Plate>

<style>
	.cap {
		font-family: var(--font-mono);
		font-size: 8.5px;
		fill: var(--ink-2);
	}
	.dim {
		fill: var(--ink-3);
	}
	.key {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		font-family: var(--font-sans);
		font-size: 9.5px;
		font-weight: 520;
		letter-spacing: 0.07em;
		text-transform: uppercase;
		color: var(--ink-3);
	}
	.swatch {
		display: inline-block;
		width: 9px;
		height: 9px;
		border: 1px solid var(--line);
	}
	.swatch.dot {
		position: relative;
		background: var(--surface-2);
	}
	.swatch.dot::after {
		position: absolute;
		inset: 2.5px;
		border-radius: 50%;
		background: var(--ink);
		opacity: 0.62;
		content: '';
	}
	.chip {
		font-family: var(--font-mono);
		font-size: 10.5px;
		line-height: 1;
		padding: 4px 7px;
		border: 1px solid var(--line);
		border-radius: 4px;
		color: var(--ink-3);
		background: transparent;
		cursor: pointer;
		transition:
			border-color 100ms ease,
			color 100ms ease;
	}
	.chip:hover {
		color: var(--ink-2);
		border-color: var(--ink-3);
	}
	.chip.on {
		color: var(--accent);
		border-color: var(--accent);
	}
</style>
