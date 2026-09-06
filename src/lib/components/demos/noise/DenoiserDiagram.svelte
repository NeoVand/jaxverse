<script lang="ts">
	// The denoiser, held still.
	//
	// The same transformer as chapter 5, with three things changed: the
	// sequence is image patches, the patches may all see each other (no causal
	// mask), and the noise level arrives as a scale and a shift applied inside
	// every block rather than as a token in the sequence.
	import Plate from '$lib/components/ui/Plate.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	// deterministic speckle, so the figure is the same picture every visit
	function noise(i: number, j: number, seed: number): number {
		const s = Math.sin(i * 127.1 + j * 311.7 + seed * 74.7) * 43758.5453;
		return s - Math.floor(s);
	}

	const P = 8; // patches across the toy tile
	const TILE = 72;
	const CELL = TILE / P;
	const grid = Array.from({ length: P * P }, (_, k) => ({
		x: k % P,
		y: Math.floor(k / P)
	}));
	const BLOCKS = [0, 1, 2, 3];
	const BX = 268;
	const BW = 40;
	const BGAP = 14;
</script>

<Plate id="denoiser" {title} {caption}>
	<div class="px-4 pt-1 pb-2 sm:px-5">
		<svg
			viewBox="0 0 660 252"
			class="mx-auto block w-full max-w-[950px]"
			role="img"
			aria-label="The denoiser drawn as a pipeline. A noisy 32-pixel square enters at the left and is cut into an 8 by 8 grid of four-by-four patches; each patch becomes one token. Four transformer blocks pass the tokens through attention and a small network, and every block is modulated by a scale and a shift computed from the noise level, which enters from above. The tokens are reassembled into a square at the right, and that square is the model's guess at which noise is present. Underneath, the update rule shows that guess being subtracted to give a slightly cleaner picture."
		>
			<defs>
				<marker
					id="dn-arrow"
					viewBox="0 0 8 8"
					refX="6.5"
					refY="4"
					markerWidth="6"
					markerHeight="6"
					orient="auto-start-reverse"
				>
					<path d="M 0 0.8 L 7 4 L 0 7.2 Z" fill="var(--ink-3)" />
				</marker>
				<marker
					id="dn-arrow-k"
					viewBox="0 0 8 8"
					refX="6.5"
					refY="4"
					markerWidth="5.5"
					markerHeight="5.5"
					orient="auto-start-reverse"
				>
					<path d="M 0 0.8 L 7 4 L 0 7.2 Z" fill="var(--cat-1)" />
				</marker>
			</defs>

			<!-- the conditioning, entering from above -->
			<rect
				x="250"
				y="10"
				width="250"
				height="21"
				rx="10.5"
				fill="var(--surface)"
				stroke="var(--cat-1)"
				stroke-width="1"
			/>
			<text x="375" y="24.5" text-anchor="middle" class="pill">
				the noise level <tspan class="sym">τ</tspan>
			</text>
			{#each BLOCKS as b (b)}
				<path
					d="M {BX + b * (BW + BGAP) + BW / 2} 31 L {BX + b * (BW + BGAP) + BW / 2} 62"
					stroke="var(--cat-1)"
					stroke-width="0.9"
					stroke-dasharray="3 2.5"
					fill="none"
					marker-end="url(#dn-arrow-k)"
				/>
			{/each}
			<text x="512" y="24.5" class="cap" style="fill: var(--cat-1);">scale · shift</text>

			<!-- input tile -->
			<g transform="translate(20, 62)">
				{#each grid as g (`${g.x}-${g.y}`)}
					<rect
						x={g.x * CELL}
						y={g.y * CELL}
						width={CELL}
						height={CELL}
						fill="var(--ink-3)"
						opacity={0.12 + noise(g.x, g.y, 3) * 0.75}
					/>
				{/each}
				<rect width={TILE} height={TILE} fill="none" stroke="var(--line)" stroke-width="1" rx="2" />
			</g>
			<text x="56" y="152" text-anchor="middle" class="label">
				<tspan class="sym">x</tspan><tspan dy="3" font-size="9">τ</tspan>
			</text>
			<text x="56" y="167" text-anchor="middle" class="cap dim">a ruined picture</text>

			<path
				d="M 100 98 L 128 98"
				stroke="var(--ink-3)"
				stroke-width="1"
				fill="none"
				marker-end="url(#dn-arrow)"
			/>

			<!-- patchify: the same square, now visibly cut up -->
			<g transform="translate(134, 62)">
				{#each grid as g (`p-${g.x}-${g.y}`)}
					<rect
						x={g.x * CELL + 0.6}
						y={g.y * CELL + 0.6}
						width={CELL - 1.2}
						height={CELL - 1.2}
						rx="1"
						fill="var(--accent)"
						opacity={0.14 + noise(g.x, g.y, 3) * 0.5}
					/>
				{/each}
			</g>
			<text x="170" y="152" text-anchor="middle" class="label">64 patches</text>
			<text x="170" y="167" text-anchor="middle" class="cap dim">one token each</text>

			<path
				d="M 214 98 L 242 98"
				stroke="var(--ink-3)"
				stroke-width="1"
				fill="none"
				marker-end="url(#dn-arrow)"
			/>

			<!-- the stack -->
			{#each BLOCKS as b (`b-${b}`)}
				<g transform="translate({BX + b * (BW + BGAP)}, 62)">
					<rect
						width={BW}
						height={TILE}
						rx="3"
						fill="var(--surface)"
						stroke="var(--line)"
						stroke-width="1"
					/>
					<rect
						x="7"
						y="10"
						width={BW - 14}
						height="22"
						rx="2"
						fill="var(--accent)"
						opacity="0.16"
					/>
					<text x={BW / 2} y="24.5" text-anchor="middle" class="cap">attn</text>
					<rect
						x="7"
						y="40"
						width={BW - 14}
						height="22"
						rx="2"
						fill="var(--cat-6)"
						opacity="0.16"
					/>
					<text x={BW / 2} y="54.5" text-anchor="middle" class="cap">mlp</text>
				</g>
				{#if b < 3}
					<path
						d="M {BX + b * (BW + BGAP) + BW} 98 L {BX + (b + 1) * (BW + BGAP) - 2} 98"
						stroke="var(--ink-3)"
						stroke-width="0.9"
						fill="none"
						marker-end="url(#dn-arrow)"
					/>
				{/if}
			{/each}
			<text x={BX + 2 * (BW + BGAP) - BGAP / 2} y="152" text-anchor="middle" class="label">
				four blocks
			</text>
			<text x={BX + 2 * (BW + BGAP) - BGAP / 2} y="167" text-anchor="middle" class="cap dim">
				every patch sees every other
			</text>

			<path
				d="M 464 98 L 492 98"
				stroke="var(--ink-3)"
				stroke-width="1"
				fill="none"
				marker-end="url(#dn-arrow)"
			/>

			<!-- output: the noise it claims is present -->
			<g transform="translate(498, 62)">
				{#each grid as g (`o-${g.x}-${g.y}`)}
					<rect
						x={g.x * CELL}
						y={g.y * CELL}
						width={CELL}
						height={CELL}
						fill="var(--good)"
						opacity={0.1 + noise(g.x, g.y, 8) * 0.62}
					/>
				{/each}
				<rect width={TILE} height={TILE} fill="none" stroke="var(--line)" stroke-width="1" rx="2" />
			</g>
			<text x="534" y="152" text-anchor="middle" class="label">
				<tspan class="sym">ε̂</tspan>
			</text>
			<text x="534" y="167" text-anchor="middle" class="cap dim">the noise it thinks is there</text>

			<!-- the update, on its own line -->
			<line x1="20" y1="192" x2="640" y2="192" stroke="var(--line-soft)" stroke-width="1" />
			<text x="20" y="214" class="label">and then the only thing you do with it:</text>
			<text x="298" y="215" class="eq">
				subtract a little of <tspan style="fill: var(--good);">ε̂</tspan> from
				<tspan style="fill: var(--ink);">x</tspan><tspan dy="3" font-size="8">τ</tspan><tspan
					dy="-3">, and repeat</tspan
				>
			</text>
			<text x="20" y="235" class="cap dim">
				the network is asked one question, at one noise level, and never sees the whole journey
			</text>
		</svg>
	</div>
</Plate>

<style>
	.label {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 12.5px;
		fill: var(--ink-2);
	}
	.cap {
		font-family: var(--font-mono);
		font-size: 8.5px;
		fill: var(--ink-2);
	}
	.dim {
		fill: var(--ink-3);
	}
	.pill {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 10.5px;
		fill: var(--cat-1);
	}
	.eq {
		font-family: var(--font-mono);
		font-size: 10.5px;
		fill: var(--ink-2);
	}
	.sym {
		font-style: italic;
	}
</style>
