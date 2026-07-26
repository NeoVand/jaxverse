<script lang="ts">
	// The hourglass, drawn once and held still: a digit goes in, the layers
	// narrow to a two-number waist, and a mirror-image stack tries to repaint
	// the digit from those two numbers alone. Static by design — the plates
	// below animate; this is the shape the reader keeps in mind while they do.

	const CY = 88; // the spine every layer is centred on
	// bar half-height ∝ √width, so 784 and 64 can share one glyph
	const hh = (w: number) => Math.round(Math.sqrt(w) * 1.5);
	const ENC = [
		{ x: 104, w: 784 },
		{ x: 148, w: 256 },
		{ x: 192, w: 64 }
	];
	const DEC = [
		{ x: 300, w: 64 },
		{ x: 344, w: 256 },
		{ x: 388, w: 784 }
	];
	const BAR = 12;
	const WX = 246; // the waist

	// the hourglass silhouette, traced along the outer edges of the bars
	const SIL = [
		`M 104 ${CY - hh(784)}`,
		`L 204 ${CY - hh(64)}`,
		`L 240 ${CY - 14} L 252 ${CY - 14}`,
		`L 300 ${CY - hh(64)}`,
		`L 400 ${CY - hh(784)}`,
		`L 400 ${CY + hh(784)}`,
		`L 300 ${CY + hh(64)}`,
		`L 252 ${CY + 14} L 240 ${CY + 14}`,
		`L 204 ${CY + hh(64)}`,
		`L 104 ${CY + hh(784)} Z`
	].join(' ');

	// one hand-drawn 3, printed twice: once as ink, once as the decoder's blur
	const THREE =
		'M 20 74 C 26 64, 46 66, 44 78 C 43 86, 34 88, 30 88 C 40 87, 48 92, 46 102 C 44 112, 26 113, 19 106';
</script>

<figure class="my-8">
	<svg
		viewBox="0 0 520 182"
		class="mx-auto block w-full max-w-[660px]"
		role="img"
		aria-label="An autoencoder as an hourglass. A 784-pixel digit enters on the left, passes through layers of 784, 256 and 64 units that narrow to a bottleneck of two numbers, and a mirror-image decoder of 64, 256 and 784 units repaints the digit on the right. The bottleneck is variational: the encoder proposes a Gaussian, the decoder is handed a sample of it, and that Gaussian is kept near the standard normal. The loss is the distance between the digit that went in and the one that came out."
	>
		<defs>
			<marker
				id="ae-arrow"
				viewBox="0 0 8 8"
				refX="6.5"
				refY="4"
				markerWidth="6"
				markerHeight="6"
				orient="auto-start-reverse"
			>
				<path d="M 0 0.8 L 7 4 L 0 7.2 Z" fill="var(--ink-3)" />
			</marker>
			<filter id="ae-soft" x="-30%" y="-30%" width="160%" height="160%">
				<feGaussianBlur stdDeviation="2.1" />
			</filter>
		</defs>

		<!-- the loss: a bracket between the two ends, because that is all it
		     compares. It turns the corner well above the layer labels. -->
		<text x="259" y="15" text-anchor="middle" class="cap"> the loss · ‖ x − D(E(x)) ‖² </text>
		<path
			d="M 34 54 L 34 34 Q 34 22 46 22 L 472 22 Q 484 22 484 34 L 484 54"
			fill="none"
			stroke="var(--ink-3)"
			stroke-width="1"
			stroke-dasharray="3 3"
			opacity="0.7"
		/>
		<circle cx="34" cy="54" r="1.7" fill="var(--ink-3)" />
		<circle cx="484" cy="54" r="1.7" fill="var(--ink-3)" />

		<!-- the digit as it arrives, and as it comes back -->
		{#each [{ dx: 0, blur: false, name: 'x', sub: '28 × 28 pixels' }, { dx: 450, blur: true, name: 'D(E(x))', sub: 'the rebuild' }] as tile (tile.dx)}
			<rect
				x={6 + tile.dx}
				y="60"
				width="56"
				height="56"
				rx="5"
				fill="var(--surface)"
				stroke="var(--line)"
			/>
			{#each [1, 2, 3, 4, 5, 6] as k (k)}
				<line
					x1={6 + tile.dx + k * 8}
					y1="60"
					x2={6 + tile.dx + k * 8}
					y2="116"
					stroke="var(--line-soft)"
					stroke-width="0.6"
				/>
				<line
					x1={6 + tile.dx}
					y1={60 + k * 8}
					x2={62 + tile.dx}
					y2={60 + k * 8}
					stroke="var(--line-soft)"
					stroke-width="0.6"
				/>
			{/each}
			<path
				d={THREE}
				transform="translate({tile.dx} 0)"
				fill="none"
				stroke="var(--ink)"
				stroke-width={tile.blur ? 6 : 5}
				stroke-linecap="round"
				opacity={tile.blur ? 0.72 : 0.85}
				filter={tile.blur ? 'url(#ae-soft)' : undefined}
			/>
			<text x={34 + tile.dx} y="130" text-anchor="middle" class="cap">{tile.name}</text>
			<text x={34 + tile.dx} y="141" text-anchor="middle" class="cap dim">{tile.sub}</text>
		{/each}

		<line
			x1="68"
			y1={CY}
			x2="98"
			y2={CY}
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#ae-arrow)"
		/>
		<line
			x1="406"
			y1={CY}
			x2="450"
			y2={CY}
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#ae-arrow)"
		/>

		<!-- the hourglass itself -->
		<path d={SIL} fill="var(--ink-3)" fill-opacity="0.06" stroke="none" />
		{#each [...ENC, ...DEC] as g (g.x)}
			<rect
				x={g.x}
				y={CY - hh(g.w)}
				width={BAR}
				height={2 * hh(g.w)}
				rx="1.5"
				fill="none"
				stroke="var(--ink-3)"
				stroke-width="1"
			/>
			<text x={g.x + BAR / 2} y={CY - hh(g.w) - 5} text-anchor="middle" class="tiny">{g.w}</text>
		{/each}

		<!-- the waist: two numbers, and nothing else gets through -->
		<rect
			x={WX - 6}
			y={CY - 15}
			width="12"
			height="30"
			rx="6"
			fill="var(--surface)"
			stroke="var(--accent)"
			stroke-width="1.2"
		/>
		<circle cx={WX} cy={CY - 6} r="3.1" fill="var(--accent)" />
		<circle cx={WX} cy={CY + 6} r="3.1" fill="var(--accent)" />
		<text x={WX} y={CY - 21} text-anchor="middle" class="tiny accent">2</text>

		<line
			x1={WX}
			y1={CY + 19}
			x2={WX}
			y2="134"
			stroke="var(--accent)"
			stroke-width="1"
			opacity="0.6"
		/>
		<text x={WX} y="149" text-anchor="middle" class="label accent">the bottleneck</text>
		<text x={WX} y="161" text-anchor="middle" class="cap dim">z = μ + σ ε</text>
		<text x={WX} y="172" text-anchor="middle" class="cap dim">kept near N(0, 1)</text>

		<text x="140" y="149" text-anchor="middle" class="label">encoder E</text>
		<text x="140" y="161" text-anchor="middle" class="cap dim">squeeze</text>
		<text x="364" y="149" text-anchor="middle" class="label">decoder D</text>
		<text x="364" y="161" text-anchor="middle" class="cap dim">repaint</text>
	</svg>
	<figcaption
		class="mx-auto mt-2 max-w-[560px] text-center font-serif text-[13.5px] text-ink-2 italic"
		style="font-variation-settings: 'opsz' 13;"
	>
		Two ordinary stacks of layers, joined at a waist too narrow to cheat through. Nothing in the
		picture knows what a 3 is; the only pressure on it is that the digit on the right should look
		like the digit on the left, and the only route between them is those two numbers — which arrive
		with a little noise on them, for reasons the next paragraphs make good on.
	</figcaption>
</figure>

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
	.tiny {
		font-family: var(--font-mono);
		font-size: 7.5px;
		fill: var(--ink-3);
	}
	.dim {
		fill: var(--ink-3);
	}
	.accent {
		fill: var(--accent);
	}
</style>
