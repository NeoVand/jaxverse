<script lang="ts">
	import { renderSensor } from '$lib/world/sensor';
	import { initialArm, stepArm } from '$lib/world/simulator';

	// Illustrative observations come from the same sensor and mechanics as the lab.
	// The eight-cell glyphs denote vector widths; they do not report learned values.
	const first = { ...initialArm(), v1: 0.4, v2: -0.35 };
	const second = stepArm(first, [0.25, -0.3], { dt: 0.24 });
	const third = stepArm(second, [-0.15, 0.35], { dt: 0.24 });
	const frames = [first, second, third].map((pose) =>
		Array.from(renderSensor(pose), (value, index) => ({
			x: index % 32,
			y: Math.floor(index / 32),
			ink: 1 - value
		})).filter((pixel) => pixel.ink > 0.005)
	);
	const cells = Array.from({ length: 8 }, (_, index) => index);
	const uid = $props.id();
</script>

{#snippet definitions(suffix: string)}
	<defs>
		<marker
			id={`${uid}-${suffix}`}
			viewBox="0 0 8 8"
			refX="6.5"
			refY="4"
			markerWidth="6"
			markerHeight="6"
			orient="auto-start-reverse"
		>
			<path d="M 0 0.8 L 7 4 L 0 7.2 Z" fill="var(--ink-3)" />
		</marker>
	</defs>
{/snippet}

{#snippet arrow(path: string, suffix: string)}
	<path d={path} class="arrow" marker-end={`url(#${uid}-${suffix})`} />
{/snippet}

{#snippet observation(x: number, y: number, index: number, size = 52)}
	<g transform={`translate(${x} ${y})`}>
		<rect
			width={size}
			height={size}
			rx="4"
			fill="var(--paper)"
			stroke="var(--warm)"
			stroke-opacity="0.65"
		/>
		<g transform={`scale(${size / 32})`}>
			{#each frames[index] as pixel (`${pixel.x}:${pixel.y}`)}
				<rect x={pixel.x} y={pixel.y} width="1" height="1" fill="var(--ink)" opacity={pixel.ink} />
			{/each}
		</g>
	</g>
{/snippet}

{#snippet vector(x: number, y: number, label: string, prediction = false)}
	<g transform={`translate(${x} ${y})`}>
		{#each cells as index (index)}
			<rect
				x={index * 7}
				width="4.5"
				height="22"
				rx="1.5"
				fill={prediction ? 'var(--good)' : 'var(--accent)'}
				opacity="0.68"
			/>
		{/each}
		<text
			x="26.75"
			y="44"
			text-anchor="middle"
			class={['symbol', prediction ? 'answer' : 'learned']}>{label}</text
		>
	</g>
{/snippet}

<figure class="architecture" aria-labelledby={`${uid}-title`}>
	<header>
		<p class="eyebrow">Inside the model</p>
		<h3 id={`${uid}-title`}>One encoder, on both sides of a prediction</h3>
	</header>

	<svg
		class="wide"
		viewBox="0 0 830 452"
		role="img"
		aria-label="Training architecture. Two context images pass separately through one shared encoder E theta, producing two eight-coordinate embeddings. These embeddings and two torque pairs enter predictor P phi. Its predicted next embedding is compared with the actual next image encoded by that same encoder. Prediction error and a weighted SIGReg penalty on all three time positions are added. Both branches of the encoder receive gradients."
	>
		{@render definitions('wide')}
		<text x="30" y="25" class="label world">two context pictures</text>
		{@render observation(30, 56, 0)}
		<text x="56" y="128" class="symbol world" text-anchor="middle">oₜ₋₁</text>
		{@render observation(30, 151, 1)}
		<text x="56" y="223" class="symbol world" text-anchor="middle">oₜ</text>
		{@render arrow('M 90 82 H 142', 'wide')}
		{@render arrow('M 90 177 H 142', 'wide')}

		<rect x="142" y="45" width="130" height="182" rx="8" class="encoder" />
		<text x="207" y="85" class="label learned" text-anchor="middle">shared encoder</text>
		<text x="207" y="123" class="function learned" text-anchor="middle"
			>E<tspan class="subscript" dy="6">θ</tspan></text
		>
		<text x="207" y="166" class="cap learned" text-anchor="middle">1,024 → 128 → 8</text>
		<text x="207" y="193" class="cap" text-anchor="middle">once per picture</text>
		{@render arrow('M 278 82 H 308', 'wide')}
		{@render arrow('M 278 177 H 308', 'wide')}
		{@render vector(311, 71, 'zₜ₋₁')}
		{@render vector(311, 166, 'zₜ')}
		<path d="M 372 82 H 390 V 177 H 372" class="arrow" />
		{@render arrow('M 390 159 H 418', 'wide')}

		<rect x="410" y="53" width="148" height="34" rx="17" class="action" />
		<text x="484" y="76" class="symbol world" text-anchor="middle">aₜ₋₁, aₜ</text>
		<text x="484" y="35" class="cap" text-anchor="middle">two recorded torque pairs</text>
		{@render arrow('M 484 87 V 116', 'wide')}
		<rect x="418" y="116" width="132" height="88" rx="8" class="predictor" />
		<text x="484" y="148" class="function learned-2" text-anchor="middle"
			>P<tspan class="subscript" dy="6">φ</tspan></text
		>
		<text x="484" y="170" class="label learned-2" text-anchor="middle">residual predictor</text>
		<text x="484" y="189" class="cap learned-2" text-anchor="middle">20 → 128 → 128 → 8</text>
		{@render arrow('M 556 159 H 590', 'wide')}
		{@render vector(592, 148, 'ẑₜ₊₁', true)}
		{@render arrow('M 652 159 H 742 V 233', 'wide')}

		<text x="30" y="263" class="label world">next picture</text>
		{@render observation(30, 280, 2)}
		<text x="56" y="357" class="symbol world" text-anchor="middle">oₜ₊₁</text>
		{@render arrow('M 90 306 H 142', 'wide')}
		<path d="M 207 232 V 267" class="shared" />
		<rect x="142" y="270" width="130" height="77" rx="8" class="encoder" />
		<text x="207" y="302" class="function learned" text-anchor="middle"
			>E<tspan class="subscript" dy="6">θ</tspan></text
		>
		<text x="207" y="329" class="cap learned" text-anchor="middle">the same weights</text>
		{@render arrow('M 278 306 H 308', 'wide')}
		{@render vector(311, 295, 'zₜ₊₁')}
		{@render arrow('M 372 306 H 681', 'wide')}
		<text x="519" y="291" class="cap learned" text-anchor="middle"
			>learned target · gradients flow here too</text
		>
		<rect x="681" y="233" width="122" height="93" rx="8" class="operator" />
		<text x="742" y="265" class="label fixed" text-anchor="middle">prediction error</text>
		<text x="742" y="293" class="cap fixed" text-anchor="middle">mean square</text>

		<path d="M 30 379 H 803" stroke="var(--line-soft)" />
		<text x="174" y="412" class="symbol learned">zₜ₋₁, zₜ, zₜ₊₁</text>
		{@render arrow('M 331 405 H 409', 'wide')}
		<rect x="410" y="389" width="156" height="36" rx="18" class="operator" />
		<text x="488" y="412" class="label fixed" text-anchor="middle">SIGReg</text>
		<text x="488" y="445" class="cap" text-anchor="middle">across examples, at each time</text>
		{@render arrow('M 571 407 H 681', 'wide')}
		<rect x="601" y="393" width="41" height="25" fill="var(--paper)" />
		<text x="621" y="412" class="symbol knob" text-anchor="middle">× λ</text>
		{@render arrow('M 742 332 V 389', 'wide')}
		<rect x="681" y="389" width="122" height="36" rx="18" class="operator" />
		<text x="742" y="412" class="label fixed" text-anchor="middle">total loss</text>
	</svg>

	<svg
		class="narrow"
		viewBox="0 0 360 616"
		role="img"
		aria-label="Training architecture. The two context pictures and next picture use the same learned encoder. Two context embeddings and two actions enter the predictor. Its answer is compared with the next picture's embedding, which is a learned target, not a predictor input. The objective adds prediction error to weighted SIGReg across independent examples at each of the three time positions. Both encoder branches receive gradients."
	>
		{@render definitions('narrow')}
		<text x="107" y="19" class="label world" text-anchor="middle">context pictures</text>
		<text x="280" y="19" class="label world" text-anchor="middle">next picture</text>
		{@render observation(38, 35, 0, 48)}
		{@render observation(122, 35, 1, 48)}
		{@render observation(256, 35, 2, 48)}
		<text x="62" y="106" class="symbol world" text-anchor="middle">oₜ₋₁</text>
		<text x="146" y="106" class="symbol world" text-anchor="middle">oₜ</text>
		<text x="280" y="106" class="symbol world" text-anchor="middle">oₜ₊₁</text>
		{@render arrow('M 62 112 V 133', 'narrow')}
		{@render arrow('M 146 112 V 133', 'narrow')}
		{@render arrow('M 280 112 V 133', 'narrow')}
		<rect x="28" y="133" width="152" height="67" rx="8" class="encoder" />
		<rect x="218" y="133" width="124" height="67" rx="8" class="encoder" />
		<text x="104" y="162" class="function learned" text-anchor="middle"
			>E<tspan class="subscript" dy="6">θ</tspan></text
		>
		<text x="104" y="185" class="cap learned" text-anchor="middle">1,024 → 128 → 8</text>
		<text x="280" y="162" class="function learned" text-anchor="middle"
			>E<tspan class="subscript" dy="6">θ</tspan></text
		>
		<text x="280" y="185" class="cap learned" text-anchor="middle">same weights</text>
		<path d="M 184 166 H 214" class="shared" />
		{@render arrow('M 62 205 V 228', 'narrow')}
		{@render arrow('M 146 205 V 228', 'narrow')}
		{@render arrow('M 280 205 V 228', 'narrow')}
		{@render vector(35, 233, 'zₜ₋₁')}
		{@render vector(119, 233, 'zₜ')}
		{@render vector(253, 233, 'zₜ₊₁')}
		<text x="280" y="304" class="cap learned" text-anchor="middle">learned target</text>
		<text x="280" y="320" class="cap" text-anchor="middle">gradients flow here too</text>
		<path d="M 62 283 V 304 H 145 V 283" class="arrow" />
		{@render arrow('M 104 304 V 344', 'narrow')}
		<rect x="28" y="344" width="152" height="72" rx="8" class="predictor" />
		<text x="104" y="374" class="function learned-2" text-anchor="middle"
			>P<tspan class="subscript" dy="6">φ</tspan></text
		>
		<text x="104" y="399" class="cap learned-2" text-anchor="middle">20 → 128 → 128 → 8</text>
		<rect x="218" y="352" width="124" height="34" rx="17" class="action" />
		<text x="280" y="375" class="symbol world" text-anchor="middle">aₜ₋₁, aₜ</text>
		{@render arrow('M 215 369 H 184', 'narrow')}
		{@render arrow('M 104 421 V 445', 'narrow')}
		{@render vector(77, 450, 'ẑₜ₊₁', true)}
		{@render arrow('M 313 244 H 352 V 474 H 342', 'narrow')}
		{@render arrow('M 141 461 H 214', 'narrow')}
		<rect x="218" y="440" width="124" height="58" rx="8" class="operator" />
		<text x="280" y="466" class="label fixed" text-anchor="middle">prediction error</text>
		<text x="280" y="485" class="cap fixed" text-anchor="middle">mean square</text>
		<path d="M 28 521 H 342" stroke="var(--line-soft)" />
		<text x="28" y="550" class="symbol learned">zₜ₋₁, zₜ, zₜ₊₁</text>
		{@render arrow('M 172 545 H 214', 'narrow')}
		<rect x="218" y="529" width="124" height="34" rx="17" class="operator" />
		<text x="280" y="552" class="label fixed" text-anchor="middle">SIGReg</text>
		<text x="185" y="588" class="cap" text-anchor="middle"
			>total loss = <tspan class="fixed">prediction error</tspan> + <tspan class="knob">λ</tspan> ×
			<tspan class="fixed">SIGReg</tspan></text
		>
		<text x="185" y="608" class="cap" text-anchor="middle"
			>distribution tested across examples, at each time</text
		>
	</svg>

	<figcaption>
		The predictor adds a learned change to the current embedding, zₜ. The future picture supplies a
		target only during training. Its embedding is learned too: prediction error updates both <span
			class="eq-model">encoder branches</span
		>
		and the
		<span class="eq-model-2">predictor</span>. SIGReg acts on the three sets of encoded
		observations, penalizing an encoder that erases their differences.
	</figcaption>
</figure>

<style>
	.architecture {
		margin-block: 32px 40px;
		padding-block: 26px 24px;
		border-block: 1px solid var(--line);
	}
	header {
		text-align: center;
		margin-bottom: 20px;
	}
	header p {
		margin: 0 0 10px;
	}
	h3 {
		margin: 0;
		font: italic 25px/1.25 var(--font-serif);
		color: var(--ink);
	}
	svg {
		display: block;
		width: 100%;
		max-width: 980px;
		margin: auto;
	}
	.narrow {
		display: none;
	}
	.arrow {
		fill: none;
		stroke: var(--ink-3);
		stroke-width: 1.1;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.shared {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1;
		stroke-dasharray: 3 3;
		opacity: 0.6;
	}
	.encoder {
		fill: var(--accent-soft);
		stroke: var(--accent);
		stroke-opacity: 0.6;
	}
	.predictor {
		fill: color-mix(in srgb, var(--cat-8) 6%, var(--paper));
		stroke: var(--cat-8);
		stroke-opacity: 0.6;
	}
	.operator {
		fill: color-mix(in srgb, var(--cat-2) 5%, var(--paper));
		stroke: var(--cat-2);
		stroke-opacity: 0.6;
	}
	.action {
		fill: var(--warm-soft);
		stroke: var(--warm);
		stroke-opacity: 0.6;
	}
	.label {
		font: italic 15px var(--font-serif);
		fill: var(--ink-2);
	}
	.symbol {
		font: italic 19px var(--font-serif);
		fill: var(--ink-2);
	}
	.function {
		font: italic 27px var(--font-serif);
	}
	.subscript {
		font-size: 65%;
	}
	.cap {
		font: 10px var(--font-mono);
		fill: var(--ink-3);
	}
	.world {
		fill: var(--warm);
	}
	.learned {
		fill: var(--accent);
	}
	.learned-2 {
		fill: var(--cat-8);
	}
	.answer {
		fill: var(--good);
	}
	.fixed {
		fill: var(--cat-2);
	}
	.knob {
		fill: var(--cat-1);
	}
	figcaption {
		max-width: 650px;
		margin: 20px auto 0;
		padding-inline: 20px;
		color: var(--ink-2);
		font: 12px/1.7 var(--font-sans);
		text-align: center;
	}
	@media (max-width: 620px) {
		.wide {
			display: none;
		}
		.narrow {
			display: block;
			max-width: 390px;
		}
		h3 {
			max-width: 280px;
			margin: auto;
			font-size: 23px;
		}
		.cap {
			font-size: 10px;
		}
		figcaption {
			text-align: left;
		}
	}
</style>
