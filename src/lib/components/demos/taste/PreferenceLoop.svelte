<script lang="ts">
	// Plate I — the chapter's spine, drawn once.
	//
	// A verdict becomes a judge; a judge becomes a policy; and the policy is
	// tethered to the model it started from. Three stations and one leash, in
	// the book's own colour constitution: what is learned is ultramarine, what
	// the world supplies is vermilion, what is frozen is plain ink, and the one
	// number you choose rather than learn is the knob colour.
	//
	// The two specimens are struck from the same geometry the rest of the
	// chapter uses, at their real stroke weights — an ornament drawn as a bare
	// outline at triple its proper weight is not a smaller ornament, it is a
	// smudge, and the first thing this plate has to do is make a comparison
	// look like something a person could actually have an opinion about.
	import Plate from '$lib/components/ui/Plate.svelte';
	import { inkOf, rosette, R_MAX, VIEW } from './rosette';
	import { N_GENES, type Gene } from '$lib/optim-rl/preference';

	const gene = (v: number[]) => Float64Array.from(v.slice(0, N_GENES)) as Gene;
	// Two specimens with visibly different characters, so the "which?" reads.
	const WINNER = rosette(gene([0.2, 0.66, 0.84, 0.16, 0.3, 0.16]));
	const LOSER = rosette(gene([0.7, 0.22, 0.52, 0.5, 0.6, 0.7]));

	const SPECIMEN = 96; // drawn size, view units
	const S = SPECIMEN / VIEW;
	const DOT_PATH = 'M 236 96 L 508 96';
</script>

<Plate
	id="balance"
	title="Which of these two"
	caption="The chapter in one line. A verdict is all anyone can reliably give; a judge is what you fit to a pile of them; a policy is what you point at the judge. The dashed tether is the only thing standing between the third box and whatever the second box turns out to have got wrong."
>
	<svg
		viewBox="0 0 720 214"
		class="mx-auto block w-full max-w-[760px]"
		role="img"
		aria-label="A pipeline: one person's pairwise verdict feeds a learned judge r-phi; the judge feeds a policy pi-theta; the policy is tethered by a beta-weighted KL penalty back to a frozen reference."
	>
		<defs>
			<marker
				id="tl-arrow"
				viewBox="0 0 8 8"
				refX="6.5"
				refY="4"
				markerWidth="7"
				markerHeight="7"
				orient="auto-start-reverse"
			>
				<path d="M 0 0.6 L 7 4 L 0 7.4 Z" fill="var(--ink-2)" />
			</marker>
		</defs>

		<path
			d="M 236 96 L 292 96 M 452 96 L 508 96"
			fill="none"
			stroke="var(--ink-2)"
			stroke-width="1"
			marker-end="url(#tl-arrow)"
		/>

		<!-- station 1 · the comparison, and the only thing a person supplies -->
		{#snippet specimen(shape: ReturnType<typeof rosette>, dx: number, chosen: boolean)}
			<g transform="translate({dx} 48) scale({S})" opacity={chosen ? 1 : 0.4}>
				<circle
					cx={VIEW / 2}
					cy={VIEW / 2}
					r={R_MAX}
					fill="none"
					stroke={inkOf(shape.mix, 0.3)}
					stroke-width="0.9"
				/>
				{#if shape.inner}
					{#each shape.inner.petals as d, i (i)}
						<path
							{d}
							fill={inkOf(shape.mix, shape.fill * shape.inner.alpha)}
							stroke={inkOf(shape.mix, shape.inner.alpha)}
							stroke-width={shape.stroke * 0.72}
							stroke-linejoin="round"
						/>
					{/each}
				{/if}
				{#each shape.outer.petals as d, i (i)}
					<path
						{d}
						fill={inkOf(shape.mix, shape.fill)}
						stroke={inkOf(shape.mix)}
						stroke-width={shape.stroke}
						stroke-linejoin="round"
					/>
				{/each}
				<circle cx={VIEW / 2} cy={VIEW / 2} r={shape.hub} fill={inkOf(shape.mix)} />
			</g>
		{/snippet}

		{@render specimen(WINNER, 24, true)}
		{@render specimen(LOSER, 128, false)}
		<!-- the verdict itself: a tick over the one that won, and nothing else -->
		<path
			d="M 62 40 l 5 6 l 10 -13"
			fill="none"
			stroke="var(--warm)"
			stroke-width="2.2"
			stroke-linecap="round"
			stroke-linejoin="round"
		/>
		<text x="130" y="172" text-anchor="middle" class="cap" fill="var(--warm)">
			y<tspan class="sub" dy="3">w</tspan><tspan dy="-3"> ≻ </tspan>y<tspan class="sub" dy="3"
				>l</tspan
			>
		</text>
		<text x="130" y="189" text-anchor="middle" class="sub-cap">one person, once</text>

		<!-- station 2 · the judge -->
		<rect
			x="292"
			y="66"
			width="160"
			height="60"
			rx="9"
			fill="var(--surface)"
			stroke="var(--line)"
		/>
		<text x="372" y="96" text-anchor="middle" class="box" fill="var(--accent)">
			r<tspan class="sub" dy="4">φ</tspan>
		</text>
		<text x="372" y="115" text-anchor="middle" class="sub-cap">the judge</text>
		<text x="372" y="150" text-anchor="middle" class="formula"
			>σ(r<tspan class="sub" dy="2">w</tspan><tspan dy="-2"> − r</tspan><tspan class="sub" dy="2"
				>l</tspan
			><tspan dy="-2">)</tspan></text
		>
		<text x="372" y="167" text-anchor="middle" class="sub-cap">fitted to the pile</text>

		<!-- station 3 · the policy, and its leash -->
		<rect
			x="508"
			y="66"
			width="160"
			height="60"
			rx="9"
			fill="var(--surface)"
			stroke="var(--line)"
		/>
		<text x="588" y="96" text-anchor="middle" class="box" fill="var(--accent)">
			π<tspan class="sub" dy="4">θ</tspan>
		</text>
		<text x="588" y="115" text-anchor="middle" class="sub-cap">the policy</text>

		<rect
			x="524"
			y="164"
			width="128"
			height="34"
			rx="8"
			fill="var(--surface-2)"
			stroke="var(--line-soft)"
		/>
		<text x="588" y="186" text-anchor="middle" class="box-mute">
			π<tspan class="sub" dy="3">ref</tspan>
		</text>
		<path
			d="M 588 164 L 588 126"
			fill="none"
			stroke="var(--cat-1)"
			stroke-width="1.5"
			stroke-dasharray="3 3"
		/>
		<text x="600" y="150" class="leash" fill="var(--cat-1)">β · KL</text>

		<circle r="3" fill="var(--accent)" class="dot" style="offset-path: path('{DOT_PATH}');" />
	</svg>
</Plate>

<style>
	.box {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 23px;
	}
	.box-mute {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 16px;
		fill: var(--ink-2);
	}
	.cap {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 16px;
	}
	.formula {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 15px;
		fill: var(--ink-2);
	}
	.sub {
		font-size: 11px;
	}
	.sub-cap {
		font-family: var(--font-sans);
		font-size: 9.5px;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		fill: var(--ink-3);
	}
	.leash {
		font-family: var(--font-mono);
		font-size: 11px;
	}
	.dot {
		opacity: 0;
		animation: tl-travel 4.4s linear infinite;
	}
	@keyframes tl-travel {
		0% {
			offset-distance: 0%;
			opacity: 0;
		}
		8% {
			opacity: 0.9;
		}
		78% {
			offset-distance: 100%;
			opacity: 0.9;
		}
		86%,
		100% {
			offset-distance: 100%;
			opacity: 0;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.dot {
			display: none;
		}
	}
	@supports not (offset-path: path('M 0 0 L 1 1')) {
		.dot {
			display: none;
		}
	}
</style>
