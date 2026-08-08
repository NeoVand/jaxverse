<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	// The canonical loop, drawn once: the agent acts; the environment answers
	// with a new state and a number. Every algorithm in this chapter lives
	// inside these two arrows. Two dots circulate (action out, feedback back)
	// via CSS offset-path — removed entirely under reduced motion or where
	// offset-path is unsupported; the figure is complete without them.
	const ACT_PATH = 'M 310 42 C 398 42 398 148 310 148';
	const OBS_PATH = 'M 170 148 C 82 148 82 42 170 42';
</script>

<Plate
	id="loop"
	title="The loop"
	caption="The whole setting, in two arrows: act, then be told how it went — nothing else crosses the line."
>
	<svg
		viewBox="0 0 480 190"
		class="mx-auto block w-full max-w-[550px]"
		role="img"
		aria-label="The reinforcement learning loop: the agent sends an action to the environment; the environment returns the next state and a reward to the agent."
	>
		<defs>
			<marker
				id="rl-loop-arrow"
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

		<!-- the two arrows -->
		<path
			d={ACT_PATH}
			fill="none"
			stroke="var(--ink-2)"
			stroke-width="1"
			marker-end="url(#rl-loop-arrow)"
		/>
		<path
			d={OBS_PATH}
			fill="none"
			stroke="var(--ink-2)"
			stroke-width="1"
			marker-end="url(#rl-loop-arrow)"
		/>

		<!-- the two parties -->
		<rect
			x="170"
			y="20"
			width="140"
			height="44"
			rx="7"
			fill="var(--surface)"
			stroke="var(--line)"
		/>
		<rect
			x="170"
			y="126"
			width="140"
			height="44"
			rx="7"
			fill="var(--surface)"
			stroke="var(--line)"
		/>
		<text x="240" y="47" text-anchor="middle" class="box-label">agent</text>
		<text x="240" y="153" text-anchor="middle" class="box-label">environment</text>

		<!-- edge labels -->
		<text x="386" y="99" text-anchor="start" class="edge-label" fill="var(--accent)">
			action a<tspan class="sub" dy="3">t</tspan>
		</text>
		<text x="94" y="86" text-anchor="end" class="edge-label" fill="var(--ink-2)">
			state s<tspan class="sub" dy="3">t+1</tspan>
		</text>
		<text x="94" y="108" text-anchor="end" class="edge-label" fill="var(--warm)">
			reward r<tspan class="sub" dy="3">t+1</tspan>
		</text>

		<!-- circulating dots: action out, feedback home -->
		<circle r="3" fill="var(--accent)" class="dot" style="offset-path: path('{ACT_PATH}');" />
		<circle
			r="3"
			fill="var(--warm)"
			class="dot dot-late"
			style="offset-path: path('{OBS_PATH}');"
		/>
	</svg>
</Plate>

<style>
	.box-label {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 15.5px;
		fill: var(--ink);
	}
	.edge-label {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 14px;
	}
	.sub {
		font-size: 10px;
	}
	.dot {
		opacity: 0;
		animation: rl-travel 5.6s linear infinite;
	}
	.dot-late {
		animation-delay: 2.8s;
	}
	@keyframes rl-travel {
		0% {
			offset-distance: 0%;
			opacity: 0;
		}
		6% {
			opacity: 0.9;
		}
		42% {
			offset-distance: 96%;
			opacity: 0.9;
		}
		48% {
			offset-distance: 100%;
			opacity: 0;
		}
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
