<!-- The key to reading an ArchDiagram: color = sign, thickness = magnitude. -->
<div class="mt-1 flex flex-wrap items-center gap-x-3.5 gap-y-1 px-1">
	<span class="legend-item">
		<svg width="16" height="6" aria-hidden="true">
			<line x1="0" x2="16" y1="3" y2="3" stroke="var(--accent)" stroke-width="2" />
		</svg>
		weight &gt; 0
	</span>
	<span class="legend-item">
		<svg width="16" height="6" aria-hidden="true">
			<line x1="0" x2="16" y1="3" y2="3" stroke="var(--warm)" stroke-width="2" />
		</svg>
		weight &lt; 0
	</span>
	<span class="legend-item">
		<svg width="22" height="8" aria-hidden="true">
			<polygon points="0,4.5 22,1 22,7 0,5.5" fill="var(--ink-3)" opacity="0.7" />
		</svg>
		thicker = larger
	</span>
</div>

<style>
	.legend-item {
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
</style>
