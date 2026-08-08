<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	import { plateLabel } from '$lib/data/plates';
	// aliased: the script below needs the global Math object
	import MathTex from '$lib/components/ui/Math.svelte';
	import { ACTIVATIONS, type ActivationId } from './activations';

	// The field guide: every activation drawn on the SAME axes — one unit is
	// the same distance everywhere, so ReLU's ramp is exactly 45° and the
	// difference between GELU and SiLU is honest, not an artifact of scaling.
	// Solid teal: σ. Dashed vermilion: σ′ — the gate every gradient
	// must pass through on the way back. Hover a tile to read exact values.
	const W = 240;
	const H = 120;
	const Z0 = -4;
	const Z1 = 4;
	const Y0 = -1.6;
	const Y1 = 2.4;
	const PX = W / (Z1 - Z0); // 30 px per unit…
	const PY = H / (Y1 - Y0); // …on both axes

	const xP = (z: number) => (z - Z0) * PX;
	const yP = (y: number) => (Y1 - y) * PY;

	function pathOf(f: (z: number) => number): string {
		let d = '';
		const n = 161;
		for (let i = 0; i < n; i++) {
			const z = Z0 + ((Z1 - Z0) * i) / (n - 1);
			d += `${i === 0 ? 'M' : 'L'}${xP(z).toFixed(1)} ${yP(f(z)).toFixed(1)}`;
		}
		return d;
	}

	// static per activation — computed once
	const tiles = ACTIVATIONS.map((a) => ({ ...a, d: pathOf(a.fn), dd: pathOf(a.dfn) }));

	let probe = $state<{ id: ActivationId; z: number } | null>(null);

	function onMove(id: ActivationId, e: PointerEvent) {
		const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const z = Z0 + ((e.clientX - rect.left) / Math.max(1, rect.width)) * (Z1 - Z0);
		probe = { id, z: Math.max(Z0, Math.min(Z1, z)) };
	}

	const minus = (s: string) => s.replace('-', '−');
</script>

<Plate
	id="atlas"
	live
	title="A field guide to activations"
	caption={`Nine bends on identical axes, one unit square everywhere. The solid teal curve is the activation σ; the dashed vermilion curve is its derivative σ′ — the gate every gradient must pass through on the way back down. Read the flat stretches of σ′ as places learning goes quiet: saturation on the classics, the dead zone on relu, and the small negative slope the modern trio keeps alive. Hover any tile for exact values; six of these are live in ${plateLabel('neuron', 'neuron')}.`}
>
	{#snippet status()}
		<span class="inline-flex items-center gap-1.5">
			<svg width="18" height="8" aria-hidden="true"
				><line x1="1" y1="4" x2="17" y2="4" stroke="var(--cat-2)" stroke-width="2" /></svg
			>
			σ(z)
		</span>
		<span class="inline-flex items-center gap-1.5">
			<svg width="18" height="8" aria-hidden="true"
				><line
					x1="1"
					y1="4"
					x2="17"
					y2="4"
					stroke="var(--warm)"
					stroke-width="1.4"
					stroke-dasharray="3 3"
				/></svg
			>
			σ′(z)
		</span>
	{/snippet}

	<div class="grid grid-cols-1 gap-px bg-line-soft sm:grid-cols-3">
		{#each tiles as t (t.id)}
			{@const p = probe?.id === t.id ? probe : null}
			<div class="bg-surface px-4 pt-3 pb-3.5">
				<div class="flex min-h-5 items-baseline justify-between gap-2">
					<span class="eyebrow" style="color: var(--ink-2);">{t.label}</span>
					{#if p}
						<span class="num text-[10px]">
							<span style="color: var(--ink-3);">z {minus(p.z.toFixed(1))}</span>
							<span class="ml-1.5" style="color: var(--cat-2);"
								>σ {minus(t.fn(p.z).toFixed(2))}</span
							>
							<span class="ml-1.5" style="color: var(--warm);"
								>σ′ {minus(t.dfn(p.z).toFixed(2))}</span
							>
						</span>
					{:else}
						<span class="num text-[10px] text-ink-3">{t.year}</span>
					{/if}
				</div>

				<svg
					width="100%"
					viewBox="0 0 {W} {H}"
					class="mt-1.5 block cursor-crosshair"
					role="img"
					aria-label="{t.label}: the function and its derivative on z from −4 to 4"
					onpointermove={(e) => onMove(t.id, e)}
					onpointerleave={() => (probe = null)}
				>
					<defs>
						<clipPath id="atlas-clip-{t.id}">
							<rect x="0" y="0" width={W} height={H} />
						</clipPath>
					</defs>

					<!-- axes: z = 0, σ = 0, and a whisper at σ = 1 (the derivative's ceiling) -->
					<line
						x1="0"
						x2={W}
						y1={yP(1)}
						y2={yP(1)}
						stroke="var(--line-soft)"
						stroke-width="1"
						stroke-dasharray="1 4"
					/>
					<line x1="0" x2={W} y1={yP(0)} y2={yP(0)} stroke="var(--line)" stroke-width="1" />
					<line x1={xP(0)} x2={xP(0)} y1="0" y2={H} stroke="var(--line)" stroke-width="1" />

					<g clip-path="url(#atlas-clip-{t.id})">
						<path
							d={t.dd}
							fill="none"
							stroke="var(--warm)"
							stroke-width="1.3"
							stroke-dasharray="3 3"
							opacity="0.85"
						/>
						<path
							d={t.d}
							fill="none"
							stroke="var(--cat-2)"
							stroke-width="1.9"
							stroke-linejoin="round"
							stroke-linecap="round"
						/>
						{#if p}
							<line
								x1={xP(p.z)}
								x2={xP(p.z)}
								y1="0"
								y2={H}
								stroke="var(--ink-3)"
								stroke-width="1"
								opacity="0.45"
							/>
							<circle
								cx={xP(p.z)}
								cy={yP(t.dfn(p.z))}
								r="2.4"
								fill="var(--warm)"
								stroke="var(--surface)"
								stroke-width="1"
							/>
							<circle
								cx={xP(p.z)}
								cy={yP(t.fn(p.z))}
								r="3"
								fill="var(--cat-2)"
								stroke="var(--surface)"
								stroke-width="1.2"
							/>
						{/if}
					</g>
				</svg>

				<div class="formula mt-2">
					<MathTex tex={'\\sigma(z) = ' + t.tex} />
				</div>
				<p class="mt-1 text-[11px] leading-snug text-ink-3">{t.note}</p>
			</div>
		{/each}
	</div>
</Plate>

<style>
	.formula :global(.katex) {
		font-size: 0.8rem;
		color: var(--ink-2);
	}
</style>
