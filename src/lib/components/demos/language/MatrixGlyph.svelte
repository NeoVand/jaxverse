<script lang="ts">
	// A weight matrix as a texture: deterministic value noise on a canvas, drawn
	// once at token colors. It stands for a matrix's SHAPE, never its contents —
	// the label carries the truth, and the caption says so. Static by design, so
	// reduced-motion needs no special case.
	interface Props {
		rows: number;
		cols: number;
		/** Shape label, e.g. "96 × 96". */
		label: string;
		/** Aspect of the drawn block in CSS pixels. */
		w?: number;
		h?: number;
		tone?: 'accent' | 'warm' | 'ink';
		seed?: number;
	}

	let { rows, cols, label, w = 96, h = 64, tone = 'ink', seed = 1 }: Props = $props();

	let canvas = $state<HTMLCanvasElement | undefined>();

	function hexRgb(hex: string): [number, number, number] {
		const s = hex.trim().replace('#', '');
		const n =
			s.length === 3
				? s
						.split('')
						.map((c) => c + c)
						.join('')
				: s;
		const v = parseInt(n, 16);
		return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
	}

	// cell grid caps out so a [96×384] block stays legible as texture
	const gx = $derived(Math.min(cols, 26));
	const gy = $derived(Math.min(rows, 18));

	$effect(() => {
		const c = canvas;
		if (!c) return;
		const gxx = gx;
		const gyy = gy;
		const dpr = Math.min(devicePixelRatio || 1, 2);
		const W = c.clientWidth;
		const H = c.clientHeight;
		if (c.width !== W * dpr || c.height !== H * dpr) {
			c.width = W * dpr;
			c.height = H * dpr;
		}
		const ctx = c.getContext('2d');
		if (!ctx) return;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, W, H);
		const style = getComputedStyle(c);
		const key = tone === 'accent' ? '--accent' : tone === 'warm' ? '--warm' : '--ink';
		const [r, g, b] = hexRgb(style.getPropertyValue(key) || '#000');
		const cw = W / gxx;
		const ch = H / gyy;
		for (let j = 0; j < gyy; j++) {
			for (let i = 0; i < gxx; i++) {
				// deterministic hash → the same texture every paint and theme
				const t = Math.abs(
					Math.sin((i + 1) * 12.9898 + (j + 1) * 78.233 + seed * 3.71) * 43758.5453
				);
				const a = 0.1 + (t % 1) * 0.55;
				ctx.fillStyle = `rgba(${r},${g},${b},${a.toFixed(3)})`;
				ctx.fillRect(i * cw + 0.4, j * ch + 0.4, cw - 0.8, ch - 0.8);
			}
		}
	});
</script>

<span class="inline-flex flex-col items-center gap-1">
	<canvas
		bind:this={canvas}
		style="width: {w}px; height: {h}px;"
		class="block rounded-sm border border-line-soft"
		aria-hidden="true"
	></canvas>
	<span class="num text-[9.5px] whitespace-nowrap text-ink-3">{label}</span>
</span>
