<script lang="ts">
	// The hero's living mark: gradient descent with momentum on a fixed 1-D
	// landscape, restarting from a random point once it settles. Ink curve,
	// vermilion ball, fading accent trail. Honors prefers-reduced-motion.
	let { height = 220 }: { height?: number } = $props();

	let canvas: HTMLCanvasElement;

	// the landscape: a pleasant double well over [0,1], normalized so the
	// drawn curve fills [0.08, 0.88] of the box height
	const raw = (x: number) => {
		const t = x * 3.9 - 1.95;
		return t * t * t * t - 2.6 * t * t + 0.9 * t;
	};
	const [rLo, rHi] = (() => {
		let lo = Infinity;
		let hi = -Infinity;
		for (let i = 0; i <= 200; i++) {
			const v = raw(i / 200);
			lo = Math.min(lo, v);
			hi = Math.max(hi, v);
		}
		return [lo, hi];
	})();
	const f = (x: number) => 0.08 + (0.8 * (raw(x) - rLo)) / (rHi - rLo);
	const df = (x: number) => {
		const h = 1e-4;
		return (f(x + h) - f(x - h)) / (2 * h);
	};

	$effect(() => {
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
		let raf = 0;
		let disposed = false;

		let x = 0.07;
		let v = 0;
		let settle = 0;
		const trail: Array<{ x: number; y: number; a: number }> = [];

		const css = () => getComputedStyle(canvas);

		function draw() {
			if (disposed) return;
			const dpr = Math.min(devicePixelRatio || 1, 2);
			const W = canvas.clientWidth;
			const H = canvas.clientHeight;
			if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
				canvas.width = W * dpr;
				canvas.height = H * dpr;
			}
			if (!ctx) return;
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, W, H);

			const style = css();
			const ink = style.getPropertyValue('--ink-3').trim() || '#999';
			const accent = style.getPropertyValue('--accent').trim() || '#2b45d8';
			const warm = style.getPropertyValue('--warm').trim() || '#d3541f';

			const pad = 18;
			const px = (u: number) => pad + u * (W - 2 * pad);
			const py = (u: number) => H - pad - u * (H - 2 * pad);

			// landscape
			ctx.beginPath();
			for (let i = 0; i <= 160; i++) {
				const u = i / 160;
				const X = px(u);
				const Y = py(f(u));
				if (i === 0) ctx.moveTo(X, Y);
				else ctx.lineTo(X, Y);
			}
			ctx.strokeStyle = ink;
			ctx.lineWidth = 1.4;
			ctx.stroke();

			// physics: momentum descent
			if (!reduced) {
				const g = df(x);
				v = 0.86 * v - 0.0022 * g * 60;
				x += v * 0.016;
				if (x < 0.02) {
					x = 0.02;
					v = Math.abs(v) * 0.4;
				}
				if (x > 0.98) {
					x = 0.98;
					v = -Math.abs(v) * 0.4;
				}
				if (Math.abs(v) < 0.004 && Math.abs(g) < 0.06) settle++;
				else settle = 0;
				if (settle > 110) {
					x = 0.04 + Math.random() * 0.9;
					v = 0;
					settle = 0;
					trail.length = 0;
				}
				trail.push({ x, y: f(x), a: 1 });
				if (trail.length > 90) trail.shift();
				for (const t of trail) t.a *= 0.975;
			}

			// trail
			for (const t of trail) {
				ctx.beginPath();
				ctx.arc(px(t.x), py(t.y), 1.6, 0, Math.PI * 2);
				ctx.globalAlpha = t.a * 0.45;
				ctx.fillStyle = accent;
				ctx.fill();
			}
			ctx.globalAlpha = 1;

			// the ball
			const bx = reduced ? 0.62 : x;
			ctx.beginPath();
			ctx.arc(px(bx), py(f(bx)), 5.2, 0, Math.PI * 2);
			ctx.fillStyle = warm;
			ctx.fill();

			if (!reduced) raf = requestAnimationFrame(draw);
		}

		draw();
		if (reduced) {
			// render one static frame; nothing moves
		}

		return () => {
			disposed = true;
			cancelAnimationFrame(raf);
		};
	});
</script>

<canvas bind:this={canvas} class="block w-full" style="height: {height}px;" aria-hidden="true"
></canvas>
