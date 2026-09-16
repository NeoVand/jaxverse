<script lang="ts">
	import { SENSOR_SIZE } from '$lib/world/sensor';
	interface Props {
		pixels: Float32Array;
		size?: number;
	}
	let { pixels, size = SENSOR_SIZE }: Props = $props();
	function draw(canvas: HTMLCanvasElement) {
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const data = ctx.createImageData(size, size);
		for (let i = 0; i < size * size; i++) {
			const value = Math.round(Math.max(0, Math.min(1, pixels[i] ?? 0)) * 255);
			data.data.set([value, value, value, 255], i * 4);
		}
		ctx.putImageData(data, 0, 0);
	}
</script>

<canvas
	width={size}
	height={size}
	{@attach draw}
	aria-label={`${size} by ${size} grayscale observation supplied to the model`}
></canvas>

<style>
	canvas {
		width: 128px;
		height: 128px;
		image-rendering: pixelated;
		border: 1px solid var(--line);
	}
</style>
