<script lang="ts">
	// The mark: a topographic map of a loss landscape — two contour rings
	// sliding off-center toward the minimum, where the vermilion ball rests.
	// Read it as orbits and it is also a little universe.
	let { size = 20 }: { size?: number } = $props();
</script>

<svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true" class="shrink-0">
	<path
		d="M 27.7 12.06 C 28.42 13.99, 28.25 16.87, 27.26 18.87 C 26.27 20.86, 23.92 22.75, 21.78 24.02 C 19.64 25.29, 16.87 26.4, 14.43 26.49 C 12 26.58, 8.92 25.84, 7.16 24.58 C 5.41 23.32, 4.41 20.93, 3.88 18.91 C 3.35 16.89, 3.24 14.54, 4 12.46 C 4.76 10.38, 6.38 7.68, 8.44 6.41 C 10.5 5.15, 13.93 4.73, 16.35 4.88 C 18.78 5.03, 21.1 6.12, 22.99 7.32 C 24.88 8.52, 26.99 10.14, 27.7 12.06 Z"
		stroke="var(--accent)"
		stroke-width="2.3"
		opacity="0.45"
	/>
	<path
		d="M 24.71 15.09 C 24.89 16.34, 24.08 17.88, 23.35 19.1 C 22.63 20.32, 21.69 21.57, 20.37 22.41 C 19.05 23.26, 16.98 24.22, 15.43 24.17 C 13.88 24.11, 12.11 23.02, 11.09 22.08 C 10.07 21.14, 9.62 19.8, 9.31 18.53 C 8.99 17.27, 8.64 15.73, 9.2 14.49 C 9.75 13.26, 11.24 11.8, 12.65 11.12 C 14.06 10.44, 16.04 10.35, 17.64 10.43 C 19.24 10.5, 21.08 10.81, 22.26 11.58 C 23.43 12.36, 24.53 13.83, 24.71 15.09 Z"
		stroke="var(--accent)"
		stroke-width="2.3"
	/>
	<circle cx="17.2" cy="17.6" r="3.2" fill="var(--warm)" />
</svg>
