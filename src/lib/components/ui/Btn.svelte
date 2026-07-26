<script lang="ts">
	import type { Snippet } from 'svelte';

	// The house button: quiet by default, ink when primary. Uppercase Inter,
	// hairline border — chrome that defers to the work.
	interface Props {
		onclick?: () => void;
		kind?: 'primary' | 'ghost';
		/** One of a set of choices, currently the active one — a quiet pressed
		 * state rather than the ink fill of `primary`, since nothing about it is
		 * the next thing to do. */
		pressed?: boolean;
		disabled?: boolean;
		title?: string;
		children: Snippet;
	}

	let { onclick, kind = 'ghost', pressed, disabled = false, title, children }: Props = $props();
</script>

<button
	{onclick}
	{disabled}
	{title}
	aria-pressed={pressed}
	class="btn inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[11px] font-medium tracking-[0.1em] uppercase transition-all duration-100 disabled:cursor-default disabled:opacity-40"
	class:btn-primary={kind === 'primary'}
	class:btn-on={pressed}
>
	{@render children()}
</button>

<style>
	.btn {
		font-family: var(--font-sans);
		border: 1px solid var(--line);
		background: var(--surface);
		color: var(--ink-2);
	}
	.btn:hover:not(:disabled) {
		border-color: var(--ink-3);
		color: var(--ink);
	}
	.btn:active:not(:disabled) {
		transform: translateY(0.5px);
	}
	.btn-on {
		background: var(--surface-2);
		border-color: var(--ink-2);
		color: var(--ink);
	}
	.btn-primary {
		background: var(--ink);
		border-color: var(--ink);
		color: var(--paper);
	}
	.btn-primary:hover:not(:disabled) {
		border-color: var(--ink);
		color: var(--paper);
		opacity: 0.88;
	}
</style>
