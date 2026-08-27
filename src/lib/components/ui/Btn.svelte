<script lang="ts">
	import type { Snippet } from 'svelte';

	// The house button: quiet by default, accent-tinted when primary.
	// Uppercase Inter, hairline border — chrome that defers to the work.
	//
	// `primary` means THE NEXT THING TO DO, and nothing else. A plate's
	// transport is therefore primary while it sits idle and ghost while it
	// runs: Train is what you came to press, Pause is only the way out of it.
	// Written down because it had drifted four ways across twenty buttons —
	// some permanently accented, some accented only while running, most never
	// accented at all — and the accent stops meaning anything the moment two
	// plates disagree about it.
	interface Props {
		onclick?: () => void;
		/** `primary` for the next thing to do. See the note above before
		 *  reaching for it: a running demo's Pause is not that. */
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
	/* primary is an accent tint, not an ink fill — an ink fill reads as a
	   glaring white block in dark mode, and these sit mid-plate */
	.btn-primary {
		background: var(--accent-soft);
		border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
		color: var(--accent);
	}
	.btn-primary:hover:not(:disabled) {
		background: var(--accent-soft);
		border-color: color-mix(in srgb, var(--accent) 70%, var(--line));
		color: var(--accent);
	}
</style>
