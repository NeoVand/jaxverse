<script lang="ts">
	import './layout.css';
	import { page } from '$app/state';
	import { base, resolve } from '$app/paths';
	import { List, Moon, Sun, MonitorSmartphone } from 'lucide-svelte';
	import { applyTheme, effectiveTheme, storedTheme, type ThemePreference } from '$lib/theme';
	import { chapterBySlug } from '$lib/data/chapters';
	import { chapterGlyphs } from '$lib/data/glyphs';
	import { themePulse, watchTheme } from '$lib/viz/tokens.svelte';
	import Logo from '$lib/components/ui/Logo.svelte';

	let { children } = $props();

	// The browser's own chrome — favicon and address-bar tint — has to follow
	// the theme the reader pinned, not only the one the OS asks for.
	watchTheme();
	let isDark = $state(false);
	$effect(() => {
		void themePulse.tick;
		isDark = effectiveTheme() === 'dark';
	});
	const mark = $derived(
		isDark
			? { ring: '93a3ff', ball: 'ff8e57', paper: '#141310' }
			: { ring: '2b45d8', ball: 'd3541f', paper: '#faf9f5' }
	);

	// writable $derived: server renders 'system'; the client re-derives from
	// storage on hydration; cycleTheme reassigns from then on
	let theme: ThemePreference = $derived(storedTheme());

	function cycleTheme() {
		theme = theme === 'system' ? 'dark' : theme === 'dark' ? 'light' : 'system';
		applyTheme(theme);
	}

	// Which chapter are we in, if any? Drives the small position note in the bar.
	const current = $derived.by(() => {
		const seg = page.url.pathname.split('/').filter(Boolean).at(-1) ?? '';
		return chapterBySlug.get(seg);
	});
	// The same glyph the contents rail gave it, so the bar names the place twice.
	const CurrentGlyph = $derived(current ? chapterGlyphs[current.slug] : undefined);

	// The colophon: a small modal hidden behind the mark itself.
	let colophon: HTMLDialogElement;
</script>

<svelte:head>
	<!-- the mark at favicon scale: two contour rings + the ball at the minimum -->
	<link
		rel="icon"
		href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' fill='none'%3E%3Cpath d='M 27.7 12.06 C 28.42 13.99 28.25 16.87 27.26 18.87 C 26.27 20.86 23.92 22.75 21.78 24.02 C 19.64 25.29 16.87 26.4 14.43 26.49 C 12 26.58 8.92 25.84 7.16 24.58 C 5.41 23.32 4.41 20.93 3.88 18.91 C 3.35 16.89 3.24 14.54 4 12.46 C 4.76 10.38 6.38 7.68 8.44 6.41 C 10.5 5.15 13.93 4.73 16.35 4.88 C 18.78 5.03 21.1 6.12 22.99 7.32 C 24.88 8.52 26.99 10.14 27.7 12.06 Z' stroke='%23{mark.ring}' stroke-width='2.2' opacity='0.55'/%3E%3Cpath d='M 24.71 15.09 C 24.89 16.34 24.08 17.88 23.35 19.1 C 22.63 20.32 21.69 21.57 20.37 22.41 C 19.05 23.26 16.98 24.22 15.43 24.17 C 13.88 24.11 12.11 23.02 11.09 22.08 C 10.07 21.14 9.62 19.8 9.31 18.53 C 8.99 17.27 8.64 15.73 9.2 14.49 C 9.75 13.26 11.24 11.8 12.65 11.12 C 14.06 10.44 16.04 10.35 17.64 10.43 C 19.24 10.5 21.08 10.81 22.26 11.58 C 23.43 12.36 24.53 13.83 24.71 15.09 Z' stroke='%23{mark.ring}' stroke-width='2.2'/%3E%3Ccircle cx='17.4' cy='18.1' r='3.4' fill='%23{mark.ball}'/%3E%3C/svg%3E"
	/>
	<meta name="theme-color" content={mark.paper} />
	<meta property="og:type" content="website" />
	<meta property="og:image" content="{base}/og.png" />
	<meta name="twitter:card" content="summary_large_image" />
</svelte:head>

<a href="#main" class="skip-link">Skip to content</a>

<header
	class="fixed inset-x-0 top-0 z-50 border-b border-line-soft"
	style="height: var(--header-h); background: var(--glass); backdrop-filter: blur(14px) saturate(1.3); -webkit-backdrop-filter: blur(14px) saturate(1.3);"
>
	<div class="mx-auto flex h-full max-w-6xl items-center justify-between px-5">
		<span class="flex items-center gap-2.5">
			<button
				onclick={() => colophon.showModal()}
				class="flex items-center justify-center rounded-md transition-opacity hover:opacity-75"
				aria-label="About jaxverse"
				title="About jaxverse"
			>
				<Logo size={20} />
			</button>
			<a
				href={resolve('/')}
				class="font-serif text-[17px] tracking-tight"
				style="font-variation-settings: 'opsz' 18; font-weight: 540;"
				aria-label="jaxverse home"
			>
				jaxverse
			</a>
		</span>

		<div class="flex items-center gap-2 sm:gap-3">
			{#if current && CurrentGlyph}
				<span class="eyebrow hidden items-center gap-1.5 md:inline-flex" aria-hidden="true">
					<CurrentGlyph size={13} class="shrink-0" style="color: var(--accent);" />
					{current.n === 0 ? 'Prologue' : `Chapter ${current.n}`} · {current.kicker}
				</span>
				<span class="hidden h-3.5 w-px bg-line md:inline-block" aria-hidden="true"></span>
			{/if}
			<a
				href="{resolve('/')}#contents"
				class="eyebrow inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 transition-colors hover:bg-surface-2 hover:text-ink"
			>
				<List size={13} aria-hidden="true" class="shrink-0" />
				Contents
			</a>
			<a
				href="https://github.com/NeoVand/jaxverse"
				rel="external"
				class="flex h-8 w-8 items-center justify-center rounded-md border border-transparent text-ink-2 transition-colors hover:border-line hover:text-ink"
				aria-label="jaxverse on GitHub"
				title="View the source on GitHub"
			>
				<!-- Feather Icons “github” (MIT) — the standard outlined mark -->
				<svg
					width="16"
					height="16"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="1.8"
					stroke-linecap="round"
					stroke-linejoin="round"
					aria-hidden="true"
				>
					<path
						d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"
					/>
				</svg>
			</a>
			<button
				onclick={cycleTheme}
				class="flex h-8 w-8 items-center justify-center rounded-md border border-transparent text-ink-2 transition-colors hover:border-line hover:text-ink"
				aria-label="Theme: {theme}. Click to change."
				title="Theme: {theme}"
			>
				{#if theme === 'dark'}
					<Moon size={15} />
				{:else if theme === 'light'}
					<Sun size={15} />
				{:else}
					<MonitorSmartphone size={15} />
				{/if}
			</button>
		</div>
	</div>
</header>

<!-- the colophon, hidden behind the mark: who made this, and where to find him -->
<dialog
	bind:this={colophon}
	class="colophon"
	aria-label="About jaxverse"
	onclick={(e) => {
		if (e.target === colophon) colophon.close();
	}}
>
	<div class="flex flex-col items-center gap-4 px-8 py-9 text-center">
		<Logo size={88} />
		<p
			class="font-serif text-[22px] tracking-tight"
			style="font-variation-settings: 'opsz' 24; font-weight: 540;"
		>
			jaxverse
		</p>
		<p class="max-w-[26ch] font-serif text-[14.5px] leading-relaxed text-ink-2 italic">
			A little universe of learning machines, designed and built by Neo&nbsp;Mohsenvand.
		</p>
		<div class="mt-1 flex items-center gap-2">
			<a
				href="https://github.com/NeoVand"
				rel="external"
				class="chip inline-flex items-center gap-1.5"
			>
				<svg
					width="12"
					height="12"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
					aria-hidden="true"
				>
					<path
						d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"
					/>
				</svg>
				GitHub
			</a>
			<a
				href="https://www.linkedin.com/in/mohsenvand/"
				rel="external"
				class="chip inline-flex items-center gap-1.5"
			>
				<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
					<path
						d="M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.47-.9 1.63-1.85 3.36-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28ZM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12ZM7.12 20.45H3.56V9h3.56v11.45ZM22.22 0H1.77C.79 0 0 .77 0 1.72v20.55C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.72C24 .77 23.2 0 22.22 0Z"
					/>
				</svg>
				LinkedIn
			</a>
		</div>
	</div>
</dialog>

<main id="main" style="padding-top: var(--header-h);">
	{@render children()}
</main>

<footer class="mt-24 border-t border-line-soft">
	<div class="mx-auto flex max-w-6xl flex-wrap items-baseline justify-between gap-3 px-5 py-8">
		<p class="text-[13px] text-ink-3">
			Every model on this site is trained live, in your browser, by
			<a
				class="underline decoration-line hover:decoration-ink-3"
				href="https://jax-js.com"
				rel="external">jax-js</a
			>
			— nothing is faked, nothing leaves your machine.
		</p>
		<div class="flex items-center gap-5">
			<a
				href="https://github.com/NeoVand/jaxverse"
				rel="external"
				class="eyebrow transition-colors hover:text-ink">GitHub</a
			>
			<a href={resolve('/epilogue')} class="eyebrow transition-colors hover:text-ink"
				>Epilogue &amp; credits</a
			>
		</div>
	</div>
</footer>

<style>
	.colophon {
		margin: auto;
		border: 1px solid var(--line);
		border-radius: 12px;
		background: var(--paper);
		color: var(--ink);
		box-shadow: 0 24px 60px -18px rgb(0 0 0 / 0.35);
	}
	.colophon::backdrop {
		background: color-mix(in srgb, var(--paper) 55%, transparent);
		backdrop-filter: blur(6px);
		-webkit-backdrop-filter: blur(6px);
	}
</style>
