<script lang="ts">
	import './layout.css';
	import { page } from '$app/state';
	import { base, resolve } from '$app/paths';
	import { Moon, Sun, MonitorSmartphone } from 'lucide-svelte';
	import { applyTheme, storedTheme, type ThemePreference } from '$lib/theme';
	import { chapterBySlug } from '$lib/data/chapters';
	import Logo from '$lib/components/ui/Logo.svelte';

	let { children } = $props();

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
</script>

<svelte:head>
	<link
		rel="icon"
		href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cpath d='M4 6 Q 13 30 17 22 Q 20 16 28 18' fill='none' stroke='%232b45d8' stroke-width='3.4' stroke-linecap='round'/%3E%3Ccircle cx='15.2' cy='24.4' r='3.4' fill='%23d3541f'/%3E%3C/svg%3E"
	/>
	<meta name="theme-color" media="(prefers-color-scheme: light)" content="#faf9f5" />
	<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#141310" />
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
		<a href={resolve('/')} class="flex items-center gap-2.5" aria-label="jaxverse home">
			<Logo size={20} />
			<span
				class="font-serif text-[17px] tracking-tight"
				style="font-variation-settings: 'opsz' 18; font-weight: 540;"
			>
				jaxverse
			</span>
		</a>

		<div class="flex items-center gap-5">
			{#if current}
				<span class="eyebrow hidden sm:inline" aria-hidden="true">
					{current.n === 0 ? 'Prologue' : `Chapter ${current.n}`} · {current.kicker}
				</span>
			{/if}
			<a href="{resolve('/')}#contents" class="eyebrow transition-colors hover:text-ink">
				Contents
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
		<a href={resolve('/epilogue')} class="eyebrow transition-colors hover:text-ink"
			>Epilogue &amp; credits</a
		>
	</div>
</footer>
