<script lang="ts">
	import type { Snippet } from 'svelte';

	// Demos are presented as numbered plates, like figures in an old textbook:
	// a hairline frame, a roman numeral, a caption that earns its place.
	interface Props {
		n?: number;
		title: string;
		caption?: string;
		/** Right side of the header — status text, live indicators. */
		status?: Snippet;
		children: Snippet;
	}

	let { n, title, caption, status, children }: Props = $props();

	function roman(x: number): string {
		const table: Array<[number, string]> = [
			[10, 'X'],
			[9, 'IX'],
			[5, 'V'],
			[4, 'IV'],
			[1, 'I']
		];
		let out = '';
		let v = x;
		for (const [k, s] of table)
			while (v >= k) {
				out += s;
				v -= k;
			}
		return out;
	}
</script>

<figure class="my-10 overflow-hidden rounded-lg border border-line bg-surface">
	<figcaption
		class="flex min-h-10 flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b border-line-soft px-4 py-2"
	>
		<span class="flex items-baseline gap-2.5">
			{#if n !== undefined}
				<span class="eyebrow" style="color: var(--warm);">Plate {roman(n)}</span>
			{/if}
			<span class="font-serif text-[15px] italic" style="font-variation-settings: 'opsz' 14;">
				{title}
			</span>
		</span>
		{#if status}
			<span class="num flex items-center gap-2 text-[11px] text-ink-3">{@render status()}</span>
		{/if}
	</figcaption>

	{@render children()}

	{#if caption}
		<p
			class="border-t border-line-soft px-4 py-2.5 font-serif text-[13.5px] text-ink-2 italic"
			style="font-variation-settings: 'opsz' 13;"
		>
			{caption}
		</p>
	{/if}
</figure>
