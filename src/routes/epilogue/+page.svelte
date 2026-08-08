<script lang="ts">
	import { resolve } from '$app/paths';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
</script>

<ChapterShell
	slug="epilogue"
	front={{
		numeral: '∞',
		eyebrow: 'Epilogue · Sources · ≈3 min',
		title: 'Sources & onward',
		deck: 'Where the ideas in this book came from, and where to go when you have finished it.'
	}}
>
	<Prose>
		<h2 class="h2">The shoulders this stands on</h2>
		<p>
			This book stands on the shoulders of work it admires. If a chapter moved you, the original is
			better:
		</p>
		<ul class="list-disc space-y-2 pl-5">
			<li>
				Christopher Olah,
				<a href="https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/" rel="external"
					>Neural Networks, Manifolds, and Topology</a
				>
				— the essay behind <em>Bending Space</em>, and the reason this book exists.
			</li>
			<li>
				<a href="https://playground.tensorflow.org" rel="external">TensorFlow Playground</a> — the original
				proof that you can feel a neural network through a browser tab.
			</li>
			<li>
				<a href="https://gradientlab.ai" rel="external">Gradient Lab</a> — the optimization laboratory
				whose landscapes the Prologue descends.
			</li>
			<li>
				<a href="https://jax-js.com" rel="external">jax-js</a> — JAX, reborn in TypeScript on WebGPU;
				the engine under every live plate here.
			</li>
			<li>
				Kenneth Li et&nbsp;al., <em>Emergent World Representations</em> (Othello-GPT) and Adam Karvonen’s
				chess-GPT work — the lineage Rook belongs to: small game-playing transformers that were never
				shown a board and turn out to carry one anyway.
			</li>
		</ul>

		<h2 class="h2">Onward</h2>
		<p>
			Start over from <a href={resolve('/')}>the beginning</a>, or open the hood: everything on this
			site — every model, chapter, and mistake — lives in its repository. Train something.
		</p>
	</Prose>
</ChapterShell>
