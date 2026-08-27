<script lang="ts">
	import { resolve } from '$app/paths';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
</script>

<ChapterShell
	slug="epilogue"
	front={{
		numeral: '∞',
		eyebrow: 'Epilogue · Credits · ≈3 min',
		title: 'Credits & onward',
		deck: 'What this book was built out of, and where to go when you have finished it.'
	}}
>
	<Prose>
		<h2 class="h2">The shoulders this stands on</h2>
		<p>
			Every chapter ends with the papers it stood on, each one linked to somewhere you can read it
			without paying. This page is for the other kind of debt: the work that decided what this book
			would be, and the code it runs on.
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
				<a href="https://github.com/shaack/cm-chessboard" rel="external">cm-chessboard</a>
				by Stefan Haack (MIT) — every board in <em>Everything at Once</em>, wearing the
				<a
					href="https://github.com/lichess-org/lila/tree/master/public/piece/staunty"
					rel="external">Staunty</a
				> pieces from lichess (CC BY-NC-SA 4.0), with the black set darkened here so it reads on this
				book's board.
			</li>
			<li>
				Kenneth Li et&nbsp;al.,
				<a href="https://arxiv.org/abs/2210.13382" rel="external"
					><em>Emergent World Representations</em></a
				>
				(Othello-GPT) and
				<a href="https://arxiv.org/abs/2403.15498" rel="external">Adam Karvonen’s chess-GPT work</a>
				— the lineage Rook belongs to: small game-playing transformers that were never shown a board and
				turn out to carry one anyway.
			</li>
		</ul>

		<h2 class="h2">Onward</h2>
		<p>
			Start over from <a href={resolve('/')}>the beginning</a>, or open the hood: everything on this
			site — every model, chapter, and mistake — lives in its repository. Train something.
		</p>
	</Prose>
</ChapterShell>
