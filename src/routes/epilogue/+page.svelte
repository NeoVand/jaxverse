<script lang="ts">
	import { resolve } from '$app/paths';
</script>

<svelte:head>
	<title>Epilogue · jaxverse</title>
	<meta name="description" content="Where the ideas in jaxverse came from, and where to go next." />
</svelte:head>

<article class="mx-auto max-w-2xl px-5 pt-16 pb-10 sm:pt-24">
	<p class="eyebrow mb-4">Epilogue</p>
	<h1
		class="font-serif tracking-tight"
		style="font-size: clamp(2.5rem, 6vw, 3.6rem); line-height: 1.05; font-weight: 480; font-variation-settings: 'opsz' 60;"
	>
		Sources &amp; onward
	</h1>

	<div class="chapter-prose mt-8">
		<p>
			This book stands on the shoulders of work it admires. If a chapter moved you, the original is
			better:
		</p>
		<ul class="list-disc space-y-2 pl-5 text-[1.05rem]">
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
				the engine under every live demo here.
			</li>
			<li>
				Kenneth Li et&nbsp;al., <em>Emergent World Representations</em> (Othello-GPT) and Adam Karvonen’s
				chess-GPT work — the lineage of Rook’s probe experiment.
			</li>
		</ul>
		<p>
			Start over from <a href={resolve('/')}>the beginning</a>, or open the hood: everything on this
			site — every model, chapter, and mistake — lives in its repository. Train something.
		</p>
	</div>
</article>
