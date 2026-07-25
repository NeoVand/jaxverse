<script lang="ts">
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import Wide from '$lib/components/ui/Wide.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Squeeze from '$lib/components/demos/latent/Squeeze.svelte';
	import LatentMap from '$lib/components/demos/latent/LatentMap.svelte';
	import ManifoldGrid from '$lib/components/demos/latent/ManifoldGrid.svelte';
	import { resolve } from '$app/paths';
</script>

<ChapterShell slug="latent">
	<Prose>
		<p>
			In the last chapter you trained a classifier, and every one of its ten thousand digits arrived
			with an answer stapled to it: this one is a 4 — be less wrong about that. The label did the
			pointing, and the network learned whatever the pointing required. But almost nothing you have
			ever learned from came labeled. The world mostly hands you the thing itself — light, sound,
			ink — and no answer key. So this chapter takes the labels away entirely and asks what is left
			to learn from.
		</p>
		<p>
			Here is the bet the whole chapter rides on: you understand a thing to the degree that you can
			rebuild it. Not name it — rebuild it, all 784 pixels of it. An <em>autoencoder</em> is that
			bet built as a machine in two halves. The <em>encoder</em>
			<Math tex={'E : \\mathbb{R}^{784} \\to \\mathbb{R}^{2}'} /> squeezes each digit down to two numbers.
			The <em>decoder</em>
			<Math tex={'D : \\mathbb{R}^{2} \\to \\mathbb{R}^{784}'} /> takes those two numbers and tries to
			repaint the digit they came from. Between the halves sits the <em>bottleneck</em> — the width-2
			waist of an hourglass — and every digit must pass through it.
		</p>
		<p>
			Training needs nothing this book has not already built. The loss is the distance between each
			digit and its own <em>reconstruction</em>,
		</p>
		<Math display tex={'\\mathcal{L} \\;=\\; \\bigl\\lVert\\, x - D(E(x)) \\,\\bigr\\rVert^{2},'} />
		<p>
			and gradient descent falls downhill on it as always. Read the formula twice and notice what is
			missing: there is no <Math tex="y" /> anywhere. The data grades itself —
			<Math tex="x" /> is both the question and the answer key. Then notice what the bottleneck is for.
			Allowed 784 numbers in the middle, the network could pass its input straight through and learn nothing.
			Allowed two, copying is impossible. The only way to reconstruct well is to keep what matters about
			a digit and discard the rest — and “what matters” is precisely the thing nobody told it.
		</p>
	</Prose>

	<Wide>
		<Plate
			n={1}
			title="The squeeze — ten thousand digits through a two-number waist"
			caption="Eight held-out digits, and beneath them the network's current rebuild of each from the bottleneck alone. The gap between the rows is the loss — reconstruction is the entire training signal, and no label appears anywhere in it. The bottleneck chips widen the waist from two numbers to three (fresh weights each time)."
		>
			<Squeeze />
		</Plate>
	</Wide>

	<Prose>
		<p>
			While it trains, consider the pinch point. Halfway through the network each digit exists as
			exactly two numbers, <Math tex="z = E(x)" /> — a 392-fold compression, and the only thing the decoder
			is ever shown. Whatever survives the squeeze is, by construction, everything the reconstruction
			needs: loop or no loop, slant, stroke weight, openness. Ten thousand digits become ten thousand
			points on a flat sheet. That sheet is called the <em>latent space</em> — the network's private map
			— and because this one happens to be two-dimensional, we can look at the whole of it at once. (Widen
			the waist to three numbers and the sheet becomes a small globe you can turn.)
		</p>
		<p>
			The plate below does exactly that, with one twist held in reserve. In ink you see the map as
			the model knows it: anonymous points, arranged purely by reconstruction convenience. In
			images, each digit is printed at its own latent address, so the map reads like an atlas of
			handwriting. And the twist — reveal tints every digit by its true label, information the
			network has never seen, not in any gradient, not once. If the bottleneck's two numbers carry
			nothing about digit-kind, the tints will fall like confetti.
		</p>
	</Prose>

	<Wide>
		<Plate
			n={2}
			title="The map — where the encoder put everything"
			caption="Left: two thousand held-out digits placed by the encoder alone — in image view roughly 600 of the 2000 are shown, one per occupied cell, so the tiles stay legible. Flip to reveal and every 3 has found the other 3s, though labels were used only for the tints, after the fact; on the right, the decoder repaints whatever point your cursor visits."
		>
			<LatentMap />
		</Plate>
	</Wide>

	<Prose>
		<p>
			They do not fall like confetti. The model was never told what a seven is, and there on the map
			is the country of sevens — because reconstructing sevens well <em>requires</em> gathering
			them. Two digits that need the same decoding must sit near each other in the bottleneck, or
			the decoder cannot serve both; kind emerges as a by-product of thrift. This is
			<em>representation learning</em>, and it is the thesis of
			<a href={resolve('/space')}>Bending Space</a> wearing new clothes: the network deforms its input
			space until the task becomes easy, and the deformed coordinates are the real product. There we had
			to trust that the hidden layers were building something. Here the map is laid flat where you can
			sweep your hand across it.
		</p>
		<p>
			The map has geography worth reading. Ones press into a thin peninsula — a stroke with little
			more than a slant to its name. Fours and nines blur along a shared border, exactly as they do
			in handwriting. And the interpolation walk crosses country lines on foot: between a 3 and an 8
			the decoder produces in-betweens no person ever wrote, rendered with perfect assurance.
		</p>
		<p>
			There is one more way to put the question, and it is the bluntest. Stop following the data
			altogether. Lay a uniform grid of addresses over the whole square — no digit chose them, we
			did — and make the decoder answer for every single one.
		</p>
	</Prose>

	<Wide>
		<Plate
			n={3}
			title="The manifold — the decoder's answer everywhere"
			caption="Every tile is the decoder's reply to “what lives at this address?” for a uniform 14 × 14 grid over the latent square — data or no data — and it re-decodes as training moves. With a three-number bottleneck, the slider chooses which slice of the cube the sheet cuts through."
		>
			<ManifoldGrid />
		</Plate>
	</Wide>

	<Prose>
		<p>
			Watch a 3 shade into an 8 and the 8 into a 5, one tile at a time. Now the empty streets are
			not an inference; they are printed. Between the islands lie coordinates where no real digit
			ever landed, and the decoder answers there anyway, sharp-edged and committed, because
			<Math tex="D" /> is a smooth function that must produce 784 pixels for every point of the plane,
			though it was trained only where the data lives. You are looking at the seed of
			<em>generative</em> models: pick a latent point, decode it, and you have manufactured a thing that
			never existed. You are also looking at their oldest defect. A confident answer from a place the
			data never touched is, in the larger systems this book is walking toward, called a hallucination.
			Same machinery, same geometry.
		</p>
		<p>
			Carry the trick forward, because the rest of the book stands on it. No one labeled anything
			today: the data was its own target, and the loss came from hiding part of the pipeline and
			demanding restoration. That maneuver is called <em>self-supervision</em>, and it scales past
			anything hand-labeling could reach. Here, a digit graded itself. In
			<a href={resolve('/language')}>the next chapter</a> a sentence grades itself — every word is the
			label for the words before it — and a model that plays that game long enough backs into knowing
			the language.
		</p>
	</Prose>
</ChapterShell>
