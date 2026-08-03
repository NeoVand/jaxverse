<script lang="ts">
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import Wide from '$lib/components/ui/Wide.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import SpaceLab from '$lib/components/demos/space/SpaceLab.svelte';
	import { resolve } from '$app/paths';
</script>

<ChapterShell slug="space">
	<Prose>
		<p>
			In the last chapter a network bent a line into any curve you asked for. That was
			approximation. Classification sounds like a different job — here are two families of points,
			blue and orange, tangled into each other; tell them apart — and the standard picture of it is
			a network <em>drawing a boundary</em> between the classes, the way you might draw a fence between
			two herds.
		</p>
		<p>
			That picture is wrong in a quietly important way, and this chapter exists to replace it. A
			neural network does not draw a boundary around the data. It picks the data up —
			<em>all of space, with the data embedded in it</em> — and deforms the whole sheet until the two
			classes stand on opposite sides of a straight line. The boundary you see in the input is just the
			crease left behind.
		</p>
		<p>Look at what one layer actually computes:</p>
		<Math display tex="h \;=\; \sigma(W x + b)" />
		<p>
			The affine part <Math tex="Wx + b" /> rotates, stretches, and shifts the plane. Then the bend
			<Math tex="\sigma" /> warps each coordinate — the classic
			<Math tex="\tanh" /> squashes it smoothly toward the interval
			<Math tex="(-1, 1)" />, though it pays for that gentleness in training time; the plates below
			open with gelu, which bends nearly as smoothly and learns in a fraction of the steps. Nothing
			here can cut, tear, or glue. When <Math tex="W" /> is invertible and the bend is smoothly invertible,
			as tanh is, the layer is a <em>homeomorphism</em> — a continuous deformation with a continuous inverse,
			the kind of move you could perform with a sheet of soft rubber. A deep network is a chain of such
			moves, finished by one boring linear classifier:
		</p>
		<Math
			display
			tex={'f(x) \\;=\\; \\underbrace{\\text{linear cut}}_{\\text{trivial}} \\;\\circ\\; \\underbrace{\\sigma(W_L \\cdot) \\circ \\cdots \\circ \\sigma(W_1 \\cdot)}_{\\text{the deformation}}\\;(x)'}
		/>
		<p>
			So the entire intelligence of the machine lives in the deformation. The final layer can only
			cut once, with a straight line; the layers before it must rearrange the world until one
			straight cut is enough. Watch it happen.
		</p>
	</Prose>

	<Wide>
		<SpaceLab
			variant="guided"
			n={1}
			title="The experiment — rings, and a network two numbers wide"
			caption="Left: the data as it is, with the network's current verdict washed behind it. Middle: the same points — the same grid — after the network's deformation, and the one straight cut the final layer makes. Right: the network itself, live — every edge is one weight. Train, and watch it strain: a ring inside a ring cannot be pulled apart without leaving the plane. Then flip hidden from 2-D to 3-D and watch the same cut succeed. Hover the middle view and press unfold to replay the deformation."
		/>
	</Wide>

	<UnderTheHood slug="space" block="device" />

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			Why it failed, and what fixed it
		</h2>
		<p>
			With two hidden units, the network's inner world is itself a plane, and you watched it try
			every rubber-sheet move it has. It cannot win, and the reason is <em>topology</em>, not
			effort. The orange ring encircles the blue disk. Any continuous deformation of the plane
			preserves that encirclement — inside stays inside. However cleverly the grid bends, no
			straight line can have the disk on one side and the whole ring on the other. The network is
			not being slow; it is attempting the impossible, and its plateaued loss curve is the honest
			report.
		</p>
		<p>
			Then you gave it one more dimension, and the impossible became easy. With three hidden units
			the deformation can <em>lift</em> — raise the inner disk out of the page like a tent pole
			under a napkin — and in three dimensions a flat plane slides between them cleanly. This is the
			move Christopher Olah's essay
			<a href="https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/" rel="external"
				>Neural Networks, Manifolds, and Topology</a
			>
			made unforgettable, and it generalizes: data tangled like linked rings or knotted strings needs
			room — extra dimensions — to be taken apart, and a network needs enough width to provide that room.
		</p>
		<p>
			Now the thesis of this whole book, in one place. The network is a
			<em>smooth, continuous transformation</em> — that is all it is allowed to be, because gradient
			descent can only search by feel, and feel requires differentiability. The data arrived tangled
			by some process in the world. Learning succeeds when the network discovers a smooth map that
			<em>undoes the tangle</em> — approximately inverts it — so that in the new coordinates the classes
			are linearly separable. When no such smooth un-tangling exists in the space it has, the network
			fails, honestly and legibly, the way it just did. Width and depth are not magic: they are degrees
			of freedom for the deformation.
		</p>
	</Prose>

	<Wide>
		<SpaceLab
			variant="free"
			n={2}
			title="The playground — pick your tangle"
			caption="Seven tangles, easiest to hardest, under the plot — spirals are the classic stress test. Watch the hidden view: tanh bends space in soft waves; relu folds it along straight creases; gelu and silu fold with the crease sanded smooth. Hover the hidden view and press unfold to replay the deformation. Widths beyond three are shown as a PCA shadow — the true untangling happens in more dimensions than a screen has."
		/>
	</Wide>

	<UnderTheHood slug="space" block="middle" />

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			The representation is the product
		</h2>
		<p>
			Click any point on the left and find its ghost on the right: the same datum, renamed by the
			network. That renaming is the product. We call the hidden layer's coordinates a
			<em>representation</em> — a coordinate system invented by training, in which the problem is trivial.
			The classifier at the end is almost an afterthought.
		</p>
		<p>
			Every chapter from here is this chapter wearing different clothes. Handwritten digits live in
			a 784-dimensional input space — one axis per pixel — and a classifier must deform
			<em>that</em> space until ten regions come apart (<a href={resolve('/digits')}>Chapter 3</a>).
			An autoencoder will squeeze the deformation through a two-dimensional bottleneck and hand us
			the map to look at (<a href={resolve('/latent')}>Chapter 4</a>). Even the language model at
			the end of the book is moving words around a space until the next token sits in a predictable
			place. The grid you just watched bend is the only mental image you need.
		</p>
	</Prose>
</ChapterShell>
