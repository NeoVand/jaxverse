<script lang="ts">
	import ChapterRef from '$lib/components/ui/ChapterRef.svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import LayerDiagram from '$lib/components/demos/space/LayerDiagram.svelte';
	import SpaceLab from '$lib/components/demos/space/SpaceLab.svelte';
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
		<Math
			display
			tex={'h \\;=\\; \\htmlClass{eq-op}{\\sigma}\\big(\\htmlClass{eq-model}{W} \\htmlClass{eq-world}{x} + \\htmlClass{eq-model-2}{b}\\big)'}
		/>
		<p>
			The affine part
			<Math tex={'\\htmlClass{eq-model}{W}\\htmlClass{eq-world}{x} + \\htmlClass{eq-model-2}{b}'} /> rotates,
			stretches, and shifts the plane. Then the bend
			<Math tex={'\\htmlClass{eq-op}{\\sigma}'} /> warps each coordinate — the classic
			<Math tex={'\\htmlClass{eq-op}{\\tanh}'} /> squashes it smoothly toward the interval
			<Math tex="(-1, 1)" />, though it pays for that gentleness in training time; the plates below
			open with gelu, which bends nearly as smoothly and learns in a fraction of the steps. Nothing
			here can cut or tear: the map is continuous, so points that started close together end up
			close together. And when <Math tex={'\\htmlClass{eq-model}{W}'} /> is invertible and the bend is
			smoothly invertible, as tanh is, the layer gives up even more than that — it becomes a
			<em>homeomorphism</em>, a deformation with a continuous inverse, the kind of move you could
			perform on a sheet of soft rubber without ever pressing two parts of it together.
		</p>
	</Prose>

	<LayerDiagram />

	<Prose>
		<p>A deep network is a chain of such moves, finished by one boring linear classifier:</p>
		<Math
			display
			tex={'f(\\htmlClass{eq-world}{x}) \\;=\\; \\underbrace{\\text{linear cut}}_{\\text{trivial}} \\;\\circ\\; \\underbrace{\\htmlClass{eq-op}{\\sigma}(\\htmlClass{eq-model}{W_L} \\cdot) \\circ \\cdots \\circ \\htmlClass{eq-op}{\\sigma}(\\htmlClass{eq-model}{W_1} \\cdot)}_{\\text{the deformation}}\\;(\\htmlClass{eq-world}{x})'}
		/>
		<p>
			So the entire intelligence of the machine lives in the deformation. The final layer can only
			cut once, with a straight line; the layers before it must rearrange the world until one
			straight cut is enough. Watch it happen.
		</p>
	</Prose>

	<SpaceLab
		variant="guided"
		id="rings"
		title="The experiment — rings, and a network two numbers wide"
		caption="Left: the data as it is, with the network's current verdict washed behind it. Middle: the same points — the same grid — after the network's deformation, and the one straight cut the final layer makes. Right: the network itself, live — every edge is one weight. Train, and watch it strain: a ring inside a ring cannot be pulled apart without leaving the plane. Then flip hidden from 2-D to 3-D and watch the same cut succeed. Hover the middle view and press unfold to replay the deformation."
	/>

	<UnderTheHood slug="space" block="device" />

	<Prose>
		<h2 class="h2">Why it failed, and what fixed it</h2>
		<p>
			With two hidden units, the network's inner world is itself a plane, and you watched it try
			every move it has. It cannot win, and the reason is not effort.
		</p>
		<p>
			Take the smooth case first, because it is the clean one. With <Math
				tex={'\\htmlClass{eq-op}{\\tanh}'}
			/> and an invertible <Math tex={'\\htmlClass{eq-model}{W}'} />, every layer is a
			homeomorphism: the sheet stretches and bends as far as you like, but it is never cut and never
			folded back onto itself, so what was inside stays inside. The orange ring encircles the blue
			disk before the deformation and encircles it after, and no straight line can have a disk on
			one side of it and a whole ring around that disk on the other. The network is not being slow.
			It is attempting something that cannot be done, and the plateaued loss curve is the honest
			report.
		</p>
		<p>
			The creasing bends slip out of that argument, and still do not help. relu <em>folds</em> the plane
			— a continuous map, but not a reversible one — and a fold can perfectly well bring an inside out.
			What stops it here is duller than topology: two units give the layer two creases, and you cannot
			enclose a bounded region with two straight cuts. You need three to make a triangle. Switch the activation
			and watch it fail the other way.
		</p>
		<p>
			Then you gave it one more dimension, and the impossible became easy. With three hidden units
			the deformation can <em>lift</em> — raise the inner disk out of the page like a tent pole
			under a napkin — and in three dimensions a flat plane slides between them cleanly. This is the
			move Christopher Olah's essay made unforgettable<Cite id="olah-2014" />, and it generalizes:
			data tangled like linked rings or knotted strings needs room — extra dimensions — to be taken
			apart, and a network needs enough width to provide that room.
		</p>
		<p>
			Now the thesis of this whole book, in one place. The network is a
			<em>smooth, continuous transformation</em> — that is all it is allowed to be, because gradient
			descent can only search by feel, and feel requires differentiability. The data arrived tangled
			by some process in the world — and that it arrived tangled rather than scattered is itself a
			supposition, with a name. The <em>manifold hypothesis</em> holds that data of enormous nominal
			size, a photograph with a million pixels, in fact lies on or near a surface of far smaller
			dimension, because whatever generated it had far fewer knobs than pixels.<Cite
				id="fefferman-2016"
			/> Learning succeeds when the network discovers a smooth map that
			<em>undoes the tangle</em> — approximately inverts it — so that in the new coordinates the classes
			are linearly separable. When no such smooth un-tangling exists in the space it has, the network
			fails, honestly and legibly, the way it just did. Width and depth are not magic: they are degrees
			of freedom for the deformation.
		</p>
	</Prose>

	<SpaceLab
		variant="free"
		id="tangles"
		title="The playground — pick your tangle"
		caption="Seven tangles, easiest to hardest, under the plot — spirals are the classic stress test. Watch the hidden view: tanh bends space in soft waves; relu folds it along straight creases; gelu and silu fold with the crease sanded smooth. Hover the hidden view and press unfold to replay the deformation. Widths beyond three are shown as a PCA shadow — the true untangling happens in more dimensions than a screen has, and the plane drawn through it is where the classifier’s one straight cut crosses the three directions being shown, not the cut itself."
	/>

	<UnderTheHood slug="space" block="middle" />

	<Prose>
		<h2 class="h2">The representation is the product</h2>
		<p>
			Click any point on the left and find its ghost on the right: the same datum, renamed by the
			network. That renaming is the product. We call the hidden layer's coordinates a
			<em>representation</em> — a coordinate system invented by training, in which the problem is
			trivial. The classifier at the end is almost an afterthought.<Cite id="bengio-2013" />
		</p>
		<p>
			You can measure this rather than admire it. Hang a bare linear classifier off each hidden
			layer of a trained network and score it: separability climbs layer by layer, and the climb is
			what this page has been drawing.<Cite id="alain-bengio-2016" /> Or count the holes. Take data shaped
			like two interlocked surfaces, push it through a trained network, and measure the topology of what
			comes out at every layer — the holes close one after another, until by the last hidden layer the
			two classes are two plain blobs. Networks with a folding bend get there in fewer layers than smooth
			ones, for the reason the creases gave two sections ago.<Cite id="naitzat-2020" />
		</p>
		<p>
			Every chapter from here is this chapter wearing different clothes. Handwritten digits live in
			a 784-dimensional input space — one axis per pixel — and a classifier must deform
			<em>that</em> space until ten regions come apart (<ChapterRef slug="digits" />). An
			autoencoder will squeeze the deformation through a two-dimensional bottleneck and hand us the
			map to look at (<ChapterRef slug="latent" />). Even the language model at the end of the book
			is moving words around a space until the next token sits in a predictable place. The grid you
			just watched bend is the only mental image you need.
		</p>
	</Prose>
</ChapterShell>
