<script lang="ts">
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import Wide from '$lib/components/ui/Wide.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import Squeeze from '$lib/components/demos/latent/Squeeze.svelte';
	import LatentMap from '$lib/components/demos/latent/LatentMap.svelte';
	import ManifoldGrid from '$lib/components/demos/latent/ManifoldGrid.svelte';
	import AutoencoderDiagram from '$lib/components/demos/latent/AutoencoderDiagram.svelte';
	import Neighbors from '$lib/components/demos/latent/Neighbors.svelte';
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
			waist of an hourglass — and every digit must pass through it. The waist is the only place in this
			book where a layer carries no non-linearity at all: nothing is bent there, because the whole point
			of the place is to be a plain coordinate system.
		</p>
	</Prose>

	<Wide>
		<AutoencoderDiagram />
	</Wide>

	<Prose>
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
		<p>
			One clause is still missing, and it is the clause that makes the map worth drawing. As
			written, the loss has no opinion whatever about <em>where</em> the encoder puts things. It is
			equally happy with a tidy disc around the origin and with a sprawl that reaches out to a
			coordinate of forty, continents of dead space in between, growing for as long as you train. So
			we ask for one more thing. Instead of a point, the encoder proposes a small Gaussian cloud per
			digit — a centre <Math tex="\mu(x)" /> and a spread <Math tex="\sigma(x)" /> — the decoder is handed
			a sample of it,
		</p>
		<Math
			display
			tex={'z \\;=\\; \\mu(x) + \\sigma(x)\\,\\varepsilon, \\qquad \\varepsilon \\sim \\mathcal{N}(0, I),'}
		/>
		<p>and the loss charges rent on how far that proposal drifts from the plain unit Gaussian:</p>
		<Math
			display
			tex={'\\mathcal{L} \\;=\\; \\underbrace{\\bigl\\lVert\\, x - D(z) \\,\\bigr\\rVert^{2}}_{\\text{rebuild it}} \\;+\\; \\beta \\underbrace{\\mathrm{KL}\\bigl(\\, \\mathcal{N}(\\mu, \\sigma^{2}) \\,\\|\\, \\mathcal{N}(0, 1) \\,\\bigr)}_{\\text{and stay put}}.'}
		/>
		<p>
			That is a <em>variational</em> autoencoder, and the two changes buy two different things. The
			noise means the decoder is never taught a single address, only a neighbourhood, so nearby
			addresses are forced to decode to similar digits: the map comes out smooth instead of a lookup
			table full of gaps. The rent means the map stays where you left it — centred on the origin, a
			couple of units across, no drift and no sprawl — which is why every plate below can frame the
			whole thing at once and keep it framed while it trains. Each digit is drawn at the centre <Math
				tex="\mu(x)"
			/> of its own proposal; the noise is a training-time device.
		</p>
	</Prose>

	<Wide>
		<Squeeze
			n={1}
			title="The squeeze — ten thousand digits through a narrow waist"
			caption="Eight held-out digits, and beneath them the network's current rebuild of each from the bottleneck alone. The gap between the rows is the loss — reconstruction is the entire training signal, and no label appears anywhere in it. Below the rows sits the shape of the hourglass: how many numbers the waist holds, how many layers squeeze down to it, and which non-linearity bends them. Any change re-rolls the weights and re-forms every map on this page."
		/>
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
			Notice also what the rebuilt row looks like when it stops improving: soft. Train it for an
			hour and it stays soft, and that is not a defect to be tuned away — it is the arithmetic
			showing its work. Squared error is minimized by an <em>average</em>, so the decoder's best
			possible answer at a given address is the mean of every digit that lands there, and two
			numbers cannot keep the difference between one person's 3 and another's. The jitter we just
			added pushes the same way — an address is trained together with its neighbours, so it answers
			for all of them. The blur is a receipt: it is exactly what the waist had to throw away. Add a
			layer to each side, or widen the waist, and the rebuild sharpens — those are the knobs a
			practitioner reaches for, and they come with the trade this chapter is built to make visible.
			At eight or sixteen numbers the digits come back nearly intact, and the map stops being
			something you can look at directly: the plate below falls back to plotting the three
			directions the cloud varies along most, a shadow of a space too wide to see. Two numbers is a
			deliberately cruel waist, chosen so the whole map fits on one page.
		</p>
		<p>
			The plate below does exactly that, with one twist held in reserve. In ink you see the map as
			the model knows it: anonymous points, arranged purely by reconstruction convenience. In
			images, each digit is printed at its own latent address, so the map reads like an atlas of
			handwriting — and the slider decides how many get printed. And the twist, which works over
			either view: colorize tints everything by its true label, information the network has never
			seen, not in any gradient, not once. If the bottleneck's two numbers carry nothing about
			digit-kind, the tints will fall like confetti.
		</p>
	</Prose>

	<Wide>
		<LatentMap
			n={2}
			title="The map — where the encoder put everything"
			caption="Left: two thousand held-out digits placed by the encoder alone, drawn as points or as printed thumbnails — one per occupied cell of a sheet whose fineness you choose. Colorize works over either, and with it on, every 3 has found the other 3s though labels were used only for the tints, after the fact. Right: the decoder repaints whatever point your cursor visits, and the walk crosses the country between two digits on foot."
		/>
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
		<ManifoldGrid
			n={3}
			title="The manifold — the decoder's answer everywhere"
			caption="A 21 × 21 grid of addresses laid over the latent square and decoded, tile by tile, live as training moves. The cursor's cell is repeated at right, and under it the nearest real held-out digit to that address, with the distance between them. Past a two-number waist the sheet is one slice through the space and the slider chooses which."
		/>
	</Wide>

	<Prose>
		<p>
			Watch a 3 shade into an 8 and the 8 into a 5, one tile at a time. Where the data actually
			lives you need not take anything on faith: pointing at a tile measures its address against
			every one of the two thousand held-out digits and reports the closest, in the waist's own
			units. Work outward and that distance grows — the prior keeps the digits packed near the
			origin, so the edges and corners of this square are addresses no real digit ever came near.
			The decoder answers there anyway, sharp-edged and committed, because
			<Math tex="D" /> is a smooth function that must produce 784 pixels for every point of the plane,
			though it was trained only where the data lives. You are looking at the seed of
			<em>generative</em> models: pick a latent point, decode it, and you have manufactured a thing that
			never existed. You are also looking at their oldest defect. A confident answer from a place the
			data never touched is, in the larger systems this book is walking toward, called a hallucination.
			Same machinery, same geometry.
		</p>
		<p>
			So far we have read the map as a picture. It has a second life as a tool, and this is the one
			the industry actually runs on. Stop thinking of <Math tex="z" /> as a location on a sheet and start
			thinking of it as a short list of numbers attached to a thing — an <em>embedding</em>. The
			encoder is now a machine that turns anything of its type into a vector, and once your things
			are vectors, “which of these is most like that one?” stops being a philosophical question and
			becomes arithmetic: embed everything once, keep the vectors, and answer a query by finding the
			nearest ones. That is <em>similarity search</em>, and it is how a photo library finds the
			other pictures of your dog, how a store recommends the next item, how duplicate documents get
			caught, and how a language model is handed the right three paragraphs before it answers you.
		</p>
		<p>
			The plate below hands you the query end of that machine. Draw a digit — your handwriting, not
			the dataset's — and it is centred, pushed through the encoder, and turned into the same short
			list of numbers every held-out digit already carries. Then the two thousand of them are sorted
			by how near they landed and the closest eight are printed. Nothing about the encoder was ever
			told what a 3 is, so when the row comes back full of 3s, that is the geometry answering, not a
			lookup.
		</p>
		<p>
			The second row is a control, and a fair one: the same stroke, the same candidates, the same
			metric, but distances measured between raw images instead of embeddings. Watch where the two
			rows disagree. Pixel distance rewards ink that lands in the same places, so it will happily
			return a fat 1 for a thin 7 and rank a slanted 3 far from an upright one; the map is looking
			for the same shape drawn any way at all. Draw badly on purpose — a wobbly 8, a 4 with an open
			top — and the rows separate fastest.
		</p>
		<p>
			It also introduces the measuring stick that the next chapters use by default. Instead of the
			straight-line distance between two vectors, compare their directions:
		</p>
		<Math
			display
			tex={'\\cos(u, v) \\;=\\; \\frac{u \\cdot v}{\\lVert u \\rVert\\, \\lVert v \\rVert} \\;\\in\\; [-1, 1].'}
		/>
		<p>
			<em>Cosine similarity</em> is 1 when two vectors point the same way, 0 when they are at right angles,
			−1 when they oppose — and it ignores length entirely. That is usually what you want of an embedding,
			because in a wide space the direction tends to carry the meaning while the length carries something
			duller, like how common or how confident the thing is. It is not a law, though, and this chapter
			is a good place to see the exception: at a two-number waist, throwing away the radius throws away
			half the map, so plain distance often wins. Try both.
		</p>
	</Prose>

	<Wide>
		<Neighbors
			n={4}
			title="Search by drawing — your stroke, and the digits nearest it"
			caption="Draw on the pad and the same two thousand held-out digits are sorted twice: by distance between embeddings, and by distance between raw images, under whichever metric you pick. Both searches read the same centred ink, and the small square shows your stroke after its round trip through the waist — what the map actually kept of it. Borrow a real digit to start from, click any neighbour to draw on it, and the meters at the bottom keep score over forty-eight held-out queries, where chance is one in ten."
		/>
	</Wide>

	<Prose>
		<p>
			The meters keep score over held-out digits rather than your handwriting, and at a two-number
			waist they say something you might not expect: the pixel row wins, and not narrowly. That is
			worth sitting with rather than explaining away. Part of it is arithmetic — you gave the map
			two numbers and the pixels 784 and then asked for a fair fight. The rest is that MNIST is the
			kindest dataset raw pixel distance will ever meet. Every digit here, and every stroke you
			draw, is centred by its ink and scaled into the same box, in white on black with no
			background, no lighting and no camera. Under those conditions overlap really is a decent
			stand-in for shape, which is why nearest-neighbour on raw MNIST pixels has been a respectable
			baseline since the 1990s. Photograph the same digits on a desk instead and that baseline is
			gone by lunchtime, while an encoder trained on desks would barely notice. Pixel distance is
			not wrong here. It is being flattered.
		</p>
		<p>
			So run it the other way. Go back to the first plate, widen the waist to sixteen, and train for
			a few minutes: the map's meter climbs out of the fifties and settles level with the pixel row,
			which has nothing to learn and never moves. Level — not past it. That is the result, and it is
			the more useful one, because the tie is being won with sixteen numbers against 784. A stored
			embedding is fifty times smaller, and every search is fifty times less arithmetic; at a
			billion items that is the difference between an index and an impossibility.
		</p>
		<p>
			It is also worth being clear about why the map only ties. Nothing in the loss ever asked for
			digits of a kind to sit together. The encoder was paid to rebuild pixels, so it kept what
			rebuilding needs — shape, but also slant, weight, size — and two 7s in different hands can
			still land apart because they genuinely differ in what the decoder must draw. Kind-ness came
			out as a side effect, and side effects tie. If you want a space where <em>same kind</em> is near
			by construction, you have to say so in the loss: show the model two views of the same thing and
			require it to pull them together while pushing everything else away. That is contrastive learning,
			it is how modern image and text encoders are actually trained, and it is what turns this chapter's
			happy accident into a design.
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
