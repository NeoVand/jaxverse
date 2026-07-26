<script lang="ts">
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import Wide from '$lib/components/ui/Wide.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Gallery from '$lib/components/demos/digits/Gallery.svelte';
	import SoftmaxPlay from '$lib/components/demos/digits/SoftmaxPlay.svelte';
	import Classifier from '$lib/components/demos/digits/Classifier.svelte';
	import PipelineDiagram from '$lib/components/demos/digits/PipelineDiagram.svelte';
	import DrawPad from '$lib/components/demos/digits/DrawPad.svelte';
	import Inside from '$lib/components/demos/digits/Inside.svelte';
	import { resolve } from '$app/paths';
</script>

<ChapterShell slug="digits">
	<Prose>
		<p>
			Every experiment in this book so far has been <em>supervised learning</em>, and we never
			stopped to say its name. Here is the whole recipe, stated plainly. Collect a dataset of paired
			examples — questions with their answers already attached:
		</p>
		<Math display tex={'\\mathcal{D} \\;=\\; \\{(x^{(i)},\\, y^{(i)})\\}_{i=1}^{N}'} />
		<p>
			Choose a family of functions <Math tex={'\\hat{y} = f(x;\\theta)'} />. Choose a loss
			<Math tex={'\\mathcal{L}'} /> that scores how much a prediction disagrees with the attached answer.
			Then run the only rule this book uses, <Math
				tex={'\\theta \\leftarrow \\theta - \\gamma \\nabla \\mathcal{L}'}
			/>, averaged over examples. Fitting a curve was this recipe with x and y both single numbers.
			The tangled spirals of <a href={resolve('/space')}>Chapter 2</a> were this recipe with x a point
			on a plane. Nothing in the recipe says x must stay small.
		</p>
		<p>
			So make x an image. MNIST is the field's old warhorse: handwritten digits scanned off forms in
			the 1990s, each one a 28 × 28 grid of gray. Read the grid row by row and the picture becomes a
			list of 784 brightnesses — a single point in
			<Math tex={'\\mathbb{R}^{784}'} />, exactly as a point on a plane was a list of two
			coordinates. In that space, “every 7 anyone might write” is not an idea but a <em>place</em>:
			a sprawling, tangled region, interleaved with the region of 1s and the region of 9s. The
			classifier's job is Chapter 2's job, word for word — deform the space until the ten regions
			come apart and flat cuts can separate them. The only thing you lose in 784 dimensions is the
			ability to watch the bending. From here on, we judge the deformation by its consequences.
			Before any model touches them, look at the raw material.
		</p>
	</Prose>

	<Wide>
		<Plate
			n={1}
			title="The dataset"
			caption="A random page from the training set — ten thousand of these ride with this chapter, each a 28 × 28 grid of gray written by a different hand."
		>
			<Gallery />
		</Plate>
	</Wide>

	<Prose>
		<p>
			Before the training, the shape of the machine. It is the same stack you built in the last two
			chapters, with two small pieces of machinery bolted on to make ten classes workable. The 784
			brightnesses go in one end. The network ends in ten outputs rather than one — a score per
			digit, a vector called the <em>logits</em> — and those scores are free to be any size, either
			sign; nothing yet says they are probabilities. So the last step makes them into some: the
			<em>softmax</em> takes the ten scores and returns ten positive numbers that sum to one. The tallest
			is the model's answer, and its height is how strongly the model means it.
		</p>
	</Prose>

	<Wide>
		<PipelineDiagram />
	</Wide>

	<Prose>
		<p>The squashing step is one line of arithmetic,</p>
		<Math display tex={'\\operatorname{softmax}_c(z) \\;=\\; \\frac{e^{z_c}}{\\sum_{j} e^{z_j}}'} />
		<p>
			— each score exponentiated, then normalized, so the numbers are positive and sum to one. An
			equation like this is better felt than read, so play with it: a bank of scores drawn from a
			bell curve, and the belief they become. Watch the biggest score take most — never all — of the
			probability; widen the spread <Math tex="\sigma" /> and the winner runs away with it; and see how
			dividing the logits by a <em>temperature</em> before the softmax sharpens or flattens the verdict.
		</p>
	</Prose>

	<Wide>
		<SoftmaxPlay
			title="The softmax, by hand"
			caption="Above: the raw scores, positive up and negative down, with the shape of their distribution beside them. Below: what the softmax makes of them. Move any dial and every probability shifts — belief is a budget of exactly one, spent by comparison. The temperature τ rescales the scores first: below one it exaggerates their differences; above one it shrugs them off. With a handful of scores you can also drag a bar to set it by hand."
		/>
	</Wide>

	<Prose>
		<p>
			The second piece of machinery is the loss. Ten probabilities came out; one of them belongs to
			the right answer. That single number is the only one the loss looks at, and the loss is its
			negative logarithm:
		</p>
		<Math
			display
			tex={'\\mathcal{L}(\\theta) \\;=\\; -\\log \\htmlClass{eq-g}{p_{\\htmlClass{eq-w}{y}}}, \\qquad \\htmlClass{eq-g}{p_{\\htmlClass{eq-w}{y}}} \\;=\\; \\operatorname{softmax}_{\\htmlClass{eq-w}{y}}\\!\\big(f(x;\\theta)\\big)'}
		/>
		<p>
			Read it term by term. <Math tex={'\\htmlClass{eq-w}{y}'} /> is the true label, the answer written
			on the form by whoever held the pen; it never changes, and it selects exactly one of the ten bars.
			<Math tex={'\\htmlClass{eq-g}{p_{\\htmlClass{eq-w}{y}}}'} /> is the height of that one bar — the
			probability the model was willing to place on the truth. The nine other bars appear nowhere in the
			formula, and yet they are punished all the same: they share one unit of belief with
			<Math tex={'\\htmlClass{eq-g}{p_{\\htmlClass{eq-w}{y}}}'} />, so the only way to raise it is
			to take from them.
		</p>
		<p>
			The minus log is what turns a probability into a complaint. It is zero when
			<Math tex={'\\htmlClass{eq-g}{p_{\\htmlClass{eq-w}{y}}} = 1'} /> — perfect confidence in the right
			answer costs nothing — and it climbs slowly at first, then without limit as the probability approaches
			zero. Nine parts in ten, the blind guess of an untrained model, costs
			<Math tex="-\log 0.1 \approx 2.30" />; a tepid but correct 0.6 costs 0.51; a confident 0.99
			costs 0.01; and a confident, <em>wrong</em> 0.01 costs 4.61. That asymmetry is the whole
			design. Cross-entropy is often called the model's <em>surprise</em> at the truth, and being
			loudly, specifically wrong is the most surprising thing it can do — so that is where the
			gradient pushes hardest. Every digit in the plate below carries its own surprise, and the
			<span class="num">loss</span> in the header is simply their average.
		</p>
		<p>
			One more piece of discipline — the piece that makes this science rather than wishful thinking.
			A network with a hundred thousand parameters can <em>memorize</em> thousands of examples: loss
			falling, accuracy climbing, and nothing learned about handwriting at all. Memorizing answers
			is not learning, so we grade on questions the model has never seen. Of the ten thousand digits
			below, 8,000 form the <em>training set</em> the gradients flow from, and 2,000 form the
			<em>test set</em>, locked away from training entirely. The plate reports both scores, and the
			pair is more honest than either alone: accuracy on rows it has studied, accuracy on rows it
			has never seen, and the shortfall between them — the memorization it got away with. Only the
			second number is a claim about handwriting.
		</p>
	</Prose>

	<Wide>
		<div id="plate-classifier">
			<Classifier
				n={2}
				title="The classifier"
				caption="Sixteen digits from the test set — the model never trains on them. Under each, ten bars: its whole belief, one per class. Then the reading, green when right and red when wrong, and beside it the surprise −log p that cross-entropy charges for it. Untrained, the bars are flat and every surprise sits near 2.30; train, and watch the beliefs spike as the readings correct themselves. Random draws a fresh sixteen, but the other two buttons rank the whole test set by the probability it gives the right answer and show you an end of it: Surest for the digits it has no trouble with, Hardest for the ones it gets confidently, expensively wrong. The right column is the machine itself: change its depth, its width, or the bend it uses, and it starts over with the new shape."
			/>
		</div>
	</Wide>

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			What the machine actually receives
		</h2>
		<p>
			It is worth pausing on what the model is given, because it is not what you see. You see a 7 —
			two strokes, a sharp corner, a thing with a top. It receives a list:
			<code>0.00, 0.00, 0.31, 0.89, 0.94, …</code> — 784 numbers in a fixed order, and nothing else. It
			is not told that pixel 215 sits directly above pixel 243; adjacency, stroke, corner, loop — every
			visual notion has to be rediscovered statistically, from which brightnesses rise and fall together
			across eight thousand examples. That “seven-ness” can be recovered from a bare list of numbers is
			the quiet astonishment of this chapter, and the mechanism is the one you already watched: the same
			bending of space as Chapter 2, performed in 784 dimensions instead of two.
		</p>
		<p>
			While it trains, the two curves in the plate tell different stories. The
			<span style="color: var(--accent);">training loss</span> nearly cannot help falling — the
			optimizer is paid to push it down, and with a hundred thousand parameters there is room to
			push. The
			<span style="color: var(--warm);">test accuracy</span> is the earned number: performance on
			questions the gradients never touched. How well a model carries over from the examples it
			studied to examples it has not seen is called <em>generalization</em>, and the shortfall
			between the two is the memorization it got away with. Here the gap stays small, and the reason
			is unglamorous: eight thousand examples of a well-posed task. It is the field's bluntest law —
			more data beats cleverness, almost every time it is allowed to compete.
		</p>
		<p>
			The 2,000 test digits are still other people's handwriting, scanned by the same machines in
			the same decade. Your handwriting, drawn with a pointer, is a genuinely harder exam. One
			courtesy is extended before each guess: MNIST digits are centered by their center of mass, so
			the pad below re-centers your stroke the same way — the model has never seen a digit hiding in
			a corner, and without that shift it would fail for reasons that have nothing to do with shape.
		</p>
	</Prose>

	<Wide>
		<Plate
			n={3}
			title="Draw your own"
			caption="A 3 is waiting on the pad; wipe it with the eraser and draw your own. The ring is your brush, and the slider on the left fattens or thins it. On the right, the same two charts as the softmax plate above, now fed by your stroke: ten raw scores over a zero rule, and the belief they become. The evidence square is the gradient of the winning score with respect to each pixel — warm pixels argued for the verdict, blue pixels argued against it, so you can see which parts of your stroke the model leaned on."
		>
			<DrawPad />
		</Plate>
	</Wide>

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			Opening the box
		</h2>
		<p>
			When it misreads your 4 — and sometimes it will — you can do better than shrug. Nothing about
			this machine is sealed. The first layer maps 784 pixels onto a bank of hidden units, so each
			unit owns exactly 784 incoming weights — one per pixel — which means
			<em>each unit is an image</em>, and you can simply look at it: the pattern of ink it rewards
			and the pattern it penalizes. Units deeper in are harder to see that way, so there is a second
			question to ask of any of them —
			<em>which digits drive you hardest?</em> — and the answer is a picture too, since you can average
			the winners. And its mistakes on the test set have structure. It does not confuse digits at random;
			it confuses digits that share a skeleton, the same pairs you would confuse squinting at a bad fax.
		</p>
	</Prose>

	<Wide>
		<Plate
			n={4}
			title="Inside the machine"
			caption="Left: an explorer over every learned layer, with two ways to look at a unit. Weights draws what it is made of — a first-layer unit owns 784 incoming weights, one per pixel, so it simply is an image, and a deeper unit is carried back to pixel space by multiplying out the matrices behind it, its linear shadow. Excites it drops the algebra and asks the test set: the two dozen digits that drive the unit hardest, averaged. Walk out to the readout in that view and the ten class templates come back as ten clean digits — the machine's idea of each name. Right: the confusion matrix over all 2,000 test digits, rows truth and columns its reading. Hover any cell to open the digits it actually got that way; the heaviest are usually 4 read as 9 and 3 read as 5, mistakes of shared shape."
		>
			<Inside />
		</Plate>
	</Wide>

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			The workhorse and its price
		</h2>
		<p>
			What you just trained is not a toy version of the real thing; it is the real thing, small.
			This exact recipe — pixels in, softmax out, cross-entropy pushed downhill — was reading the
			amounts on a sizable share of American bank checks by the late 1990s, and sorting mail by ZIP
			code before that. Scaled up, it reads street signs for cars and flags tumors on scans.
			Whenever the world hands you questions paired with answers, supervised learning is the tool
			you reach for first, and usually the tool that wins.
		</p>
		<p>
			Its price is hiding in that phrase: <em>paired with answers</em>. Every label in this chapter
			was a person's judgment, recorded one digit at a time; the recipe consumed eight thousand of
			them before it could read a single stroke of yours. Labels are supervised learning's fuel and
			its ceiling — expensive to collect, and the model learns to predict them and nothing more. So
			delete them. Keep the ten thousand images and burn every answer: is anything left to learn?
			The <a href={resolve('/latent')}>next chapter</a> squeezes the digits through a two-number bottleneck
			with no labels at all — and a map of handwriting appears on its own.
		</p>
	</Prose>
</ChapterShell>
