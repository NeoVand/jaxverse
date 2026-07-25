<script lang="ts">
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import Wide from '$lib/components/ui/Wide.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Gallery from '$lib/components/demos/digits/Gallery.svelte';
	import SoftmaxPlay from '$lib/components/demos/digits/SoftmaxPlay.svelte';
	import Classifier from '$lib/components/demos/digits/Classifier.svelte';
	import DrawPad from '$lib/components/demos/digits/DrawPad.svelte';
	import Inside from '$lib/components/demos/digits/Inside.svelte';
	import { lab } from '$lib/components/demos/digits/digits-context.svelte';
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
			Two small pieces of machinery make ten classes workable. The network now ends in ten outputs —
			one score per digit, a vector called the <em>logits</em>. The <em>softmax</em> squashes those scores
			into probabilities,
		</p>
		<Math display tex={'\\operatorname{softmax}_c(z) \\;=\\; \\frac{e^{z_c}}{\\sum_{j} e^{z_j}}'} />
		<p>
			— each score exponentiated, then normalized, so the ten numbers are positive and sum to one.
			An equation like this is better felt than read, so play with it: six scores under your
			control, and the belief they become. Watch the biggest score take most — never all — of the
			probability, and how dividing the logits by a <em>temperature</em> before the softmax sharpens or
			flattens the verdict.
		</p>
	</Prose>

	<Wide>
		<Plate
			title="The softmax, by hand"
			caption="Move a score and every probability shifts — belief is a budget of exactly one, spent by comparison. The temperature T rescales the scores first: below one it exaggerates their differences; above one it shrugs them off."
		>
			<SoftmaxPlay />
		</Plate>
	</Wide>

	<Prose>
		<p>
			And the loss, called <em>cross-entropy</em>, is the negative log of whatever probability the
			model gave to the correct class:
		</p>
		<Math
			display
			tex={'\\mathcal{L}(\\theta) \\;=\\; -\\log \\operatorname{softmax}_{y}\\!\\big(f(x;\\theta)\\big)'}
		/>
		<p>
			Read it as the model's surprise at the right answer. Sure and correct: probability near one,
			loss near zero. Sure and wrong: probability near zero, loss enormous. Gradient descent on this
			loss pushes the correct logit up and the other nine down, hardest exactly when the model is
			confidently mistaken.
		</p>
		<p>
			One more piece of discipline — the piece that makes this science rather than wishful thinking.
			A network with a hundred thousand parameters can <em>memorize</em> thousands of examples: loss
			falling, accuracy climbing, and nothing learned about handwriting at all. Memorizing answers
			is not learning, so we grade on questions the model has never seen. Of the ten thousand digits
			below, 8,000 form the <em>training set</em> the gradients flow from, and 2,000 form the
			<em>test set</em>, locked away from training entirely. Every accuracy quoted in this chapter
			is measured on held-out digits alone. It is the only number that means anything.
		</p>
	</Prose>

	<Wide>
		<div id="plate-classifier">
			<Plate
				n={2}
				title="The classifier"
				caption="Twelve digits from the test set — the model never trains on them — with its current reading under each; wrong readings in red. Untrained, it guesses at chance, one in ten. Train, and watch the labels correct themselves."
			>
				{#snippet status()}
					{#if lab.phase === 'ready'}
						<span>
							step {lab.step} · loss {Number.isFinite(lab.loss) ? lab.loss.toFixed(3) : '—'} · test
							{Number.isFinite(lab.testAcc) ? (lab.testAcc * 100).toFixed(1) + '%' : '—'} · {lab.msPerStep.toFixed(
								0
							)} ms/step
						</span>
					{/if}
				{/snippet}
				<Classifier />
			</Plate>
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
			optimizer is paid to push it down, and with 109,386 parameters there is room to push. The
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
			caption="Draw a digit with your pointer — the circle is your brush, and the σ slider fattens or thins it. The ten bars are the softmax, the model's belief updating as you draw. The evidence map is the gradient of the winning score with respect to each pixel: warm pixels argued for the verdict, blue pixels argued against it."
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
			this machine is sealed. The first layer maps 784 pixels to 128 units, so each unit owns
			exactly 784 incoming weights — one per pixel — which means <em>each unit is an image</em>, and
			you can simply look at it: the pattern of ink it rewards and the pattern it penalizes. And its
			mistakes on the test set have structure. It does not confuse digits at random; it confuses
			digits that share a skeleton, the same pairs you would confuse squinting at a bad fax.
		</p>
	</Prose>

	<Wide>
		<Plate
			n={4}
			title="Inside the machine"
			caption="Left: the incoming weights of 24 first-layer units, each reshaped back into a 28 × 28 image — the templates it presses against your digit. Right: the confusion matrix over all 2,000 test digits; the green diagonal is truth, the red smudges are error. The heaviest are 4 read as 9 and 3 read as 5 — mistakes of shared shape."
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
