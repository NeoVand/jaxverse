<script lang="ts">
	import ChapterRef from '$lib/components/ui/ChapterRef.svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
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
			Two chapters ago you fit a curve through points; in the last one you bent space until two
			tangled classes came apart. Both were the same ritual, and the ritual has a name we never
			stopped to say: <em>supervised learning</em>. Here is the whole recipe. Collect a dataset of
			paired examples — questions with their answers already attached:
		</p>
		<Math
			display
			tex={'\\mathcal{D} \\;=\\; \\{(\\htmlClass{eq-world}{x^{(i)}},\\, \\htmlClass{eq-world}{y^{(i)}})\\}_{i=1}^{N}'}
		/>
		<p>
			Choose a family of functions <Math
				tex={'\\htmlClass{eq-out}{\\hat{y}} = f(\\htmlClass{eq-world}{x};\\htmlClass{eq-model}{\\theta})'}
			/>. Choose a loss
			<Math tex={'\\mathcal{L}'} /> that scores how much a prediction disagrees with the attached answer.
			Then run the only rule this book uses, <Math
				tex={'\\htmlClass{eq-model}{\\theta} \\leftarrow \\htmlClass{eq-model}{\\theta} - \\htmlClass{eq-knob}{\\eta} \\htmlClass{eq-world}{\\nabla \\mathcal{L}}'}
			/>, averaged over examples. Fitting a curve was this recipe with x and y both single numbers.
			The tangled spirals of <ChapterRef slug="space" /> were this recipe with x a point on a plane. Nothing
			in the recipe says x must stay small.
		</p>
		<p>
			So make x an image. MNIST is the field's old warhorse<Cite id="lecun-1998" />: handwriting
			collected at the start of the 1990s from Census Bureau clerks and American high-school
			students, every digit scanned and reduced to a 28 × 28 grid of gray. Read the grid row by row
			and the picture becomes a list of 784 brightnesses — a single point in
			<Math tex={'\\mathbb{R}^{784}'} />, exactly as a point on a plane was a list of two
			coordinates. In that space, “every 7 anyone might write” is not an idea but a <em>place</em>:
			a sprawling, tangled region, interleaved with the region of 1s and the region of 9s. The
			classifier's job is <ChapterRef slug="space" />'s job, word for word — deform the space until
			the ten regions come apart and flat cuts can separate them. The only thing you lose in 784
			dimensions is the ability to watch the bending. From here on, we judge the deformation by its
			consequences. Before any model touches them, look at the raw material.
		</p>
	</Prose>

	<Plate
		id="dataset"
		title="The dataset"
		caption="A random page from the training set — eight thousand of these teach the model on this page, and two thousand more are held back to grade it, each a 28 × 28 grid of gray off somebody's pen. Reshuffle a few times and the difficulty announces itself: 4s that could be 9s, 3s a squint away from 5s, 1s with and without a foot. Whatever rule tells them apart, nobody will get to write it down — it will have to be learned."
	>
		<Gallery />
	</Plate>

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

	<PipelineDiagram />

	<Prose>
		<p>The squashing step is one line of arithmetic,</p>
		<Math
			display
			tex={'\\htmlClass{eq-op}{\\operatorname{softmax}}_c(z) \\;=\\; \\frac{e^{z_c}}{\\sum_{j} e^{z_j}}'}
		/>
		<p>
			— each score exponentiated, then normalized, so the numbers are positive and sum to one. An
			equation like this is better felt than read, so play with it: a bank of scores drawn from a
			bell curve, and the belief they become. Watch the biggest score take most — never all — of the
			probability; widen the spread <Math tex="\sigma" /> and the winner runs away with it; and see how
			dividing the logits by a <em>temperature</em> before the softmax sharpens or flattens the verdict.
		</p>
	</Prose>

	<SoftmaxPlay
		title="The softmax, by hand"
		caption="Above: the raw scores, positive up and negative down, with the shape of their distribution beside them. Below: what the softmax makes of them. Move any dial and every probability shifts — belief is a budget of exactly one, spent by comparison. The temperature τ rescales the scores first: below one it exaggerates their differences; above one it shrugs them off. With a handful of scores you can also drag a bar to set it by hand."
	/>

	<Prose>
		<p>
			The second piece of machinery is the loss. Ten probabilities came out; one of them belongs to
			the right answer. That single number is the only one the loss looks at, and the loss is its
			negative logarithm:
		</p>
		<Math
			display
			tex={'\\mathcal{L}(\\htmlClass{eq-model}{\\theta}) \\;=\\; -\\log \\htmlClass{eq-out}{p_{\\htmlClass{eq-world}{y}}}, \\qquad \\htmlClass{eq-out}{p_{\\htmlClass{eq-world}{y}}} \\;=\\; \\htmlClass{eq-op}{\\operatorname{softmax}}_{\\htmlClass{eq-world}{y}}\\!\\big(f(\\htmlClass{eq-world}{x};\\htmlClass{eq-model}{\\theta})\\big)'}
		/>
		<p>
			Take the two symbols in turn. <Math tex={'\\htmlClass{eq-world}{y}'} /> is the true label, the answer
			written on the form by whoever held the pen; it never changes, and it selects exactly one of the
			ten bars.
			<Math tex={'\\htmlClass{eq-out}{p_{\\htmlClass{eq-world}{y}}}'} /> is the height of that one bar
			— the probability the model was willing to place on the truth. The nine other bars appear nowhere
			in the formula, and yet they are punished all the same: they share one unit of belief with
			<Math tex={'\\htmlClass{eq-out}{p_{\\htmlClass{eq-world}{y}}}'} />, so the only way to raise
			it is to take from them.
		</p>
		<p>
			The minus log is what turns a probability into a complaint. It is zero when
			<Math tex={'\\htmlClass{eq-out}{p_{\\htmlClass{eq-world}{y}}} = 1'} /> — perfect confidence in the
			right answer costs nothing — and it climbs slowly at first, then without limit as the probability
			approaches zero. One part in ten, which is where an untrained model spreading its belief evenly
			over ten classes starts, costs <Math tex="-\log 0.1 \approx 2.30" />; a tepid but correct 0.6
			costs 0.51; a confident 0.99 costs 0.01; and a confident, <em>wrong</em> 0.01 costs 4.61. That
			asymmetry is the whole design. Cross-entropy is often called the model's <em>surprise</em> at
			the truth, and being loudly, specifically wrong is the most surprising thing it can do — so
			that is where the gradient pushes hardest. Every digit in the plate below carries its own
			surprise, and the
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
		<p>
			Even this discipline leaks. A test set is only held out until people start choosing what to
			keep by how it scores, and then, one published result at a time, the field begins fitting to
			it too. When researchers built a fresh test set for a famous image benchmark by repeating the
			original collection procedure as closely as they could, every model scored worse on the new
			one — not catastrophically, but consistently, in the order of their original ranks.<Cite
				id="recht-2019"
			/> Nothing here was cheated. The number was simply optimistic in a way nobody could see from inside.
		</p>
	</Prose>

	<Classifier
		title="The classifier"
		caption="Sixteen digits from the test set — the model never trains on them. Under each, ten bars: its whole belief, one per class. Then the reading, green when right and red when wrong, and beside it the surprise −log p that cross-entropy charges for it. Untrained, the bars are flat and every surprise sits near 2.30; train, and watch the beliefs spike as the readings correct themselves. Random draws a fresh sixteen, but the other two buttons rank the whole test set by the probability it gives the right answer and show you an end of it: Surest for the digits it has no trouble with, Hardest for the ones it gets confidently, expensively wrong. The right column is the machine itself: change its depth, its width, or the bend it uses, and it starts over with the new shape."
	/>

	<UnderTheHood slug="digits" block="training" />

	<Prose>
		<h2 class="h2">What the machine actually receives</h2>
		<p>
			What the model is given is not what you see. You see a 7 — two strokes, a sharp corner, a
			thing with a top. It receives a list:
			<code>0.00, 0.00, 0.31, 0.89, 0.94, …</code> — 784 numbers in a fixed order, and nothing else.
			It is not told that pixel 215 sits directly above pixel 243; adjacency, stroke, corner, loop —
			every visual notion has to be rediscovered statistically, from which brightnesses rise and
			fall together across eight thousand examples. That “seven-ness” can be recovered from a bare
			list of numbers is the quiet astonishment of this chapter, and the mechanism is the one you
			already watched: the same bending of space as <ChapterRef slug="space" />, performed in 784
			dimensions instead of two.
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
			more data beats cleverness, almost every time it is allowed to compete. And not vaguely:
			across vision, speech and language, error falls as a power law in the number of training
			examples, with enough regularity that you can forecast what another ten times the data will
			buy before you go and collect it.<Cite id="hestness-2017" />
		</p>
		<p>
			The 2,000 test digits are still other people's handwriting, scanned by the same machines in
			the same decade. Your handwriting, drawn with a pointer, is a genuinely harder exam. One
			courtesy is extended before each guess: MNIST digits are centered by their center of mass, so
			the pad below re-centers your stroke the same way — the model has never seen a digit hiding in
			a corner, and without that shift it would fail for reasons that have nothing to do with shape.
		</p>
		<p>
			Beside the pad sits an <em>evidence square</em>, and it arrives with a caution attached. It is
			the gradient of the winning score with respect to each pixel: brighten this pixel a little,
			and the score for the model's answer goes up by this much, or down. Laid out as an image, that
			vector is a <em>saliency map</em>, and it does look like an explanation.<Cite
				id="simonyan-2014"
			/> Read it as one carefully. A gradient describes the model's behaviour in an infinitesimal neighbourhood
			of exactly this input, which is not the same thing as the reason for the verdict — and some popular
			relatives of this method have been caught producing the same handsome picture even after the network's
			weights are replaced with random numbers.<Cite id="adebayo-2018" /> A picture that survives the
			destruction of the model was never describing the model. The bare gradient shown here is the honest,
			unretouched version: noisier than the pretty ones, and answering a question it can actually answer.
		</p>
	</Prose>

	<Plate
		id="drawpad"
		live
		title="Draw your own"
		caption="A 3 is waiting on the pad; wipe it with the eraser and draw your own. The ring is your brush, and the slider on the left fattens or thins it. On the right, the same two charts as the softmax plate above, now fed by your stroke: ten raw scores over a zero rule, and the belief they become. The evidence square is the gradient of the winning score with respect to each pixel — ultramarine pixels argued for the verdict, vermilion pixels argued against it, so you can see which parts of your stroke the model leaned on."
	>
		<DrawPad />
	</Plate>

	<UnderTheHood slug="digits" block="saliency" />

	<Prose>
		<h2 class="h2">Opening the box</h2>
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

	<Plate
		id="inside"
		live
		title="Inside the machine"
		caption="Left: an explorer over every learned layer, with two ways to look at a unit. Weights draws what it is made of — a first-layer unit owns 784 incoming weights, one per pixel, so it simply is an image, and a deeper unit is carried back to pixel space by multiplying out the matrices behind it, its linear shadow. Excites it drops the algebra and asks the test set: the two dozen digits that drive the unit hardest, averaged. Walk out to the readout in that view and the ten class templates come back as ten clean digits — the machine's idea of each name. Right: the confusion matrix over all 2,000 test digits, rows truth and columns its reading. Hover any cell to open the digits it actually got that way; the heaviest are usually 4 read as 9 and 3 read as 5, mistakes of shared shape."
	>
		<Inside />
	</Plate>

	<Prose>
		<p>
			The machine's idea of a 7 is further from yours than its mistakes suggest. Take an image it
			reads correctly and confidently, compute which direction in pixel space would most reduce that
			confidence, and step a little way along it — a change so small that the two pictures are
			indistinguishable side by side. The verdict flips, often to a confident wrong answer. These
			are <em>adversarial examples</em>, and they are not a bug in one model: they were found in
			every network anyone tried, and the same doctored image usually fools models trained
			separately on different data.<Cite id="szegedy-2014" /> The explanation is less exotic than the
			effect. In a space of 784 dimensions, a nudge too small to see in any single pixel is a large step
			overall if you take it in the one direction the model is most sensitive to — and a network built
			out of nearly-linear pieces has such a direction everywhere.<Cite id="goodfellow-adv-2015" /> The
			model has genuinely learned something about handwriting. It has not learned what you would call
			seeing.
		</p>
		<h2 class="h2">The workhorse and its price</h2>
		<p>
			What you just trained is not a toy version of the real thing; it is the real thing, small.
			This exact recipe — pixels in, softmax out, cross-entropy pushed downhill — was reading the
			amounts on several million American bank checks a day by the late 1990s<Cite
				id="lecun-1998"
			/>, and sorting mail by ZIP code a decade before that.<Cite id="lecun-1990" /> Scaled up, it reads
			street signs for cars and flags tumors on scans. Whenever the world hands you questions paired with
			answers, supervised learning is the tool you reach for first, and usually the tool that wins.
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
