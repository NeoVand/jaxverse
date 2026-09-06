<script lang="ts">
	import ChapterRef from '$lib/components/ui/ChapterRef.svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import Paths from '$lib/components/demos/flow/Paths.svelte';
	import Steps from '$lib/components/demos/flow/Steps.svelte';
	import Studio from '$lib/components/demos/flow/Studio.svelte';
	import Guidance from '$lib/components/demos/flow/Guidance.svelte';
	import Compose from '$lib/components/demos/flow/Compose.svelte';
	import Styles from '$lib/components/demos/flow/Styles.svelte';
	import { resolve } from '$app/paths';
</script>

<ChapterShell slug="flow">
	<Prose>
		<p>
			The model in <ChapterRef slug="noise" /> could draw, and it charged fifty forward passes for every
			picture. The reason was geometric: it had been taught to follow a path that curves, and you cannot
			take long strides along a curve without leaving it. This chapter asks what happens if you pick a
			straighter path — and the answer turns out to be shorter, plainer, and, once you have it, steerable
			in a way the curved version never was.
		</p>
		<p>
			The last chapter chose carefully between two schedules and never asked whether it needed one
			of that kind at all. Ask now. Look at what training actually requires: a way to produce a
			corrupted picture, a number saying how corrupted it is, and a target to be scored against. Any
			interpolation between a picture and some noise supplies all three. Both of the last chapter's
			candidates got their shape from making a derivation come out cleanly, and neither was chosen
			for being convenient to walk along.
		</p>
		<p>So use the shortest one there is. A straight line.</p>
		<Math
			display
			tex={'\\htmlClass{eq-world}{x}_{\\htmlClass{eq-knob}{\\tau}} \\;=\\; (1 - \\htmlClass{eq-knob}{\\tau})\\,\\htmlClass{eq-world}{x}_0 \\;+\\; \\htmlClass{eq-knob}{\\tau}\\,\\htmlClass{eq-mute}{\\varepsilon}.'}
		/>
		<p>
			At <Math tex={'\\htmlClass{eq-knob}{\\tau} = 0'} /> that is the picture; at <Math
				tex={'\\htmlClass{eq-knob}{\\tau} = 1'}
			/> it is pure noise; in between it slides evenly from one to the other. Now differentiate it, which
			takes one line, because a straight line has the same slope everywhere:
		</p>
		<Math
			display
			tex={'\\frac{d\\,\\htmlClass{eq-world}{x}_{\\htmlClass{eq-knob}{\\tau}}}{d\\htmlClass{eq-knob}{\\tau}} \\;=\\; \\htmlClass{eq-mute}{\\varepsilon} - \\htmlClass{eq-world}{x}_0.'}
		/>
		<p>
			That quantity is the <em>velocity</em>: the direction and speed at which this particular
			picture is turning into this particular noise. Train the network to predict it, with the same
			squared error as before, and you have swapped one target for another and changed nothing else
			about the machine — same transformer, same conditioning, same optimizer, same corpus.
		</p>
		<Math
			display
			tex={'\\mathcal{L}(\\htmlClass{eq-model}{\\theta}) \\;=\\; \\mathbb{E}\\left[\\;\\big\\lVert\\, (\\htmlClass{eq-mute}{\\varepsilon} - \\htmlClass{eq-world}{x}_0) \\;-\\; \\htmlClass{eq-out}{\\hat v}_{\\htmlClass{eq-model}{\\theta}}(\\htmlClass{eq-world}{x}_{\\htmlClass{eq-knob}{\\tau}},\\, \\htmlClass{eq-knob}{\\tau})\\,\\big\\rVert^2\\;\\right].'}
		/>
		<p>
			Sampling is now the plainest thing in this book. You are standing at some point with an
			estimate of your velocity, and you want to be a little further along: multiply and subtract.
			That is Euler's method, from 1768, and it is the entire sampler. No schedule, no square roots,
			no variance bookkeeping — take the step.
		</p>
		<p>
			There is one catch, and the plates below will make it look smaller than it is. The straight
			line is straight for <em>one pair</em> — this picture and this noise. Training sees the pair; sampling
			never does. Standing at some point halfway out, the model cannot know which picture it is on the
			way to, so the velocity it reports is an average over every pair that could have put it there —
			a velocity belonging to none of them. Follow those averages and you trace a curve, not a line.
		</p>
		<p>
			It is a far gentler curve, and it can be flattened further still by training a second time on
			the model's own start-and-finish pairs.<Cite id="liu-2023-rectflow" /> Three groups posted versions
			of the idea within a month of each other in the autumn of 2022 — as rectified flow, as
			<em>flow matching</em>
			over arbitrary interpolants,<Cite id="lipman-2023" /> and as stochastic interpolants between two
			densities<Cite id="albergo-2023" /> — and all three are, underneath, the observation that you can
			regress on a velocity you know in closed form. The lineage runs back to treating a network as the
			derivative of a trajectory and letting a solver supply the layers.<Cite id="chen-node-2018" />
		</p>
		<p>
			The claim is testable on this page, so test it. Both models are here: the noise-predicting one
			from the last chapter and the velocity-predicting one from this, the same size, trained on the
			same pictures for the same length of time. At every rung of the walk you can ask each of them
			the same question — <em
				>given where you are, what finished picture do you think you are heading toward?</em
			> A straight path should have a confident answer early. A curved one should keep revising.
		</p>
	</Prose>

	<Paths
		title="Where each model thinks it is going"
		caption="Not the pictures as they are, but the pictures each model believes it is on its way to, read off at eight moments during a forty-eight-step walk from the same static. What to watch for is how early each row stops changing its mind: a straight path should settle on a destination and then only sharpen it, while a curved one keeps revising where it is going right up to the end. The rows are clamped identically and both skip the first moment of the walk, where dividing by an almost-vanished signal saturates whatever either model said."
	/>

	<Prose>
		<p>
			Now spend a smaller budget. Sampling cost is steps times the cost of one forward pass, and the
			forward pass is fixed, so the only question to ask of a sampler is how few steps it can get
			away with. The plate below answers it, and hands over something that was not asked for: at
			this size, the two models differ more in kind than in stamina.
		</p>
	</Prose>

	<Steps
		title="The same weights, on a smaller budget"
		caption="Five emoji from each model at fifty steps, then twenty, ten, four and two — the same seeds in every row, and neither model retrained between them. Read across before reading down, because the first thing the plate shows is not about step counts at all: the velocity model puts an object on an empty background, and the noise model fills the frame edge to edge. At this size the choice of target changed the pictures more than the budget does. Then read down the velocity column, which is where the argument above can actually be tested: it holds its shape to ten and thins out at four. The noise column barely changes until two, for the unhappier reason that it has no crisp version to degrade from."
	/>

	<Prose>
		<p>
			Be careful about what that proves. Two models this small, given the same handful of hours,
			cannot settle which parameterization is better in general — and the gap the plate shows is
			partly the well-known trouble the noise target has when the picture is nearly gone, where a
			small error in the predicted noise is divided by a very small number on the way back to a
			predicted picture. The honest summary is that the velocity target was easier to get working
			here, and that it degrades more gracefully when the budget is cut. Both of those are why
			production image models are built the way they are. Stable Diffusion 3's two structural
			choices were a rectified-flow objective — with a new way of choosing which noise levels to
			train on — and a transformer over patches, in their case one carrying separate weights for the
			text and image streams.<Cite id="esser-2024" /> Strip those refinements and what is left is the
			pair of choices these two chapters make, at a few thousand times the size. The step count is not
			a detail of the research; it is most of what a user of one of these systems experiences.
		</p>
		<p>
			It is also not the end of the line. If a straighter path buys fewer steps, a student trained
			to take in one step what its teacher takes in two buys fewer again,<Cite
				id="salimans-ho-2022"
			/> and a model trained to jump from anywhere on the path directly to its endpoint collapses the
			walk into a single evaluation.<Cite id="song-2023-consistency" /> Each of those is the same trade:
			some fidelity, and a great deal of arithmetic, exchanged for a shorter walk.
		</p>
		<h2 class="h2">Telling it what to draw</h2>
		<p>
			Everything you have been shown so far has been unconditional. The model draws an emoji — some
			emoji, whichever one the noise it started from happens to become — and you have no say.
			Getting a say is structurally easy, and the easy version turns out to hide something else.
		</p>
		<p>
			The model already takes a second input besides the image: the noise level, which enters as a
			scale and a shift applied inside every block. Widen that input. Alongside the noise level,
			pass a vector saying which of 310 tag words describe this picture — <em>cat</em>,
			<em>heart</em>, <em>red</em>, <em>face</em> — and another saying which of the eight drawing
			styles it came from. The tags are taken from the Unicode names and keywords the emoji already
			ship with, so nobody labelled anything by hand. A typed prompt becomes a bag of those words,
			which is a crude idea of language and exactly enough for a vocabulary of a thousand pictures.
			Real systems replace this part with a text encoder trained to put captions and images in one
			space<Cite id="radford-2021" />; the wiring downstream of it is unchanged.
		</p>
		<p>
			Then the trick. During training, throw the prompt away one time in ten and the style one time
			in ten, independently, and let the model see the picture with no description attached. One set
			of weights therefore learns two things at once: what a <em>cat</em> looks like, and what a picture
			looks like when nothing has been asked for. At sampling time you run both and read the difference.
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-out}{\\hat v}_{\\text{used}} \\;=\\; \\underbrace{\\htmlClass{eq-out}{\\hat v}(\\htmlClass{eq-world}{x}, \\varnothing)}_{\\text{any picture}} \\;+\\; \\htmlClass{eq-knob}{w}\\,\\big[\\,\\underbrace{\\htmlClass{eq-out}{\\hat v}(\\htmlClass{eq-world}{x}, \\htmlClass{eq-model-2}{c}) - \\htmlClass{eq-out}{\\hat v}(\\htmlClass{eq-world}{x}, \\varnothing)}_{\\text{what the prompt adds}}\\,\\big].'}
		/>
		<p>
			The middle term is doing something slightly startling. The difference between what the model
			would draw with the prompt and what it would draw without it is a vector at every point in
			image space — it is the direction the prompt is pulling. Multiplying it by <Math
				tex={'\\htmlClass{eq-knob}{w}'}
			/> and adding it back is insisting on that direction harder than the model would on its own. At
			<Math tex={'\\htmlClass{eq-knob}{w} = 1'} /> you get the conditional model exactly as trained; above
			that you get a picture more prompt-like than any real picture, which is why the Stable Diffusion
			lineage ships this dial with a default up around 7. This model, being small and steered by tag words
			rather than sentences, wants much less — its useful setting is nearer two, and the plate below is
			how you would find that out. This is
			<em>classifier-free guidance</em>, and it replaced an earlier version that needed a separately
			trained classifier to supply the push.<Cite id="dhariwal-nichol-2021" /> Dropping that classifier
			and getting the same effect from one network's two moods is the whole contribution.<Cite
				id="ho-salimans-2022"
			/>
		</p>
	</Prose>

	<Studio
		title="The studio"
		caption="Type something and the model draws eight of them, here, from weights that came down with this page. The prompt is matched against a 310-word vocabulary built out of the emoji's own Unicode names — matched words are shown as chips and unmatched ones are struck through, because a prompt that quietly did nothing would be the most confusing thing on this page. Pick a hand to draw in or leave it on any. Guidance is the dial from the equation above; steps is the budget from the plate above, and you can watch for yourself what dropping it to four costs."
	/>

	<UnderTheHood slug="flow" block="steer" />

	<Prose>
		<p>
			Guidance has two failure modes, and the good setting is defined by them rather than by any
			principle. Turn it down and the prompt stops mattering. Turn it up and every picture becomes
			the same over-saturated caricature: the model is being pushed toward a region that maximizes
			prompt-likeness rather than one that contains pictures, and variety is the first thing it
			spends.
		</p>
	</Prose>

	<Guidance
		title="How hard to insist"
		caption="The same prompt and the same four seeds at five strengths of guidance. At zero the prompt is not consulted at all and the four seeds go four unrelated ways. At one the colour has arrived but the obedience is patchy — that is the conditional model exactly as it was trained. Two is where this model is happiest: the prompt is answered and the four seeds still disagree about how. By four they have stopped disagreeing, and eight adds nothing but saturation. Past the middle of this dial the pictures stop being samples and start being an argument."
	/>

	<Prose>
		<h2 class="h2">Two things at once</h2>
		<p>
			Now the reason the arithmetic was written out term by term. The guidance term is a vector
			field: at every point in image space, the prompt contributes a direction. There is nothing
			special about there being one of them. Give the model a second prompt, compute a second
			difference, and add that too.
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-out}{\\hat v}_{\\text{used}} \\;=\\; \\htmlClass{eq-out}{\\hat v}(\\htmlClass{eq-world}{x}, \\varnothing) \\;+\\; \\htmlClass{eq-knob}{w_A}\\big[\\htmlClass{eq-out}{\\hat v}(\\htmlClass{eq-world}{x}, \\htmlClass{eq-model}{A}) - \\htmlClass{eq-out}{\\hat v}(\\htmlClass{eq-world}{x}, \\varnothing)\\big] \\;+\\; \\htmlClass{eq-knob}{w_B}\\big[\\htmlClass{eq-out}{\\hat v}(\\htmlClass{eq-world}{x}, \\htmlClass{eq-model-2}{B}) - \\htmlClass{eq-out}{\\hat v}(\\htmlClass{eq-world}{x}, \\varnothing)\\big].'}
		/>
		<p>
			Two pushes, applied to the same walk, with a weight each. Under a reading of these models as
			energy-based, that sum is what conjunction <em>means</em> — asking for a picture that is
			simultaneously probable under both conditions — which is where the recipe comes from and why
			it behaves better than averaging the two prompts into one bag would.<Cite
				id="liu-2022-composable"
			/>
		</p>
		<p>
			Nothing in training prepared for this. The corpus contains a cat and it contains a heart. It
			does not contain a cat with a heart for a face, in any of its eight styles, and no gradient
			ever pointed toward one. The combination is manufactured at sampling time, out of two
			directions that were learned separately and happen to be addable. Turn the two weights up and
			down and you can walk between the two ideas, through pictures that exist nowhere.
		</p>
	</Prose>

	<Compose
		title="A moon, a face, and something in between"
		caption="Three rows from the same six seeds: the first prompt on its own, the second on its own, and both pushes applied to one walk. Only the third row contains anything new — the first two are the model doing what it was trained to do. Move the weights and the third row slides between its neighbours; take one to zero and it becomes the other row. Moon and smiling face is the clearest pair to start on, because the two ingredients are separable and the result is neither: a plain yellow disc from one, a set of features from the other, and a moon with a face on it that appears in no style in the corpus. Other pairs mix less tidily and settle for blending the two palettes, which is its own kind of honest."
	/>

	<Prose>
		<p>
			The style condition is a separate block of that vector, dropped separately during training,
			and so it can be steered separately too. That separation is the only reason the corpus was
			built the way it was: eight drawings of every idea, so that <em>what a thing is</em> and
			<em>how it is drawn</em> vary independently in the data and the model has no way to confuse them.
			Ask for the same idea eight times, changing only the style, and you can see how well that worked.
		</p>
	</Prose>

	<Styles
		title="One idea, eight hands"
		caption="A single prompt and a single set of seeds, drawn once in each style the model was trained on: Google's Noto, Twemoji, OpenMoji in colour and in bare line art, three renderings of Microsoft's Fluent set, and the blobs Google retired in 2017. The silhouette should stay roughly put while the treatment changes — glossy gradients here, flat fills there, and in the line-art row a picture with almost no colour in it at all. Where a row breaks down, it is usually a style that drew this particular idea unusually and the model only had one example to go on."
	/>

	<Prose>
		<p>
			The size of what is on this page: 2.6 million parameters, trained on 8,656 pictures at 32 ×
			32, three and a half orders of magnitude below a production image model in parameters and five
			in pictures. It shows in every blurred edge. The recipe is not smaller — the interpolation,
			the velocity target, the Euler sampler, the dropped condition, the guidance subtraction, the
			second push added on top.
		</p>
		<p>
			One structural difference remains, and it is where the arithmetic happens. Running diffusion
			on megapixel images directly is wasteful, so production systems compress first and decode at
			the end, and the thing doing the compressing is an autoencoder.<Cite id="rombach-2022" /> The hourglass
			from <ChapterRef slug="latent" /> is still in the machine six chapters later, holding the same waist
			open for the same reason.
		</p>
		<p>
			And that is the last of them. In the Prologue there was a blind walker on a smooth landscape,
			feeling the ground for a downhill direction and taking one small step. Everything since has
			been that walker on richer ground: a curve bent to fit any shape you drew; space folded until
			two tangled colours came apart; handwriting read out of 784 numbers and no picture; a map that
			drew itself with nobody holding the pen; a sentence learning to guess its own next word; a
			boat that found the wind with nothing but a clock; a judge fitted to your eye and then broken
			by an optimizer; a pocket of numbers taught chess by nothing but games — and, these last two
			chapters, a machine that is handed nothing at all and hands back a picture. If you want the
			whole descent laid out, the <a href={resolve('/epilogue')}>epilogue</a> lists it, says who to thank,
			and tells you where to go next.
		</p>
	</Prose>
</ChapterShell>
