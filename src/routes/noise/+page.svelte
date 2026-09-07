<script lang="ts">
	import ChapterRef from '$lib/components/ui/ChapterRef.svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import PlateRef from '$lib/components/ui/PlateRef.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import Ladder from '$lib/components/demos/noise/Ladder.svelte';
	import DenoiserDiagram from '$lib/components/demos/noise/DenoiserDiagram.svelte';
	import Learn from '$lib/components/demos/noise/Learn.svelte';
	import Walk from '$lib/components/demos/noise/Walk.svelte';
</script>

<ChapterShell slug="noise">
	<Prose>
		<p>
			Every machine in this book so far has been asked a question and has answered it. Name this
			digit. Say the next word. Choose a move. Even the autoencoder of
			<ChapterRef slug="latent" />, which drew rather than named, was really answering:
			<em>here is a digit, give it back to me</em>. This chapter builds the first machine that is
			handed nothing and produces something anyway.
		</p>
		<p>
			That chapter left a debt. Its decoder rebuilt each digit from two numbers and the results came
			back soft — threes you could see through. The diagnosis was that the blur belonged to the loss
			and not to the network: a model asked for one picture and scored on squared error will hedge,
			and the hedge that minimizes squared error is the average of every digit that could plausibly
			have been meant. Averages of pictures are fog. That page named the fix in a sentence and moved
			on: change the question, not the size of the machine. This chapter is the sentence, unpacked.
		</p>
		<p>
			Here is the changed question. It is smaller, not bigger. Do not ask for a picture. Asking for
			a picture demands seven hundred and eighty-four numbers committed at once with nothing in
			hand, and against that demand hedging is genuinely the best available strategy. Ask instead
			for something close to trivial: <em
				>here is a picture with a little noise sprinkled on it — which noise?</em
			> That is a question about a small local correction rather than about the whole picture, and a small
			local correction can be got nearly right. A model that answers it fifty times in a row, each time
			on its own slightly cleaner output, will have produced an image without ever once being asked to
			imagine one.
		</p>
		<p>
			The idea is from 2015, in a paper that had the whole shape of the thing and then waited five
			years for the parameterization, the architecture and the compute that made it work.<Cite
				id="sohl-dickstein-2015"
			/> Its authors took the analogy from physics: a drop of ink in water spreads until the water is
			uniformly grey, and the spreading is easy to describe. Running it backwards is not easy. But if
			each forward step is small enough, each backward step has the same simple shape as the forward one
			— so a model that only ever has to undo a
			<em>small</em> step can be trained to undo the whole thing, one small step at a time.
		</p>
		<h2 class="h2">Ruining a picture, on a schedule</h2>
		<p>
			Half of this is free. Destroying an image takes no learning at all: read the pixels as a
			vector, draw a vector of Gaussian noise the same shape, and mix the two in a ratio you choose.
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-world}{x}_{\\htmlClass{eq-knob}{\\tau}} \\;=\\; \\sqrt{\\htmlClass{eq-knob}{\\bar\\alpha(\\tau)}}\\,\\htmlClass{eq-world}{x}_0 \\;+\\; \\sqrt{1 - \\htmlClass{eq-knob}{\\bar\\alpha(\\tau)}}\\;\\htmlClass{eq-mute}{\\varepsilon}, \\qquad \\htmlClass{eq-mute}{\\varepsilon} \\sim \\mathcal{N}(0, I).'}
		/>
		<p>
			Two symbols carry everything. <Math tex={'\\htmlClass{eq-knob}{\\tau}'} /> runs from 0 to 1 and
			says how far along the ruin you are; <Math tex={'\\htmlClass{eq-knob}{\\bar\\alpha}'} /> is the
			<em>schedule</em>, the fraction of the original signal still present at that point. The two
			square roots are chosen so the ingredients always add up to one unit of variance — the
			<em>variance-preserving</em> convention, which keeps the network's input from growing as the ruin
			deepens. (It holds exactly for data of unit variance. These garments sit nearer 0.71, so the scale
			does drift upward across the ladder; bounded and predictable is all the network needs.) Notice also
			what the formula does not contain — any mention of the steps in between. You can jump to any noise
			level in one multiplication, which is why training never has to simulate a chain.
		</p>
	</Prose>

	<Ladder
		title="Eleven rungs down"
		caption="One garment, mixed with the same fixed noise in eleven increasing proportions, intact at the left and indistinguishable from static at the right. The curve underneath is the schedule — how much of the picture survives at each rung — with the alternative drawn behind it as a dashed line. Switch from cosine to linear and watch the middle of the row collapse: under the schedule diffusion was first published with, the picture is essentially gone by the halfway rung, and everything past it is a level at which there is nothing much left to learn to remove. Change the class, or the picture, and the story is the same."
	/>

	<Prose>
		<p>
			The choice of schedule looks like a detail and is not. Under the original linear schedule the
			signal is essentially gone by the halfway rung, so the whole back half of the ladder is levels
			at which there is nothing left to learn to remove. Replacing it with a cosine, so the picture
			gives way gradually and holds through the middle, buys a clear improvement in likelihood and a
			real one in sample quality — but only at small sizes. The linear schedule is fine at 256
			pixels square and bad at 32, which is nearly the size this chapter works at.<Cite
				id="nichol-dhariwal-2021"
			/> Most of the constants in a working diffusion model are like this: not derived, but found by someone
			who plotted the right thing.<Cite id="karras-2022" />
		</p>
		<h2 class="h2">The only thing the model is asked</h2>
		<p>
			Now the half that has to be learned. Take a picture, pick a <Math
				tex={'\\htmlClass{eq-knob}{\\tau}'}
			/> at random, corrupt it by that much, and hand the result to a network along with the number
			<Math tex={'\\htmlClass{eq-knob}{\\tau}'} /> itself. Ask it to name the noise. You know the answer,
			because you drew it a moment ago, so the loss writes itself:
		</p>
		<Math
			display
			tex={'\\mathcal{L}(\\htmlClass{eq-model}{\\theta}) \\;=\\; \\mathbb{E}\\left[\\;\\big\\lVert\\, \\htmlClass{eq-mute}{\\varepsilon} \\;-\\; \\htmlClass{eq-out}{\\hat\\varepsilon}_{\\htmlClass{eq-model}{\\theta}}(\\htmlClass{eq-world}{x}_{\\htmlClass{eq-knob}{\\tau}},\\, \\htmlClass{eq-knob}{\\tau})\\,\\big\\rVert^2\\;\\right].'}
		/>
		<p>
			That is the entire training objective, and it is a squared error — the same loss whose hedging
			produced the fog in the first place. Be exact about what has and has not been fixed here,
			because the tempting answer is wrong. Nothing has been fixed about the hedge. Given a ruined
			picture, the average noise that might have been added and the average picture that might have
			been meant are the same estimate in two coordinate systems, related by the corruption formula
			run backwards; both are fog, and neither is sharper than the other.
		</p>
		<p>
			What changed is that the fog is no longer the output. The model's average is used as a
			<em>direction</em>. One short step is taken along it, the noise level drops by a notch, and
			the question is asked again from the new position — and because each step is short, the answer
			only has to be right locally, where the plausible pictures have not yet diverged. Fifty
			hedges, each trusted only a little way, compose into a sample. That is the whole difference
			between this and the autoencoder: not a better estimate of the mean, but a way of never having
			to publish one.
		</p>
		<p>
			The same loss has a second reading, and that one explains why any of this works. Write down
			the direction in which pictures become more common as you move — the gradient, at wherever you
			are standing, of the log density of the noisy data:
		</p>
		<Math
			display
			tex={'\\nabla\\,\\log q(\\htmlClass{eq-world}{x}_{\\htmlClass{eq-knob}{\\tau}}) \\;=\\; -\\,\\frac{\\htmlClass{eq-out}{\\hat\\varepsilon}(\\htmlClass{eq-world}{x}_{\\htmlClass{eq-knob}{\\tau}},\\, \\htmlClass{eq-knob}{\\tau})}{\\sqrt{1 - \\htmlClass{eq-knob}{\\bar\\alpha(\\tau)}}}.'}
		/>
		<p>
			That vector is called the <em>score</em>, and nobody can compute it: it asks for the density
			of all possible pictures, which no one has. The network trained by the paragraph above is
			reporting it anyway, at the optimum, up to the sign and the scale written here. Mind the
			minus. The model names the noise, and noise points <em>away</em> from where pictures live —
			which is exactly why the sampler subtracts its answer rather than following it. That denoising
			and score estimation are the same act was proved in 2011,<Cite id="vincent-2011" />
			extending an older method for fitting a distribution you can only evaluate up to an unknown constant.<Cite
				id="hyvarinen-2005"
			/> So the model was never learning this picture. It was learning the shape of the whole space of
			pictures, one noise level at a time — which is how diffusion came to be arrived at a second time
			from the other direction, as
			<em>score matching</em>, before anyone noticed the two lines of work were one.<Cite
				id="song-ermon-2019"
			/>
		</p>
	</Prose>

	<DenoiserDiagram
		title="What answers the question"
		caption="The denoiser is the transformer from The Next Token with three changes. The sequence is not words but the forty-nine four-by-four patches of a 28 × 28 square. Nothing is masked — a patch in the corner may attend to a patch in the middle, because an image has no past. And the noise level does not enter as a token; it is turned into a scale and a shift applied inside every block, which is how one set of weights behaves differently at every rung of the ladder. Out the far side comes a square the same size as the input, holding the model's claim about which noise is present. That side input has room for more than a noise level, and the next chapter uses it."
	/>

	<Prose>
		<p>
			The classical choice here is a different shape of network, one that slides small filters over
			the picture at full resolution. The first version of this chapter used one and it lost on the
			clock: 264 milliseconds a step against the transformer's 88, at the same batch on the same
			laptop, because everything the transformer does happens after the image has been cut down to
			forty-nine tokens. That is also the direction the field went, and for better reasons than ours
			— the patch transformer scales more gracefully,<Cite id="peebles-xie-2023" /> and the current generation
			of image models is built from it.
		</p>
		<p>
			The corpus is twelve thousand training pictures at 28 × 28, drawn from Zalando's
			Fashion-MNIST: ten classes of clothing, grayscale, photographed against an empty field. A
			boot, a bag, a shirt — each a silhouette a reader already knows how to judge. That is the
			whole point of the diet. The next chapter will ask the model for one of those ten by name;
			here the pictures are just the thing being ruined and restored.
		</p>
		<p>
			So press Train and watch. The first thing to appear is not a sleeve but a silhouette — the
			model learns that a garment is a compact blob of ink in the middle of an empty square long
			before the blob has a neckline. Then the outline arrives. Then, some thousands of steps later,
			a boot grows a shaft and a shirt grows sleeves.
		</p>
		<p>
			Settle one question before any pictures arrive, because it decides whether the rest of this is
			interesting: will the model be drawing, or remembering? Two and a half million parameters
			against 12,000 pictures is not obviously room enough to memorize, and not obviously too little
			either. The check is cheap. Take a sample and search the whole corpus for the picture it most
			resembles. Do that with the trained weights and what comes back is a cousin, not a copy — a
			different boot, a different bag. It is inventing. Crudely, at this size, but genuinely.
		</p>
	</Prose>

	<Learn
		title="From static, here"
		caption="The top row is eight pictures drawn from eight fixed seeds by weights that started as noise when you pressed Train; the bottom row is the same eight seeds drawn from the checkpoint that came down with this page. Nothing is cached and nothing is replayed — the top row is your GPU. Reset re-rolls the weights and starts the top row over. The gap between the rows is not a difference of architecture, data or objective; it is only steps, and the top row will close a visible fraction of it in the time it takes to read the next page."
	/>

	<UnderTheHood slug="noise" block="denoise" />

	<Prose>
		<h2 class="h2">Walking back</h2>
		<p>
			Training never simulated a chain, but sampling has to. Start from pure static — an image of
			nothing, drawn from the same Gaussian the corruption used — and repeat: ask the model which
			noise is present, use its answer to work out what clean picture is implied, and then step back
			to a slightly lower noise level. Each step is small enough that the model's guess is reliable,
			and the sequence of small reliable guesses is the picture.
		</p>
		<p>
			How you take that step is a choice made after training, not before it. The original recipe
			puts a little fresh noise back at every rung, which makes sampling a random walk: the same
			starting static gives a different picture each time.<Cite id="ho-2020" /> A later one drops the
			re-injection and follows a deterministic path instead. The two land on different pictures from the
			same seed, but they agree about the noise levels training was scored against — so one set of weights
			serves both, and the deterministic one is allowed to skip: fifty steps where the original wanted
			a thousand.<Cite id="song-2021-ddim" /> The slider below is the dial between them. Both are discretizations
			of the same family of differential equations, a view that arrived a little later and tidied the
			whole subject up.<Cite id="song-2021-sde" />
		</p>
		<p>
			The plate lays a single sample's whole journey out along a row. Take the step count down and
			watch where it breaks.
		</p>
	</Prose>

	<Walk
		title="One picture's whole journey"
		caption="Eight moments from a single walk out of static, left to right, with the noise level printed underneath each. Turn η up and fresh noise is stirred back in at every rung, which is the sampler diffusion was published with; at η = 0 the walk is deterministic and the seed names the picture exactly. Then cut the step count. At fifty the row still lands somewhere; at ten it arrives smeared, and at five it does not arrive at all — the walk has been asked to take strides longer than the curve it is following stays straight for."
	/>

	<UnderTheHood slug="noise" block="walk" />

	<Prose>
		<p>
			That last failure is the scheme's honest cost, and its cause is specific. The model was
			trained to answer at one noise level at a time, and its answer is correct only in a small
			neighbourhood — it is the direction to move, not the destination. Stepping along a direction
			is only accurate while the direction holds, and under this schedule it does not hold for long.
			So sampling needs many steps, each one a full forward pass through the network, and once the
			next chapter starts steering, two or three passes per step. Recognizing a picture costs one.
			That ratio is why diffusion was an academic curiosity in 2015 and a product in 2022 — though
			the wait was not only for hardware. The shape of the idea survived from the first paper
			unchanged, but the parameterization, the schedule, the architecture and the steering trick
			were all found in between, and only then did a hundred passes a picture become something you
			could sell.<Cite id="dhariwal-nichol-2021" />
		</p>
		<p>
			There are two ways out. One is to make each pass cheaper, and the industry's answer is to run
			the whole process on the compressed code an autoencoder produces rather than on the pixels
			themselves, which buys back an order of magnitude before the denoiser has done anything at
			all.<Cite id="rombach-2022" /> The other is to need fewer passes, and that is a question about the
			shape of the path rather than the speed of the network. Look again at
			<PlateRef id="walk" lower /> and ask what would have to be true for five steps to be enough. The
			walk would have to be nearly straight.
		</p>
		<p>
			Nothing about the way this model was trained encouraged that. The path it follows was handed
			to it by a schedule borrowed from physics, and physics had no reason to make it straight. The
			next chapter changes the training so that it is — and then, having a path worth steering,
			finally tells the model what to draw.
		</p>
	</Prose>
</ChapterShell>
