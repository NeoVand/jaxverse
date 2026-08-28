<script lang="ts">
	import { resolve } from '$app/paths';
	import ChapterRef from '$lib/components/ui/ChapterRef.svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import PlateRef from '$lib/components/ui/PlateRef.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import Comparator from '$lib/components/demos/taste/Comparator.svelte';
	import Fence from '$lib/components/demos/taste/Fence.svelte';
	import Goodhart from '$lib/components/demos/taste/Goodhart.svelte';
	import JudgePlate from '$lib/components/demos/taste/JudgePlate.svelte';
	import Leash from '$lib/components/demos/taste/Leash.svelte';
	import PreferenceLoop from '$lib/components/demos/taste/PreferenceLoop.svelte';
</script>

<ChapterShell slug="taste">
	<Prose>
		<p>
			The learners of <ChapterRef slug="reward" lower /> were paid by a number the world handed them:
			a harbour worth twelve, a shoal worth minus nine, a pole still upright. Nobody had to be asked.
			The reward was a fact about the situation, and the only interesting question was how to divide the
			credit for it.
		</p>
		<p>
			Now try to write that number for the things people actually want from a machine. Score this
			answer for helpfulness. Score this paragraph for tact. Score this ornament for beauty. Each of
			those is a goal you can recognize in a heartbeat and cannot express as arithmetic, and no
			verifier anywhere will settle it for you. This chapter is about what you do then. The answer
			is the most consequential idea in how any assistant you have used was built, and it arrives
			with a failure mode attached that you are going to inflict on yourself, on purpose, about six
			minutes from now.
		</p>
	</Prose>

	<PreferenceLoop />

	<Prose>
		<p>
			Start with the part everyone gets wrong. The obvious move is to ask people for scores — rate
			this out of ten — and the obvious move is a disaster. People are miserable absolute raters.
			Their scale drifts within a single sitting, it drifts further between one person and the next,
			and a seven from a generous morning means nothing against a seven from a tired afternoon.
		</p>
		<p>
			But the same person, shown two things at once, is superb. You cannot put a number on a cup of
			coffee. You know instantly which of two you would rather drink. So do not collect ratings.
			Collect <em>comparisons</em>: a prompt, a winner, a loser, and no numbers anywhere.
		</p>
		<p>
			Which needs something to have taste about, so here is a space of printer's rosettes — the sort
			of engraved device a book like this one would strike on a title page. Each is drawn from six
			numbers: how many arms, how far they curl, how far they reach, how heavy the stroke, how full
			the inner bloom, and what ink. Nobody has told the page which regions of that space are any
			good, and nothing on this page knows until you say so.
		</p>
	</Prose>

	<Comparator />

	<Prose>
		<h2 class="h2">From verdicts to a number</h2>
		<p>
			Now turn that pile of comparisons into a score, which sounds like alchemy and is in fact
			ordinary statistics, and older than the name it travels under. Zermelo wrote it down in 1929
			to rank chess players from a tournament table<Cite id="zermelo-1929" />; Bradley and Terry
			rediscovered it in 1952 and got their names on it. Posit that every ornament has a hidden
			worth
			<Math tex={'\\htmlClass{eq-model}{r_\\phi}(y)'} />, and that a person prefers stochastically
			according to the <em>gap</em> between two worths, squashed into a probability:
		</p>
		<Math
			display
			tex={'P(\\htmlClass{eq-world}{y_w} \\succ \\htmlClass{eq-world}{y_l}) \\;=\\; \\htmlClass{eq-op}{\\sigma}\\big(\\htmlClass{eq-model}{r_\\phi}(\\htmlClass{eq-world}{y_w}) - \\htmlClass{eq-model}{r_\\phi}(\\htmlClass{eq-world}{y_l})\\big)'}
		/>
		<p>
			That sideways symbol just means <em>was preferred to</em>, and
			<Math tex={'\\htmlClass{eq-op}{\\sigma}'} /> is the same S-curve that has been squashing scores
			into probabilities since <ChapterRef slug="neuron" lower />. Feel the shape with numbers: a
			gap of zero gives 0.50, a coin flip; a gap of one gives about 0.73; a gap of three gives 0.95.
			So the thing being fitted is calibrated in <em>how reliably a person would pick this one</em>,
			which is a claim you can check.
		</p>
		<p>
			Then fit it the only way anything is ever fitted here. For each pair, ask how surprised the
			model was that you picked the winner; average that surprise; walk downhill.
		</p>
		<Math
			display
			tex={'\\mathcal{L}(\\htmlClass{eq-model}{\\phi}) \\;=\\; -\\,\\mathbb{E}\\Big[\\, \\log \\htmlClass{eq-op}{\\sigma}\\big(\\htmlClass{eq-model}{r_\\phi}(\\htmlClass{eq-world}{y_w}) - \\htmlClass{eq-model}{r_\\phi}(\\htmlClass{eq-world}{y_l})\\big) \\,\\Big]'}
		/>
		<p>
			Look at what that is. A model that takes two things, scores each, subtracts, and squashes the
			difference to predict a winner. That is Elo. That is a credit score. That is every
			probability-this-deal-closes spreadsheet ever built. The exotic-sounding <em>reward model</em>
			at the heart of modern alignment is the oldest, most boring shape in statistics, pointed at pairs
			of essays instead of pairs of chess players — and the mystique evaporates the moment you can see
			the two forward passes and the subtraction. Bolting a neural network onto that shape and optimising
			against the result is far more recent than the shape itself: it was first done in 2017, on Atari
			games and simulated robots, where about a thousand human comparisons bought behaviours nobody could
			have written a reward function for.<Cite id="christiano-2017" />
		</p>
		<p>
			One number below is worth more than the rest. It is easy to score a model on data it was
			trained on and learn nothing. So before each of your clicks, the judge — knowing only the
			clicks that came before it — was asked to call the pair in advance. That running score is the
			only honest thing on the plate.
		</p>
	</Prose>

	<JudgePlate />

	<UnderTheHood slug="taste" block="judge" />

	<Prose>
		<h2 class="h2">The proxy and the goal</h2>
		<p>
			You now have a number that stands in for your taste, and a number is a thing an optimizer can
			climb. This is the moment the whole field turns, so slow down for it.
		</p>
		<p>
			Your judge is not your taste. It is four hundred parameters fitted to a few dozen of your
			clicks, and away from those clicks it is guessing. Where you never voted, it extrapolates, and
			extrapolation from a small sample is not opinion but arithmetic accident. There are regions of
			ornament space where the judge is confidently, splendidly wrong, and neither of you has any
			idea where they are.
		</p>
		<p>
			Nor is the industrial version any purer. When this is done for real, the people supplying the
			comparisons agree with each other only about three times in four,<Cite id="stiennon-2020" /> so
			the judge is fitted to a noisy majority and inherits that noise as a ceiling — one no amount of
			extra data will lift.
		</p>
		<p>
			An optimizer will find them. Not out of malice — it has no model of your intentions and no
			interest in them. It has a function and a gradient, and the fastest route uphill runs straight
			through whichever hole is nearest. <strong>Goodhart's law</strong>: when a measure becomes a
			target, it ceases to be a good measure. The Soviet nail factory judged by the weight of nails
			produced a few enormous useless ones; judged by count, a million tiny useless ones. Neither
			factory was malfunctioning. Both were optimizing exactly what they were told. (That factory is
			a cartoon from a Soviet satirical magazine rather than a documented plant, which does not make
			it less exact.)
		</p>
		<p>
			The plate below turns an optimizer loose on your judge with no restraint of any kind, and
			photographs the policy as it travels — the distance measured in <em>nats</em> of KL from where
			it started, which is the field's honest ruler for <em>how hard did we optimize</em>. Two
			curves get drawn on the same axis. One is your judge's opinion, which costs nothing and can be
			computed a million times a second. The other is yours, and every point of it costs you a
			click. Watch where they part company, and notice that nothing on the first curve marks the
			spot.
		</p>
	</Prose>

	<Goodhart />

	<UnderTheHood slug="taste" block="goodhart" />

	<Prose>
		<p>
			This is not a quirk of a four-hundred-parameter judge on a toy space. Run the same experiment
			with real reward models and real language models and you get the same two curves, in the same
			arrangement, reliably enough to fit an equation to: the proxy climbs steadily with distance
			travelled while the true score rises, peaks and turns over — and where it turns over moves
			predictably with how large the reward model is and how much data it was fitted to.<Cite
				id="gao-2023"
			/> The failure has a shape, and the shape has parameters. That is a good deal better than knowing
			it can happen, and still nowhere near being able to see the peak from the inside.
		</p>
		<h2 class="h2">The leash</h2>
		<p>
			So the optimizer must be stopped, and the question is what to charge it for. Not distance in
			the parameters — parameters lie about behaviour, and always have. Charge it for how far its
			<em>behaviour</em> has drifted from something already trusted. Distance between two
			distributions has a standard meter, and here at last is the debt
			<ChapterRef slug="language" lower /> left unpaid:
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-op}{\\mathrm{KL}}\\big(\\htmlClass{eq-model}{\\pi_\\theta} \\,\\|\\, \\htmlClass{eq-mute}{\\pi_{\\text{ref}}}\\big) \\;=\\; \\mathbb{E}_{y \\sim \\htmlClass{eq-model}{\\pi_\\theta}}\\left[\\, \\log \\frac{\\htmlClass{eq-model}{\\pi_\\theta}(y)}{\\htmlClass{eq-mute}{\\pi_{\\text{ref}}}(y)} \\,\\right]'}
		/>
		<p>
			Read the subscript before anything else, because the subscript is the safety property. The
			average is taken over draws from <em>your own</em> policy, so the penalty only ever notices places
			you actually go — and it bites hardest exactly where you are putting mass the reference never would,
			which is the definition of the blot. Written the other way round it would patrol territory you never
			visit and shrug at the exploit. KL is not symmetric, and here the asymmetry is not a wart to apologize
			for. It is the entire mechanism.
		</p>
		<p>
			Put the two halves together and you have the single most important objective in modern AI:
		</p>
		<Math
			display
			tex={'\\max_{\\htmlClass{eq-model}{\\theta}}\\; \\mathbb{E}_{y \\sim \\htmlClass{eq-model}{\\pi_\\theta}}\\big[\\, \\htmlClass{eq-model}{r_\\phi}(y) \\,\\big] \\;-\\; \\htmlClass{eq-knob}{\\beta}\\, \\htmlClass{eq-op}{\\mathrm{KL}}\\big(\\htmlClass{eq-model}{\\pi_\\theta} \\,\\|\\, \\htmlClass{eq-mute}{\\pi_{\\text{ref}}}\\big)'}
		/>
		<p>
			This is the objective the assistant you have talked to was actually trained on<Cite
				id="ouyang-2022"
			/>. In English: score as high as you can <em>while remaining recognizable</em>.
			<Math tex={'\\htmlClass{eq-knob}{\\beta}'} /> is the length of the leash — the exchange rate between
			reward points and strangeness. Push it to infinity and the policy never moves. Push it to zero and
			you get the previous plate. Everything worth having is in between.
		</p>
		<p>
			And this objective does not need a search. It has an exact answer, which you can write down in
			one line and which is the same shape you will meet every time a paper says <em>soft</em>,
			<em>temperature</em>, or <em>log-sum-exp</em>:
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-model}{\\pi^{*}}(y) \\;\\propto\\; \\htmlClass{eq-mute}{\\pi_{\\text{ref}}}(y)\\, \\exp\\!\\left(\\frac{\\htmlClass{eq-model}{r_\\phi}(y)}{\\htmlClass{eq-knob}{\\beta}}\\right)'}
		/>
		<p>
			Take the distribution you already trust and <em>tilt it exponentially by reward</em>. Boost,
			don't replace. Feel why it has to be this before checking that it is: you want mass where
			reward is high, and you are charged for leaving home, so the answer must be home multiplied by
			something that grows with reward. The plate draws all three curves — the reference, the tilt,
			and their product — along whichever gene you like, and hands you β.
		</p>
	</Prose>

	<Leash />

	<UnderTheHood slug="taste" block="leash" />

	<Prose>
		<h2 class="h2">One more fence, for a different fear</h2>
		<p>
			The leash prices drift. There is a second instrument for a problem that sounds similar and is
			not, and it belongs to the fact <ChapterRef slug="reward" lower /> kept circling: a policy generates
			its own training data. In ordinary supervised learning a step that is too large is an inconvenience
			— the dataset sits there, patient and fixed, and the next few steps recover. Here, a destructively
			large step collapses the policy into some degenerate habit, and every sample it collects from then
			on is garbage, so the gradients computed <em>from</em> that garbage cannot guide you back. You did
			not stumble. You sawed off the branch you were standing on.
		</p>
		<p>
			So the question becomes: what is the biggest step that is still safe? PPO's answer<Cite
				id="schulman-2017"
			/> is disarmingly cheap. Weight each action by the ratio between what the new policy thinks of it
			and what the policy that collected the data thought — <Math
				tex={'\\rho = \\htmlClass{eq-model}{\\pi_\\theta} / \\htmlClass{eq-mute}{\\pi_{\\theta_{\\text{old}}}}'}
			/> — and then compute both the honest objective and a fenced one, and always take whichever is worse
			for you.
		</p>
		<Math
			display
			tex={'L^{\\text{CLIP}} \\;=\\; \\mathbb{E}\\Big[\\, \\min\\big(\\, \\rho\\,\\hat{A},\\; \\htmlClass{eq-op}{\\operatorname{clip}}(\\rho,\\, 1-\\htmlClass{eq-knob}{\\varepsilon},\\, 1+\\htmlClass{eq-knob}{\\varepsilon})\\,\\hat{A} \\,\\big) \\,\\Big]'}
		/>
		<p>Don't memorize it — walk it. The design is the insight, and it is asymmetric on purpose.</p>
	</Prose>

	<Fence />

	<Prose>
		<h2 class="h2">The heist</h2>
		<p>
			One last thing, because it is the prettiest result in this part of the field and it needs no
			plate — only the two equations already on this page, read in the other direction.
		</p>
		<p>
			Look again at the exact solution, <Math
				tex={'\\htmlClass{eq-model}{\\pi^{*}} \\propto \\htmlClass{eq-mute}{\\pi_{\\text{ref}}} \\exp(\\htmlClass{eq-model}{r}/\\htmlClass{eq-knob}{\\beta})'}
			/>, and instead of solving for the policy, solve for the reward:
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-model}{r}(y) \\;=\\; \\htmlClass{eq-knob}{\\beta} \\log \\frac{\\htmlClass{eq-model}{\\pi_\\theta}(y)}{\\htmlClass{eq-mute}{\\pi_{\\text{ref}}}(y)} \\;+\\; \\htmlClass{eq-knob}{\\beta} \\log Z'}
		/>
		<p>
			Sit with what that says. Any policy that is optimal for <em>some</em> leashed objective is
			carrying its own reward function around inside it, and that reward is nothing but the
			log-ratio of the policy to the reference. Ranking things by
			<em>how much more likely fine-tuning made this than the reference would have</em> is ranking them
			by reward. Your language model is secretly a reward model.
		</p>
		<p>
			That normalizer <Math tex="Z" /> is a sum over every string a keyboard could produce, which is less
			a large number than a category error, and for years it made the formula a pretty footnote. Then
			somebody noticed where the reward was going to be <em>used</em>: inside
			<Math
				tex={'\\htmlClass{eq-op}{\\sigma}(\\htmlClass{eq-model}{r}(\\htmlClass{eq-world}{y_w}) - \\htmlClass{eq-model}{r}(\\htmlClass{eq-world}{y_l}))'}
			/>, which touches rewards only through a <em>difference</em> between two responses to the same
			prompt. Same prompt, same <Math tex="Z" />, and the uncomputable term annihilates itself. What
			is left is one supervised loss on preference pairs that provably targets the same optimum — no
			reward model to train, no sampling, no rollouts, no critic. That is <strong>DPO</strong><Cite
				id="rafailov-2023"
			/>, and the whole four-model circus collapses into ordinary fine-tuning.
		</p>
		<p>
			With one bill, which this book will not hide. DPO never generates anything, so it can never
			discover a response better than the ones already in its dataset; all it can do is re-weight
			what is already there. To get reliably better than your data, the model has to produce, be
			judged, and update. That gap has been measured, and it falls where you would expect: widest
			when the good responses are rare in the dataset, which is the case anyone actually cares
			about.<Cite id="tajwar-2024" /> There is no way around the loop.
		</p>

		<h2 class="h2">What you just did to yourself</h2>
		<p>
			Step back and count the moves, because every one of them has an industrial twin. You gave
			comparisons, not scores, because comparisons are the only judgment people give reliably. A
			Bradley–Terry head turned that pile into a number. An optimizer aimed at the number found the
			places the number was wrong, and your own eye — expensive, slow, and the only thing that was
			ever the goal — peaked and turned over while the dashboard kept climbing. A KL leash back to
			something trusted was what stood between the two. Swap ornaments for essays and four hundred
			parameters for a trillion, and that is RLHF, unchanged in every particular.
		</p>
		<p>
			It also tells you exactly where this pipeline is weakest, and therefore where the next chapter
			goes. Everything here rests on a judge that had to be <em>learned</em>, from a small and
			expensive pile of human clicks, and could therefore be wrong in ways nobody can see. Suppose
			instead the judge were a program that cannot be flattered, cannot be bribed, and is never
			wrong — a referee that simply knows. Then there is no proxy to game, no annotator's taste in a
			box, and no peak to fall off.
		</p>
		<p>
			That judge exists for some questions and not others, and which side a question falls on does
			more than anything else to decide whether machines are getting rapidly better at it.
			<a href={resolve('/rook')}>Meet Rook</a> — and watch <PlateRef id="fence" lower /> turn out to be
			machinery it barely needs.
		</p>
	</Prose>
</ChapterShell>
