<script lang="ts">
	import { resolve } from '$app/paths';
	import OptimizerRace from '$lib/components/demos/descent/OptimizerRace.svelte';
	import StepSize from '$lib/components/demos/descent/StepSize.svelte';
	import ChapterRef from '$lib/components/ui/ChapterRef.svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import { progress } from '$lib/data/progress.svelte';
</script>

<ChapterShell slug="descent">
	<Prose>
		<p>
			Strip any learning machine down — the ones that read handwriting, the ones that talk — and you
			find a function with knobs. The knobs are numbers called <em>parameters</em>, written
			<Math tex={'\\htmlClass{eq-model}{\\theta}'} />, and turning them changes what the function
			computes. Feed it an example, compare what it produced with what you wanted, and you can score
			the disagreement with a single number. That number is the <em>loss</em>, written
			<Math tex={'\\mathcal{L}(\\htmlClass{eq-model}{\\theta})'} /> — the machine's whole report card,
			one value that says how wrong its current knobs are.
		</p>
		<p>
			Framed this way, learning stops being a mystery and becomes a landscape. Give every knob its
			own axis and add one more for the loss: each possible setting of the machine is a point on a
			surface, bad settings up on the ridges, good ones down in the valleys.
			<em>Learning is descending that surface.</em> One rule does the descending, and the rest of this
			book repeats it like a heartbeat:
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-model}{\\theta_{t+1}} \\;=\\; \\htmlClass{eq-model}{\\theta_t} \\;-\\; \\htmlClass{eq-knob}{\\eta}\\,\\htmlClass{eq-world}{\\nabla \\mathcal{L}(\\theta_t)}'}
		/>
		<p>
			Read it term by term. The <em>gradient</em>
			<Math tex={'\\htmlClass{eq-world}{\\nabla \\mathcal{L}(\\theta_t)}'} /> is a list with one entry
			per knob: nudge this knob up, it says, and the loss rises by this much. Taken together the entries
			point in the direction of steepest ascent — the quickest way to make things worse — which is exactly
			why the rule subtracts them. The <em>learning rate</em>
			<Math tex={'\\htmlClass{eq-knob}{\\eta}'} /> sets the stride: how far to trust each reading of the
			slope before stopping to feel the ground again. The direction is the world's to give; the stride
			is yours to choose; there is nothing else in the rule. (Other books write that stride
			<Math tex="\alpha" />, and some write <Math tex="\gamma" />. This one keeps
			<Math tex={'\\htmlClass{eq-knob}{\\eta}'} /> throughout, because
			<Math tex="\gamma" /> has a different job waiting for it in
			<ChapterRef slug="reward" />.)
		</p>
		<p>
			All of it rests on one requirement, and that requirement is the quiet thesis of this whole
			book. The walker is blind: no map, no view of the valley, only the tilt of the ground directly
			under its feet. If the landscape had cliffs and teleports — if a whisker of change in some
			knob could throw the loss anywhere at all — that local tilt would tell it nothing, and no
			amount of walking would help. So we build our machines entirely from smooth parts, arranged so
			the loss turns gently whenever a knob turns: the landscape is <em>differentiable</em>, and
			that is the price of admission. On smooth ground, a blind walker feeling only the slope
			beneath its feet can find its way down a landscape it will never see.
		</p>

		<h2 class="h2">Five walkers, one landscape</h2>
		<p>
			Below is a real loss surface, drawn the way a hiker would draw it: contour lines join settings
			of equal loss, and the shaded pools mark the basins. Five walkers run the update rule on it,
			each with its own temperament — plain <em>gradient descent</em>, the bare rule and nothing
			more; <em>momentum</em><Cite id="polyak-1964" />, which keeps a running average of its recent
			gradients; <em>Adam</em><Cite id="kingma-ba-2015" />, which gives every knob its own step
			size; and, off by default, <em>AdamW</em><Cite id="loshchilov-hutter-2019" /> and
			<em>Lion</em><Cite id="chen-2023" />. Click anywhere on the map to drop all of them there at
			once, and flip the view to 3-D to see the terrain the contours are describing. One dial drives
			all five: each walker's stride is pre-scaled to its temperament, because the raw-gradient pair
			barely move at a <Math tex={'\\htmlClass{eq-knob}{\\eta}'} /> the adaptive pair finds comfortable.
		</p>
	</Prose>

	<OptimizerRace onraced={() => progress.reach('descent:done')} />

	<UnderTheHood slug="descent" block="gradient" />

	<Prose>
		<h2 class="h2">Reading the race</h2>
		<p>
			Choose the ravine and the temperaments separate within seconds. Its walls are steep and its
			floor is nearly flat, so the gradient points almost entirely <em>across</em> the trench: plain descent
			zigzags wall to wall, each step mostly undoing the last, and only creeps toward the minimum. Momentum
			averages its recent steps, so the wall-to-wall part cancels itself out while the along-the-floor
			part quietly accumulates — it ends up surfing the very trench it was bouncing across.
		</p>
		<p>
			Adam keeps a second running average — the typical <em>size</em> of the gradient, tracked separately
			for each knob — and divides by it. Steep directions get reined in, flat ones get amplified, and
			every axis ends up moving at a steady walking pace; that is why Adam strides along the ravine's
			flat floor where plain descent crawls.
		</p>
		<p>
			Neither half of Adam was new. AdaGrad<Cite id="duchi-2011" /> had already given every parameter
			its own step size by dividing through by the gradients it had seen so far — but it added those up
			forever, so the steps shrank toward nothing and training ground to a halt. RMSProp<Cite
				id="tieleman-hinton-2012"
			/> fixed that by making the average forgetful: recent gradients count, old ones fade. Adam is RMSProp
			with momentum bolted on and a small correction for the opening steps, when an average that started
			at zero has almost nothing to average yet.
		</p>
		<p>
			The opt-in pair are variations. AdamW adds <em>weight decay</em>, a constant gentle pull
			toward small parameters — watch it settle deliberately short of where Adam does. The name
			marks a finer point than it looks: adding a penalty on large weights to the loss and shrinking
			the weights directly are the same operation for plain descent, and are <em>not</em> the same
			operation once Adam divides by the gradient's typical size, which had quietly been weakening
			the penalty exactly where gradients were loud. Lion discards magnitude altogether and steps a
			fixed <Math tex={'\\htmlClass{eq-knob}{\\eta}'} /> along the sign of its momentum, which makes it
			quick off the mark and restless at the bottom, dancing in place forever.
		</p>
		<p>
			Now break it. Raise <Math tex={'\\htmlClass{eq-knob}{\\eta}'} /> and watch Adam begin to ring across
			the basin instead of settling into it; raise it further and plain descent and momentum are flung
			off the map entirely — the ledger under the field reads <em>diverged</em>, and it means
			precisely what it says: the step outran the valley it was measuring. Notice whose knob that
			was. Nobody learned <Math tex={'\\htmlClass{eq-knob}{\\eta}'} />; you chose it. It is a
			<em>hyperparameter</em> — a knob about the knobs — and it matters enough to deserve a landscape
			with nothing else in it.
		</p>

		<h2 class="h2">One dimension, three fates</h2>
		<p>
			On the parabola <Math
				tex={'\\mathcal{L}(\\htmlClass{eq-model}{\\theta}) = \\htmlClass{eq-model}{\\theta}^2'}
			/> the gradient is <Math tex={'\\htmlClass{eq-world}{2\\theta}'} />, and the update rule
			collapses into a single multiplication:
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-model}{\\theta_{t+1}} \\;=\\; \\htmlClass{eq-model}{\\theta_t} - \\htmlClass{eq-knob}{\\eta}\\cdot \\htmlClass{eq-world}{2\\theta_t} \\;=\\; \\bigl(1 - 2\\htmlClass{eq-knob}{\\eta}\\bigr)\\,\\htmlClass{eq-model}{\\theta_t}'}
		/>
		<p>
			Every step multiplies <Math tex={'\\htmlClass{eq-model}{\\theta}'} /> by the same factor, so the
			walker's fate hangs on whether
			<Math tex={'\\lvert 1 - 2\\htmlClass{eq-knob}{\\eta} \\rvert'} /> is smaller than one. Keep <Math
				tex={'\\htmlClass{eq-knob}{\\eta}'}
			/> below 0.5 and the factor is a gentle fraction: the ball slides down its own side of the bowl.
			At 0.5 the factor is zero — the bottom, in one step. Past 0.5 each step overshoots the minimum and
			lands on the far side, still lower than before; past 1.0 the factor outgrows one, and every bounce
			carries the ball higher than the last. Feel all three with the dial, then try the other two curves:
			a double well, where a bold <Math tex={'\\htmlClass{eq-knob}{\\eta}'} /> hops the ball between two
			valleys, and <Math tex={'\\lvert\\htmlClass{eq-model}{\\theta}\\rvert'} />, whose kink never
			lets the steps shrink — a first taste of why smoothness was the price of admission.
		</p>
		<p>
			That threshold is not a quirk of this parabola. Close enough up, every smooth valley looks
			like one, and the number that decides the fate of the step is how fast the slope itself
			changes — the <em>sharpness</em> of the tightest direction. Gradient descent holds together
			while the stride stays under two divided by that sharpness, and flies apart above it, exactly
			as the bowl did at <Math tex={'\\htmlClass{eq-knob}{\\eta} = 1'} />.
		</p>
		<p>
			Which makes what real networks do with that rule genuinely odd. Train one by gradient descent
			and it does not find a gentle valley and settle in. The sharpness along its path
			<em>climbs</em> — the ground it chooses keeps getting tighter — until it reaches almost
			exactly the largest value the chosen stride can survive. And then it stays there, balanced on
			the threshold for the rest of the run, the loss coming down in a slow sawtooth rather than a
			smooth glide.<Cite id="cohen-2021" /> The step size is not only a setting you tune to fit the landscape.
			It decides which landscape you end up on.
		</p>
	</Prose>

	<StepSize />

	<UnderTheHood slug="descent" block="jax" />

	<Prose>
		<h2 class="h2">The picture is honest</h2>
		<p>
			A fair objection: real models do not have two knobs. The first network you will train has a
			few hundred; the language model at the end of this book has hundreds of thousands; the models
			in the news have billions. Nobody can draw those landscapes. But nothing on this page was a
			cartoon of the algorithm — the rule you have been racing is, line for line, the rule they all
			train by, and in <a href={resolve('/neuron')}>the next chapter</a> you will watch this same subtraction
			sculpt a real neural network, live, as you read.
		</p>
		<p>Two things do change on the way up, and the usual account of each is worth correcting.</p>
		<p>
			The first is noise. Every gradient on this page was exact: the landscape is a formula, so the
			slope at a point can simply be computed. No real training run can afford that. The loss is an
			average over millions of examples, and evaluating all of them to take one step would be
			absurd, so the gradient is estimated from a few dozen examples at a time and every reading of
			the ground is slightly wrong. That is the <em>S</em> in SGD, and the licence for it is older
			than neural networks: Robbins and Monro showed in 1951<Cite id="robbins-monro-1951" /> that a walker
			stepping on noisy estimates still converges, provided its steps shrink neither too fast nor too
			slowly. The noise is not merely survivable. A rough gradient is a cheap gradient, so the same compute
			buys hundreds of times more steps, and the jitter shakes the walker off the first flat place it
			stumbles into.
		</p>
		<p>
			The second is the shape of the ground. The picture of a walker stranded in a shallow valley,
			unable to see the deeper one next door, is the most repeated image in popular accounts of this
			subject, and in high dimensions it is very nearly a myth. For a flat spot to be a floor, the
			surface has to curve upward along <em>every one</em> of the model's directions at once, and
			with a million directions to agree that is an extraordinary coincidence. Almost every flat
			spot is a saddle instead — uphill in most directions, downhill in a few — and a walker
			carrying any momentum or any noise at all eventually finds one of the few.<Cite
				id="dauphin-2014"
			/> What actually costs a training run its time is not being trapped at the bottom of the wrong bowl.
			It is crawling across ground that is nearly level in almost every direction at once.
		</p>
		<p>
			Some of this can be checked directly. Take a network's random starting weights and its trained
			ones, draw the straight line between them in parameter space, and read the loss along it: it
			falls the whole way, with no ridge to climb over.<Cite id="goodfellow-2015" /> Training does not
			walk that line — the route it takes is long and curved — but the fact that a straight one exists
			tells you the terrain is kinder than the cartoon. And when you do meet a rendered picture of a loss
			surface, ridges and basins and all, remember what it is: a two-dimensional slice through something
			with a million axes, whose apparent ruggedness depends on how the slice was scaled.<Cite
				id="li-2018"
			/>
		</p>
		<p>
			What survives once the picture is gone is the curve. In a million dimensions the landscape is
			invisible, but the walker's altitude is still a single number at every step, and plotting it
			against <Math tex="t" /> gives the training <em>loss curve</em> — the landscape's shadow. Every
			chapter ahead shows you that curve while its model trains. When you see it fall, stall on some plateau,
			then find a way down again, you will know what you are watching edge-on: a ball, a valley, a rule.
		</p>
		<p>
			That is the whole trick, and you have now held every part of it: a smooth function, a score
			for wrongness, and a walk downhill taken one step at a time. Everything from here on — telling
			digits apart, drawing a map of meaning, learning chess from applause — is this same walker,
			set loose on richer ground.
		</p>
	</Prose>
</ChapterShell>
