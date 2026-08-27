<script lang="ts">
	import { resolve } from '$app/paths';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import OneNeuron from '$lib/components/demos/neuron/OneNeuron.svelte';
	import ActivationAtlas from '$lib/components/demos/neuron/ActivationAtlas.svelte';
	import CurveFit from '$lib/components/demos/neuron/CurveFit.svelte';
</script>

<ChapterShell slug="neuron">
	<Prose>
		<p>
			The prologue left you on a landscape. Every knob of a machine became an axis, wrongness became
			height, and learning became one small move repeated: measure the slope, step downhill. What it
			never said is what the machine is. This chapter builds one — the smallest one worth building —
			and then hands its knobs to gradient descent while you watch.
		</p>
		<p>
			The working part is small enough to hold in your head. A <em>neuron</em> takes a number in,
			multiplies it by a <em>weight</em>
			<Math tex={'\\htmlClass{eq-model}{w}'} />, adds a
			<em>bias</em>
			<Math tex={'\\htmlClass{eq-model-2}{b}'} />, and passes the result through a fixed curve
			called an
			<em>activation</em>. This book starts with the gentlest activation, the hyperbolic tangent:
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-out}{a} = \\htmlClass{eq-op}{\\sigma}(\\htmlClass{eq-model}{w}x + \\htmlClass{eq-model-2}{b}), \\qquad \\htmlClass{eq-op}{\\sigma}(z) = \\htmlClass{eq-op}{\\tanh}(z)'}
		/>
		<p>
			That is the whole organism. In one dimension its output is a smooth step: flat, a rise, flat
			again. The weight sets how steep the rise is and which way it faces; the bias slides it along
			the input. One step is a poor vocabulary — so we hire more. A <em>layer</em> is a row of
			neurons reading the same x, each with its own weight and bias: a palette of steps, tilted and
			placed differently. An <em>output layer</em> mixes the palette, scaling each step by an output weight
			and adding them all up:
		</p>
		<Math
			display
			tex={'f(\\htmlClass{eq-world}{x}) = \\sum_i \\htmlClass{eq-model-3}{v_i}\\,\\htmlClass{eq-op}{\\tanh}\\!\\left(\\htmlClass{eq-model}{w_i}\\,x + \\htmlClass{eq-model-2}{b_i}\\right) + c'}
		/>
		<p>
			Subtract one step from a slightly shifted copy and you get a bump. Sums of steps are therefore
			sums of bumps: little hills of influence you can place, widen, and flip. The
			<em>universal approximation theorem</em> makes this precise — with enough hidden neurons, f
			can match any continuous curve on an interval as closely as you like.<Cite
				id="cybenko-1989"
			/>
			It holds from the other direction too: keep the layer narrow, barely wider than the input, and stack
			it deep enough, and you get the same guarantee.<Cite id="lu-2017" />
		</p>
		<p>
			Read either of them honestly, though. They say the right weights <em>exist</em>; they say
			nothing about how to find them, and “enough” can be an absurd number. And a theorem about what
			a machine <em>could</em> represent is a weaker thing than it sounds — a large enough lookup table
			is a universal approximator too, and no use to anyone. The question that decides whether a network
			is worth building is not which curves it could draw in principle. It is which ones it finds easily,
			from a random start, by walking downhill. That is the previous chapter's business: write the error
			down as a loss, and send gradient descent looking.
		</p>

		<h2 class="h2">One neuron, three knobs</h2>
		<p>
			Before training anything, get your hands on the unit itself. Below is the same neuron twice:
			on the left as a circuit — the ultramarine edge carries <Math
				tex={'\\htmlClass{eq-model}{w}'}
			/>, the violet edge carries <Math tex={'\\htmlClass{eq-model-2}{b}'} />, and the blue-cyan
			edge on the way out carries the amplitude <Math tex={'\\htmlClass{eq-model-3}{v}'} />, with
			the teal disk between them plotting whichever <Math tex={'\\htmlClass{eq-op}{\\sigma}'} /> is selected
			— and on the right as
			<Math
				tex={'\\htmlClass{eq-model-3}{v}\\,\\htmlClass{eq-op}{\\sigma}(\\htmlClass{eq-model}{w}x + \\htmlClass{eq-model-2}{b})'}
			/>, the one shape it can draw. The third knob, <Math tex={'\\htmlClass{eq-model-3}{v}'} />, is
			just another coefficient — an amplitude that stretches the bend taller or flips it upside
			down. Move each slider until you can predict what both views will do before you touch it. Then
			swap <Math tex={'\\htmlClass{eq-op}{\\sigma}'} /> itself and see which of your intuitions survive:
			the same three knobs pull very different shapes out of a relu than out of a tanh.
		</p>
	</Prose>

	<OneNeuron />

	<UnderTheHood slug="neuron" block="bare" />

	<Prose>
		<h2 class="h2">The bend is the whole point</h2>
		<p>
			Strip <Math tex={'\\htmlClass{eq-op}{\\sigma}'} /> out and a neuron is just <Math
				tex={'\\htmlClass{eq-model}{w}x + \\htmlClass{eq-model-2}{b}'}
			/> — a line. Stack a hundred all-linear layers and the stack collapses: a line of a line is still
			a line, so the deepest such network can be multiplied out into a single matrix wearing a hundred
			costumes. The activation is the only part of the machine that refuses to be linear, and everything
			a network can do that a line cannot — every curve, corner, and decision — is purchased at that little
			bend.
		</p>
		<p>
			Training cares about a second, quieter property: the slope. Gradient descent reaches every
			weight through the chain rule, and the chain rule multiplies by <Math
				tex={"\\htmlClass{eq-op}{\\sigma'}"}
			/> at each layer it crosses on the way back. The field guide below therefore draws every activation
			twice — the function solid, its derivative dashed. Wherever the dashed curve hugs zero, learning
			goes quiet. The two classics saturate at both ends, which is how deep sigmoid networks starved for
			decades — the <em>vanishing gradient</em>, and a large part of why the field spent those
			decades shallow.<Cite id="glorot-bengio-2010" />
		</p>
		<p>
			relu<Cite id="nair-hinton-2010" /> is silent across its entire left half instead, which sounds worse
			and is not: on the right it never saturates at all, so a gradient can cross ten layers without being
			multiplied down to nothing. The cost is that a unit driven far enough left stops receiving gradient
			and is called
			<em>dead</em>; its leaky cousin<Cite id="maas-2013" /> keeps a trickle flowing on purpose. That
			half of relu's units are silent at any moment turns out to be part of the appeal rather than a defect
			— the network computes on a changing, sparse subset of itself.<Cite id="glorot-2011" />
		</p>
		<p>
			The last three — gelu<Cite id="hendrycks-gimpel-2016" />, silu<Cite id="elfwing-2017" />, mish<Cite
				id="misra-2019"
			/> — sands the corner off and keeps a little slope slightly below zero, which is one reason the
			transformer era settled on them. Silu arrived twice: once from researchers building reinforcement
			learners, and again, years later, out of an automated search over candidate formulas that reported
			it as a discovery under a different name.<Cite id="ramachandran-2017" /> That is roughly how this
			corner of the field works. Nobody derives an activation from first principles; they are found, measured,
			and kept if they win.
		</p>
		<p>
			None of this changes what networks <em>can</em> express — the universal approximation theorem is
			not picky about the bend. What it changes is the handwriting of the fit and the temperament of training,
			and you will see both first-hand in the workshop that follows.
		</p>
	</Prose>

	<ActivationAtlas />

	<Prose>
		<h2 class="h2">The curve workshop</h2>
		<p>
			Now give the knobs away. The workshop below holds a whole network and trains it live, in a
			worker beside this page, by the only rule this book uses —
			<Math
				tex={'\\htmlClass{eq-model}{\\theta} \\leftarrow \\htmlClass{eq-model}{\\theta} - \\htmlClass{eq-knob}{\\eta} \\htmlClass{eq-world}{\\nabla \\mathcal{L}}'}
			/> — with the loss set to the <em>mean-squared error</em>, the average of <Math
				tex={'(f(\\htmlClass{eq-world}{x}) - \\htmlClass{eq-world}{y})^2'}
			/> over the training points. Every step pulls the network’s curve a little closer to a target.
		</p>
		<p>
			You see the machine three ways at once. On the left sits the network itself: every edge is one
			weight, drawn thicker as it grows, ultramarine when positive and vermilion when negative. On
			the right, the curve it currently draws against the dashed target. And underneath, the palette
			— the curves the deeper layers output, ending in the last hidden layer’s contributions, each
			drawn in the color of its output weight’s sign and thicker as that weight grows. Hover any
			neuron in the diagram to light up its wiring — and, if it owns a tile, the curve it outputs;
			or hover a tile to find its neuron.
		</p>
		<p>
			Train on the sine and watch the palette organize — units that began as arbitrary steps drift
			into position, each claiming a stretch of the curve. Switch the target to bumps mid-training
			and the units renegotiate. Then take the pen and draw your own curve straight onto the plot:
			the network will chase it while you drag.
		</p>
	</Prose>

	<CurveFit />

	<UnderTheHood slug="neuron" block="engine" />

	<Prose>
		<h2 class="h2">What width buys, what depth buys</h2>
		<p>
			Width buys vocabulary. Every extra hidden unit is one more step the output layer can place, so
			a wider palette affords finer detail: at width 2 the sine defeats the network — two steps
			cannot make three bends — while at width 16 it has bumps to spare. Watch the parameter count
			as you widen, though. Vocabulary is not free.
		</p>
		<p>
			Depth buys reuse. With one hidden layer, every wiggle must be purchased with its own neurons.
			Add a second layer and the new neurons stop reading x directly: they read the first layer’s
			steps, and can bend an already-bent thing — bumps of bumps. A third layer bends the bumps of
			bumps again. The same budget of parameters goes further because pieces are reused instead of
			re-made.
		</p>
		<p>
			That is not just a nicer story; it is a measurable gap. Count the straight pieces a relu
			network's fit is made of and the count grows roughly in proportion to width, but
			<em>multiplies</em> with each layer of depth — every layer folds the folds beneath it.<Cite
				id="montufar-2014"
			/> And there are functions a deep network draws with a handful of units that no shallow network
			can match without an exponential number of them.<Cite id="telgarsky-2016" /> In one dimension the
			difference is subtle; in the chapters ahead, where inputs are images, it is most of the story.
		</p>
		<p>
			The activation sets the network’s handwriting. tanh is a soft wave, so its sums are smooth and
			rounded everywhere. The <em>rectified linear unit</em> — relu, zero on the left, a straight ramp
			on the right — creases instead of curves: its sums are piecewise-linear, and you can count the folds
			in the fit. Switch the workshop to relu on the |x| target and it lands almost at once, because the
			target is itself two creases. Switch back to tanh and watch it round a corner it can never make
			sharp. The workshop also carries the modern pair from the field guide — gelu and silu — which crease
			like relu with the corners sanded smooth; try them and read the difference straight off the palette.
		</p>
		<p>
			One honest warning before you scale everything up. The workshop samples its target at 256
			points packed densely along the interval, so fitting the data and fitting the curve are nearly
			the same task. They usually aren’t. With few points and many neurons, a network can pass
			through every training point exactly and still be wrong everywhere between them — wiggling
			where it should glide. That failure is called <em>overfitting</em>, and nothing in the loss
			ever asked the network to behave between the points. This book meets it properly once real
			data arrives.
		</p>
		<p>
			The old lesson drawn from that is: capacity is dangerous, so keep the model small. The last
			decade made a mess of the lesson. Take a network that reads photographs well, shuffle the
			labels so every answer is now noise, and train it again — it fits all of them, perfectly,
			memorising the lot.<Cite id="zhang-2017" /> Its capacity to overfit is total. Hand back the real
			labels and the very same network generalizes. Whatever stops it from doing the same to those, it
			is not a shortage of room.
		</p>
		<p>
			Stranger still is what happens if you keep growing it. Test error rises as the textbook
			promises, peaks right at the size where the model can just barely fit its training set exactly
			— and then, as the model gets bigger still, comes down again, often below anything the small
			models managed.<Cite id="belkin-2019" /> The curve has two descents, and the classical one is only
			the first. The models in the news live far out on the second, in the region the textbook picture
			calls hopeless, and the same double dip appears in how long you train, not only in how big you build.<Cite
				id="nakkiran-2019"
			/> None of this repeals overfitting. It moves the explanation somewhere more interesting: what keeps
			an over-parameterised network honest is not how few parameters it has, but which of the enormous
			number of perfect fits gradient descent happens to walk to.
		</p>
		<p>
			This chapter happened in one dimension on purpose: you could see every neuron, every bump, and
			the whole sum at once. Nothing was hidden — the palette under the plot <em>is</em> the network,
			laid out flat.
		</p>
		<p>
			The next chapter keeps the machine and changes the canvas. In two dimensions a network stops
			looking like a sum of bumps and starts doing something stranger and better: it bends the space
			the data lives in, until problems that looked hopeless become straight lines. When you’re
			ready, <a href={resolve('/space')}>go bend space</a>.
		</p>
	</Prose>
</ChapterShell>
