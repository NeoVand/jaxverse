<script lang="ts">
	import { resolve } from '$app/paths';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import Wide from '$lib/components/ui/Wide.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import OneNeuron from '$lib/components/demos/neuron/OneNeuron.svelte';
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
			multiplies it by a <em>weight</em> w, adds a <em>bias</em> b, and passes the result through a
			fixed curve called an <em>activation</em>. This book starts with the gentlest activation, the
			hyperbolic tangent:
		</p>
		<Math display tex="a = \sigma(wx + b), \qquad \sigma(z) = \tanh(z)" />
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
			tex={'f(x) = \\sum_i \\htmlClass{eq-a}{v_i}\\,\\tanh\\!\\left(\\htmlClass{eq-w}{w_i}\\,x + b_i\\right) + c'}
		/>
		<p>
			Subtract one step from a slightly shifted copy and you get a bump. Sums of steps are therefore
			sums of bumps: little hills of influence you can place, widen, and flip. The
			<em>universal approximation theorem</em> makes this precise — with enough hidden neurons, f
			can match any continuous curve on an interval as closely as you like. Read it honestly,
			though. The theorem says the right weights <em>exist</em>; it says nothing about how to find
			them, and “enough” can be an absurd number. What turns possible into practical is the previous
			chapter: write the error down as a loss, and send gradient descent looking.
		</p>

		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			One neuron, three knobs
		</h2>
		<p>
			Before training anything, get your hands on the unit itself. Below is the same neuron twice:
			on the left as a circuit — input, weighted sum, activation — and on the right as
			<Math tex={'\\htmlClass{eq-a}{v}\\,\\tanh(\\htmlClass{eq-w}{w}x + b)'} />, the one shape it
			can draw. Move each slider until you can predict what both views will do before you touch it.
		</p>
	</Prose>

	<Wide>
		<OneNeuron />
	</Wide>

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			The curve workshop
		</h2>
		<p>
			Now give the knobs away. The workshop below holds a whole network and trains it live, in a
			worker beside this page, by the only rule this book uses —
			<Math tex={'\\theta \\leftarrow \\theta - \\gamma \\nabla \\mathcal{L}'} /> — with the loss set
			to the <em>mean-squared error</em>, the average of <Math tex="(f(x) - y)^2" /> over the training
			points. Every step pulls the network’s curve a little closer to a target.
		</p>
		<p>
			You see the machine three ways at once. On the left sits the network itself: every edge is one
			weight, drawn thicker as it grows, ultramarine when positive and vermilion when negative. On
			the right, the curve it currently draws against the dashed target. And underneath, the palette
			— one numbered row per hidden unit, the very pieces the output layer is adding up. Hover a
			numbered node in the diagram to light up the bump it contributes.
		</p>
		<p>
			Things to try: train on the sine and watch the palette organize — units that began as
			arbitrary steps drift into position. Switch the target to bumps mid-training and watch them
			renegotiate. Then take the pen and draw your own curve straight onto the plot; the network
			will chase it while you drag.
		</p>
	</Prose>

	<Wide>
		<CurveFit />
	</Wide>

	<Prose>
		<h2
			class="mt-14 mb-2 font-serif text-[1.6rem] tracking-tight"
			style="font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			What width buys, what depth buys
		</h2>
		<p>
			Width buys vocabulary. Every extra hidden unit is one more step the output layer can place, so
			a wider palette affords finer detail: at width 2 the sine defeats the network — two steps
			cannot make three bends — while at width 16 it has bumps to spare. Watch the parameter count
			as you widen, though. Vocabulary is not free.
		</p>
		<p>
			Depth buys reuse. With one hidden layer, every wiggle must be purchased with its own neurons.
			Add a second layer and the new neurons stop reading x directly: they read the first layer’s
			steps, and can bend an already-bent thing — bumps of bumps. The same budget of parameters goes
			further because pieces are reused instead of re-made. In one dimension the difference is
			subtle; in the chapters ahead, where inputs are images, it is most of the story.
		</p>
		<p>
			The activation sets the network’s handwriting. tanh is a soft wave, so its sums are smooth and
			rounded everywhere. The <em>rectified linear unit</em> — relu, zero on the left, a straight ramp
			on the right — creases instead of curves: its sums are piecewise-linear, and you can count the folds
			in the fit. Switch the workshop to relu on the |x| target and it lands almost at once, because the
			target is itself two creases. Switch back to tanh and watch it round a corner it can never make
			sharp.
		</p>
		<p>
			One honest warning before you scale everything up. The workshop samples its target at 256
			points packed densely along the interval, so fitting the data and fitting the curve are nearly
			the same task. They usually aren’t. With few points and many neurons, a network can pass
			through every training point exactly and still be wrong everywhere between them — wiggling
			where it should glide. That failure is called <em>overfitting</em>, and it is the tax on
			flexibility. This book meets it properly once real data arrives; for now, notice that nothing
			in the loss ever asked the network to behave between the points.
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
