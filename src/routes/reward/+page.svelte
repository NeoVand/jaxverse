<script lang="ts">
	import { resolve } from '$app/paths';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import Wide from '$lib/components/ui/Wide.svelte';
	import DoublePendulum from '$lib/components/demos/reward/DoublePendulum.svelte';
	import Gridworld from '$lib/components/demos/reward/Gridworld.svelte';
	import RlLoopDiagram from '$lib/components/demos/reward/RlLoopDiagram.svelte';
</script>

<ChapterShell slug="reward">
	<Prose>
		<p>
			Every learner in this book so far had something to copy. The classifier of
			<a href={resolve('/digits')}>Chapter 3</a> was handed the right answer with every example; the
			models of Chapters 4 and 5 found their answers hidden inside the data itself. This chapter
			removes the answer entirely. An <em>agent</em> lives inside an <em>environment</em>: at each
			moment it observes a <em>state</em>
			<Math tex="s" />, chooses an <em>action</em>
			<Math tex="a" />, and the world replies with a new state and a single number <Math tex="r" /> —
			the <em>reward</em>. That number is the entire curriculum. Nobody demonstrates the right move.
			Nobody marks the wrong one. The world only ever says <em>more</em> or <em>less</em>.
		</p>

		<RlLoopDiagram />

		<p>
			The trouble is the timing. Suppose balancing a pole takes two hundred little pushes and the
			score arrives only as the run ends — you lasted this long, nothing more. Which of the two
			hundred pushes was the mistake? The third, which overcorrected? The ninetieth, which drifted
			toward the edge? This is the <em>credit assignment problem</em>, and it is the reason this
			paradigm is hard. Supervised learning is told the right action at every step; self-supervised
			learning finds the answer printed in the data. Here the answer exists nowhere. It has to be
			<em>discovered</em>, by acting, and then divided fairly among the actions that led to it.
		</p>
		<p>
			What the agent learns is a <em>policy</em>: a rule <Math
				tex={'\\htmlClass{eq-a}{\\pi_\\theta}(a \\mid s)'}
			/> giving, in every state, a probability for every action — with parameters <Math
				tex="\theta"
			/> we can tune. The objective is plain to state: make the <em>return</em> — the total reward
			of a whole episode — large on average, <Math
				tex={'J(\\theta) = \\mathbb{E}\\,[\\text{return}]'}
			/>. But the world between the parameters and the reward is physics and dice; you cannot
			differentiate through it. The escape is one of the loveliest results in the field, the
			<em>policy gradient theorem</em>, whose simplest form is an algorithm called
			<em>REINFORCE</em>:
		</p>
		<Math
			display
			tex={'\\nabla_\\theta\\, J(\\theta) \\;=\\; \\mathbb{E}\\Big[\\; \\htmlClass{eq-w}{G_t}\\;\\nabla_\\theta \\log \\htmlClass{eq-a}{\\pi_\\theta}(a_t \\mid s_t) \\;\\Big]'}
		/>
		<p>
			Read it plainly: <em
				>make the actions that preceded high return more probable, in proportion to that return.</em
			>
			No model of the world, no derivative of the physics — only the policy's own log-probabilities, which
			we can differentiate, weighted by the return <Math tex={'\\htmlClass{eq-w}{G_t}'} /> that actually
			happened. And the update is the same little rule as ever — parameters, plus a step along a gradient
			— walking uphill this time, because <Math tex="J" /> is something to maximize.
		</p>
		<p>
			The return itself needs one refinement. To split a late score among early actions, we credit
			each moment with everything that followed it, shrunk by a <em>discount</em>
			<Math tex="\gamma" /> per step of delay:
		</p>
		<Math
			display
			tex={'G_t \\;=\\; r_t + \\gamma\\, r_{t+1} + \\gamma^2 r_{t+2} + \\cdots \\;=\\; \\sum_{k=0}^{\\infty} \\gamma^k\\, r_{t+k}'}
		/>
		<p>
			so an action is answerable for its whole future, near consequences weighing more than distant
			ones. One housekeeping note: <Math tex="\gamma" /> is the discount here, not the learning rate —
			reinforcement learning claimed the letter first, so this chapter's sliders say
			<em>learning rate</em> in words.
		</p>
		<p>
			Now the testbed — the classic cart-and-pole made meaner twice over: a hinge sliding on a short
			rail, <em>two</em> pendulum links stacked on top of it, and the stack starting where gravity
			wants it — hanging straight down. The policy must learn to <em>swing it up</em>: pump energy
			in with rhythmic pushes, steer the top link <em>through</em> the bottom one (the hinge is all
			it can touch), and then catch the whole thing upright, where the ±18° cones are. The state is
			six numbers — hinge position and velocity, two angles and their velocities — read through
			sixteen hand-crafted gauges: swing gauges that fade out near the top, fine balance gauges that
			fade in there. Five pushes are available, hard or gentle, left or right, or nothing, fifty
			times a second. The reward is simply the height of the tip at every tick, plus a hefty bonus
			for standing caught and calm — no rulebook, no instructions, just
			<em>higher is better</em>. Nothing ever fails and nothing resets: an episode is an
			eight-second window, graded and moved past. One honest confession about the practice regime:
			half the policy's headless practice runs start hanging, a quarter start already balanced (to
			learn the hold before the climb), and a quarter start at the brink — tilted and moving near
			the top, exactly where the catch is decided. Below, that policy starts as a coin flip. Press
			Train, watch it discover swinging, then climbing, then — patience — the catch. Shove the hinge
			any time: click or drag on the stage.
		</p>
	</Prose>

	<Wide>
		<DoublePendulum />
	</Wide>

	<Prose>
		<h2
			class="mt-12 mb-3 tracking-tight"
			style="font-size: 1.55rem; line-height: 1.25; font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			Reading the recovery
		</h2>
		<p>
			What changed between the flailing policy and the one that climbs? Eighty numbers — the weights
			on sixteen gauges for five pushes — nudged by REINFORCE after every episode. Each push is
			judged by the return that followed it, minus a <em>baseline</em>: the running average of how
			episodes usually go from that moment. The difference is called the
			<em>advantage</em>, and it makes "good" mean <em>better than usual</em> — a push followed by
			ten seconds of height teaches nothing once that height is normal, but the same push while the
			stack usually dangles is news. That reward-minus-usual idea is exactly what
			<a href={resolve('/rook')}>Chapter 7</a> will reuse on a language model, under the same name.
		</p>
		<p>
			Notice, too, what exploration looks like here: there is no dial for it. The policy is a
			probability distribution and never stops being one — early on it pushes almost at random,
			which is how it discovers that leaning left wants a leftward push at all, and even trained it
			keeps a sliver of probability on the losing action. And your shoves were not outside the
			lesson: a disturbance the policy survives becomes an episode with high return, so recovering
			from you is precisely what it was rewarded for learning. It learned feedback, not a script —
			that is why it holds against shoves it has never seen.
		</p>
		<p>
			One thing the pendulum cannot show you is the policy itself — its states are six continuous
			numbers, and there are infinitely many of them. To look at a whole policy at once we need a
			world small enough to draw. Below is an 8-by-6 field: a start (the small ring), a treasure ◆
			worth +10, pits worth −8, and a tax of −0.15 on every step — walls block you, and bumping one
			still costs. The policy is a bare table, four numbers per cell drawn as four arrows whose
			opacity is their probability; the <em>value</em> chip overlays its baseline — the running estimate
			of each cell's worth — as a blue wash. And it is worth saying out loud that nothing on this page
			has touched your GPU: the policies are a few dozen numbers, the gradients a few multiplications,
			hundreds of episodes a second in plain TypeScript. The hard part of reinforcement learning was never
			the compute. It is the credit.
		</p>
		<p>
			Press Play. The dashed line in the return chart is the best any route can do — the shortest
			safe path, found by ordinary search — so you can see exactly how far from perfect the policy
			is at every moment. And a warning worth experiencing on purpose: push the learning rate high
			and the policy can seize up while still ignorant, freezing on a route to nowhere — a
			policy-space cousin of Chapter 0's divergence. Reset θ forgives everything.
		</p>
	</Prose>

	<Wide>
		<Gridworld />
	</Wide>

	<Prose>
		<h2
			class="mt-12 mb-3 tracking-tight"
			style="font-size: 1.55rem; line-height: 1.25; font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			Reading the river
		</h2>
		<p>
			Watch the arrows in the first seconds: nearly uniform, four faint directions in every cell.
			That haze <em>is</em> exploration — the same explore-exploit bargain the pendulum struck,
			visible cell by cell. Then somewhere a random walk stumbles into the treasure, return-to-go
			hands credit back down the whole path, and those arrows brighten. Runs that revisit the route
			reinforce it. Within a few hundred episodes the gradient of <em>more</em> has carved a river through
			the grid — and if two routes were equally short, it chose one essentially by coin flip and then
			deepened its own channel.
		</p>
		<p>
			Now read the reward like a designer. The step tax is why the river is <em>short</em>: every
			wasted move subtracts from the very number the policy climbs. The pits teach the opposite
			lesson: an −8 next to a corridor makes the arrows lean away from the edge like a walker
			hugging the inside of a cliff path. And you don't have to take this on faith — flip on
			<em>edit</em> and redesign the world under a live learner. Wall off the river mid-course and watch
			the arrows go hazy, then carve a new channel around your wall; drop a pit on the old route; drag
			the treasure across the map and watch a confident policy become a wanderer again. The policy optimizes
			the reward you wrote, not the intention you had. Every reinforcement learning system inherits this
			clause.
		</p>
		<p>
			And the credit question from the opening has its answer. The nineteenth step of a fifty-step
			episode is paid <Math tex={'G_{19}'} /> — everything that happened after it, discounted — minus
			the baseline for the cell it stood in, so its update reads:
			<em>from here, this choice went better than usual</em>. Early steps get credit for late
			treasure; steps before a pit get the blame; and "usual" is judged cell by cell, which is why
			the wash under the value chip is itself a little map of hope.
		</p>

		<h2
			class="mt-12 mb-3 tracking-tight"
			style="font-size: 1.55rem; line-height: 1.25; font-weight: 520; font-variation-settings: 'opsz' 28;"
		>
			The same move, one more time
		</h2>
		<p>
			The table of arrows you trained is not a toy version of a policy. It <em>is</em> a policy — the
			same mathematical object the phrase always means — and nothing in REINFORCE ever looked inside it.
			The update touched only two things: the probability the policy assigned to an action, and the return
			that followed. The pendulum's eighty weights sat in the same seat. Any object that can assign probabilities
			and accept gradients can sit there.
		</p>
		<p>
			So swap the table for a neural network. Let the state be a conversation so far, the action a
			next token, the episode a whole reply — and let the reward be a human preference score, or a
			checkable fact: the proof verifies, the tests pass, the answer matches. The equation at the
			top of this chapter does not change. That is RLHF and RLVR — the techniques that turned raw
			next-token predictors into usable assistants — and they are this gridworld wearing a language
			model, with the baseline idea reappearing under the name <em>advantage</em>.
		</p>
		<p>
			One chapter remains, and it is the whole pipeline at once: a small language model that learns
			chess by prediction, then by imitation of better games, then by verifiable reward —
			pretraining, fine-tuning, and reinforcement learning in one small machine.
			<a href={resolve('/rook')}>Meet Rook</a>.
		</p>
	</Prose>
</ChapterShell>
