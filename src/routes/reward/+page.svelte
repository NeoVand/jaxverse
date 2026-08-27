<script lang="ts">
	import { resolve } from '$app/paths';
	import ChapterRef from '$lib/components/ui/ChapterRef.svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import PlateRef from '$lib/components/ui/PlateRef.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import DoublePendulum from '$lib/components/demos/reward/DoublePendulum.svelte';
	import RlLoopDiagram from '$lib/components/demos/reward/RlLoopDiagram.svelte';
	import SeaChart from '$lib/components/demos/reward/SeaChart.svelte';
</script>

<ChapterShell slug="reward">
	<Prose>
		<p>
			Every learner in this book so far had something to copy. The classifier of
			<ChapterRef slug="digits" /> was handed the right answer with every example; the models of
			<ChapterRef slug="latent" /> and <ChapterRef slug="language" /> found their answers hidden inside
			the data itself. This chapter removes the answer entirely. An <em>agent</em> lives inside an
			<em>environment</em>: at each moment it observes a <em>state</em>
			<Math tex={'\\htmlClass{eq-world}{s}'} />, chooses an <em>action</em>
			<Math tex={'\\htmlClass{eq-world}{a}'} />, and the world replies with a new state and a single
			number <Math tex={'\\htmlClass{eq-world}{r}'} /> — the <em>reward</em>. That number is the
			entire curriculum. Nobody demonstrates the right move. Nobody marks the wrong one. The world
			only ever says <em>more</em> or <em>less</em>.
		</p>
	</Prose>

	<RlLoopDiagram />

	<Prose>
		<p>
			The trouble is the timing. Suppose reaching a harbour takes forty separate decisions and the
			score arrives only at the end — you got there, this is what it cost. Which of the forty was
			the mistake? The third, which pointed too high? The nineteenth, which stood on too long? This
			is the <em>credit assignment problem</em><Cite id="minsky-1961" />, and it is the reason this
			paradigm is hard. Supervised learning is told the right action at every step; self-supervised
			learning finds the answer printed in the data. Here the answer exists nowhere. It has to be
			<em>discovered</em>, by acting, and then divided fairly among the actions that led to it.
		</p>
		<p>
			What the agent learns is a <em>policy</em>: a rule <Math
				tex={'\\htmlClass{eq-model}{\\pi_\\theta}(a \\mid s)'}
			/> giving, in every state, a probability for every action — with parameters <Math
				tex={'\\htmlClass{eq-model}{\\theta}'}
			/> we can tune. The objective is plain to state: make the <em>return</em> — the total reward
			of a whole episode — large on average, <Math
				tex={'J(\\htmlClass{eq-model}{\\theta}) = \\mathbb{E}\\,[\\text{return}]'}
			/>. But the world between the parameters and the reward is physics and dice; you cannot
			differentiate through it. The escape is one of the loveliest results in the field, the
			<em>policy gradient theorem</em><Cite id="sutton-2000" />, whose simplest form is an algorithm
			called <em>REINFORCE</em><Cite id="williams-1992" />:
		</p>
		<Math
			display
			tex={'\\nabla_{\\htmlClass{eq-model}{\\theta}}\\, J(\\htmlClass{eq-model}{\\theta}) \\;=\\; \\mathbb{E}\\Big[\\; \\htmlClass{eq-world}{G_t}\\;\\nabla_{\\htmlClass{eq-model}{\\theta}} \\log \\htmlClass{eq-model}{\\pi_\\theta}(\\htmlClass{eq-world}{a_t} \\mid \\htmlClass{eq-world}{s_t}) \\;\\Big]'}
		/>
		<p>
			Read it plainly: <em
				>make the actions that preceded high return more probable, in proportion to that return.</em
			>
			No model of the world, no derivative of the physics — only the policy's own log-probabilities, which
			we can differentiate, weighted by the return <Math tex={'\\htmlClass{eq-world}{G_t}'} /> that actually
			happened. And the update is the same little rule as ever — parameters, plus a step along a gradient
			— walking uphill this time, because <Math tex="J" /> is something to maximize.
		</p>
		<p>
			There is a catch inside that expectation, and it is the reason this paradigm burns through so
			much experience. The estimate is <em>unbiased</em> — average enough episodes and it points the
			right way — but one episode is a dreadful sample of it. A single number, the return,
			multiplies the gradient of <em>every</em> action taken; a lucky passage praises the third leg
			that pointed too high along with the twenty that were sound, and an unlucky one condemns the
			lot together. Set that beside <ChapterRef slug="digits" lower />, where every single example
			handed back a full gradient telling each parameter which way to move. Here a whole episode
			hands back one scalar. Nearly everything built on top of REINFORCE since exists to get more
			signal out of that scalar, or to need fewer of them.
		</p>
		<p>
			The return itself needs one refinement. To split a late score among early actions, we credit
			each moment with everything that followed it, shrunk by a <em>discount</em>
			<Math tex={'\\htmlClass{eq-knob}{\\gamma}'} /> per step of delay:
		</p>
		<Math
			display
			tex={'\\htmlClass{eq-world}{G_t} \\;=\\; \\htmlClass{eq-world}{r_t} + \\htmlClass{eq-knob}{\\gamma}\\, \\htmlClass{eq-world}{r_{t+1}} + \\htmlClass{eq-knob}{\\gamma}^2 \\htmlClass{eq-world}{r_{t+2}} + \\cdots \\;=\\; \\sum_{k=0}^{\\infty} \\htmlClass{eq-knob}{\\gamma}^k\\, \\htmlClass{eq-world}{r_{t+k}}'}
		/>
		<p>
			so an action is answerable for its whole future, near consequences weighing more than distant
			ones. And here is the letter <ChapterRef slug="descent" lower /> was saving:
			<Math tex={'\\htmlClass{eq-knob}{\\gamma}'} /> is the <em>discount</em>, and in this field it
			never means anything else — which is exactly why the stride in every update rule in this book
			is written
			<Math tex={'\\htmlClass{eq-knob}{\\eta}'} />. Two knobs, two letters, and from here on they
			sit side by side in the same equations.<Cite id="sutton-barto-2018" />
		</p>

		<h2 class="h2">A harbour you cannot sail to</h2>
		<p>
			Now the world. It is a sea chart: fourteen cells by ten, a start, a harbour, an islet, two
			shoals, and a wind. The boat picks one of eight compass headings each leg, and the reward is
			simply time — every leg costs, the harbour pays, a shoal ends the passage badly.
		</p>
		<p>
			Two rules make this more than a maze. A sailing boat <em>cannot sail into the wind</em>: point
			within 35° of it and the sails luff, the boat stops, and the clock keeps running. And a boat
			is not equally quick on every heading — it is fastest with the wind on the beam, at ninety
			degrees, slower close-hauled, and slower again running dead downwind. So the cost of a leg
			depends on its angle to the wind, which is a real thing sailors call a polar diagram and is
			the entire physics of this page.
		</p>
		<p>
			Look at where the harbour is: almost dead upwind of the start. There is no route that points
			at it. Whatever the policy comes back with, it will be something nobody wrote down.
		</p>
	</Prose>

	<SeaChart />

	<UnderTheHood slug="reward" block="reinforce" />

	<Prose>
		<h2 class="h2">Reading the chart</h2>
		<p>
			Watch the first seconds: a small blob in every cell, near enough to a circle. That blob
			<em>is</em> the policy. Each cell carries one <em>current rose</em> — a radius in each of the eight
			compass directions, drawn to the probability the policy gives it — so a learner with no opinion
			draws a circle, and a learner that has made up its mind draws a kite. The headings the wind forbids
			are not drawn cell by cell, because they are the same in every cell: they are the shaded wedge on
			the compass rose beside the chart, and they are the one mistake this world punishes immediately.
			Everything else has to wait for the end of the passage to find out whether it helped.
		</p>
		<p>
			Then somewhere a wandering boat blunders into the harbour, return-to-go hands credit back down
			the whole track, and the blobs along it stretch into kites. Within a few thousand passages the
			field has organized into something you can read at a glance — and the shape it organizes into
			is a
			<strong>zigzag</strong>. To make ground upwind the boat sails as close to the wind as it can
			on one side, then swaps to the other, then back. That is <em>tacking</em>, and it is what
			every sailor on earth does, and nothing anywhere in the reward mentions it. It falls out of a
			rule about what a boat can do and a number that counts time.
		</p>
		<p>
			That is the difference between this world and a maze. In a maze the best policy is the
			shortest path, which is the answer you already knew before the learner started, so watching it
			arrive teaches you only that the code works. Here the best policy is a manoeuvre with a name,
			and the learner has no idea it has a name.
		</p>
		<p>
			Now spin the compass rose. The wind is the only thing that changes, and the entire field
			reorganizes underneath a learner that never stopped running — new no-go headings go vermilion,
			the old route stops paying, and a different zigzag grows in. Turn the wind behind the boat and
			the zigzag vanishes altogether, because now it can simply go: the policy did not become
			smarter, the problem became easier, and telling those two apart by looking at a reward curve
			is most of what applied reinforcement learning actually is.
		</p>

		<h2 class="h2">Exploring, and failing to</h2>
		<p>
			There is no dial for exploration on this plate, and that is the point. The policy is a
			probability distribution and never stops being one — early on it sails almost at random, which
			is how it discovers anything at all, and even when trained every rose keeps a little width to
			it: a sliver of belief on headings it has decided against.
		</p>
		<p>
			Underneath the roses is what the learner has worked out about the water itself. The
			<span class="num">value</span> layer draws the baseline — its running estimate of how a
			passage tends to go from each cell — as soundings: ultramarine where a boat tends to get home
			from, vermilion where it tends not to, and a heavier line marking the shore between them,
			which sits exactly where the expected return crosses zero. Watch that shoreline early on and
			you can see the harbour's influence seeping outward, one cell at a time, long before any
			single passage looks competent. That spreading is what return-to-go <em>is</em>.
		</p>
		<p>
			But a wind does something to exploration that a maze cannot, and this world had to be built
			twice before it worked. A policy that picks headings uniformly <em>drifts downwind</em>. So a
			harbour set to windward is worse than far. A random walk essentially never reaches it, and on
			some wind directions the learner never found it at all. What it found instead was that sailing
			straight onto a shoal ended the passage for nine points, while wandering until the clock ran
			out cost far more. So it did that. Deliberately, reliably, and with a reward curve that simply
			flattened out and told you nothing.
		</p>
		<p>
			The fix is the one <PlateRef id="pendulum" lower /> uses on a much harder problem: one passage in
			five starts not at the mooring but at a random patch of water. Practising what you can already almost
			do is how the gradient stays informative, and with that one change every wind direction is learnable.
			This is not a detail of the demo. Where an episode begins is part of the algorithm rather than part
			of the problem<Cite id="kakade-langford-2002" />, and the strongest version of the idea runs
			it backwards on purpose: start the agent beside the goal, where succeeding is nearly free, and
			walk the start states away as it improves.<Cite id="florensa-2017" /> Somebody had to find that
			out, and the only symptom was a curve that looked fine.
		</p>
		<p>
			Which is also your invitation to break it. Switch on <em>edit</em> and redesign the sea under a
			live learner: wall off the tacking corridor and watch the roses go round again before a new channel
			carves itself; drop a shoal on the layline; drag the harbour into a corner and watch a confident
			policy become a wanderer. Some of the worlds you build will not be learnable at all, and the plate
			will tell you so honestly — the arrival rate falls, the dashed line stays where it is, and nothing
			else announces the problem. The policy optimizes the reward you wrote in the world you built. Every
			reinforcement learning system inherits both clauses.
		</p>

		<h2 class="h2">Where the real work is</h2>
		<p>
			The chart is small enough to see whole. Almost nothing in practice is, so here is the same
			algorithm on something that is not: the classic cart-and-pole, made meaner twice over. Barto,
			Sutton and Anderson set the original in 1983, and it was a study in credit assignment from the
			start: the learner was told nothing at all until the pole fell past an angle or the cart ran
			off the end of its track — one bit, at the end of everything it had done.<Cite
				id="barto-1983"
			/> A hinge sliding on a short rail, <em>two</em> pendulum links stacked on top of it, and the
			stack starting where gravity wants it — hanging straight down. The policy must learn to
			<em>swing it up</em>: pump energy in with rhythmic pushes, steer the top link
			<em>through</em> the bottom one, and then catch the whole thing upright inside the ±18° cones. It
			reads six numbers through hand-crafted gauges, and five pushes are available, fifty times a second.
		</p>
		<p>
			Read the next two paragraphs as the real subject of this section, because they are not
			apparatus notes — they are what reinforcement learning is actually like once the world stops
			fitting on a page. The update rule does not change. Everything else does.
		</p>
		<p>
			The reward splits at a height, and its shape is the answer to a cheat. Down low the policy is
			paid once, at the moment it <em>delivers</em> the tip past the hand-off, graded on exactly
			what a catcher would ask for: arrive slowly, links near vertical, cart clear of the rail's
			ends. Up high, every calm tick inside the cones pays rent. Everything else is fines — spin
			beyond what an honest pump needs, energy beyond what the top requires, loitering against a
			bumper — because a stack spun like a propeller <em>looks</em> upright twice per revolution,
			and without the fines a policy happily farms those moments forever. Every one of those clauses
			is scar tissue from a policy that found the loophole first — which is the ordinary experience
			of writing a reward, not an unlucky one. The canonical example is a boat in a racing game that
			discovered it could score more by circling forever through a patch of pickups than by
			finishing the course, and there is a long, funny, faintly alarming catalogue of the rest.<Cite
				id="krakovna-2020"
			/>
		</p>
		<p>
			And the practice regime is the chart's scatter, grown up. Half the headless practice swings up
			— from hanging, or from mid-tumble, so that braking a botched attempt back into a clean swing
			is a practised move — a quarter starts balanced to learn the hold, and a quarter
			<em>replays the swing's own deliveries</em>, eased toward vertical at first and raw as the
			success rate earns it. That reverse curriculum is what makes the catch learnable at all.
			Below, the policy starts as a coin flip. Press Train, watch it discover swinging, then
			delivering, then the catch — and shove the hinge any time; knock it clean over and it will
			swing back up.
		</p>
	</Prose>

	<DoublePendulum />

	<UnderTheHood slug="reward" block="pendulum" />

	<Prose>
		<h2 class="h2">The same move, one more time</h2>
		<p>
			What changed between the flailing policy and the one that climbs? A hundred and sixty-five
			numbers, nudged by REINFORCE after every episode. Each push is judged by the return that
			followed it, minus a <em>baseline</em>: the running average of how episodes usually go from
			that moment. The difference is called the <em>advantage</em>, and it makes "good" mean
			<em>better than usual</em> — a push followed by ten seconds of height teaches nothing once that
			height is normal, but the same push while the stack usually dangles is news. This is the cheapest
			of the repairs to that one noisy scalar: subtracting a baseline leaves the average gradient exactly
			where it was, and narrows the spread around it. Free, in the only currency that matters here.
		</p>
		<p>
			Notice that your shoves were never outside the lesson: a disturbance the policy survives
			becomes an episode with high return, so recovering from you is precisely what it was rewarded
			for learning. It learned feedback, not a script — that is why it holds against shoves it has
			never seen.
		</p>
		<p>
			And nothing on this page has touched your GPU. The policies are a few dozen numbers, the
			gradients a few multiplications, hundreds of passages a second in plain TypeScript. The hard
			part of reinforcement learning was never the compute. It is the credit — and, as the chart
			showed you, getting the agent to stumble into anything worth assigning credit for.
		</p>
		<p>
			The table of roses you trained is a policy in the full sense of the word — the same
			mathematical object the phrase always means — and nothing in REINFORCE ever looked inside it.
			The update touched only two things: the probability the policy assigned to an action, and the
			return that followed. The pendulum's weights sat in the same seat. Any object that can assign
			probabilities and accept gradients can sit there.
		</p>
		<p>
			So swap the table for a neural network. Let the state be a conversation so far, the action a
			next token, the episode a whole reply. The equation at the top of this chapter does not
			change. What does change is the hardest question in the subject, and this chapter has been
			quietly ducking it: <em>where did the number come from?</em> A harbour is worth twelve because I
			typed twelve. Nobody can type the number for a helpful answer, a tactful paragraph, a well-set page.
		</p>
		<p>
			<a href={resolve('/taste')}>The next chapter</a> is what you do then — and what it costs you.
		</p>
	</Prose>
</ChapterShell>
