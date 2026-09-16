<script lang="ts">
	import { onDestroy } from 'svelte';
	import ChapterRef from '$lib/components/ui/ChapterRef.svelte';
	import ChapterShell from '$lib/components/ui/ChapterShell.svelte';
	import Cite from '$lib/components/ui/Cite.svelte';
	import Math from '$lib/components/ui/Math.svelte';
	import PlateRef from '$lib/components/ui/PlateRef.svelte';
	import Prose from '$lib/components/ui/Prose.svelte';
	import UnderTheHood from '$lib/components/ui/UnderTheHood.svelte';
	import WorldPlate from '$lib/components/demos/world/WorldPlate.svelte';
	import ArchitectureDiagram from '$lib/components/demos/world/ArchitectureDiagram.svelte';
	import { createWorldLab } from '$lib/world/lab.svelte';

	const lab = createWorldLab();
	onDestroy(() => lab.dispose());
</script>

<ChapterShell slug="world">
	<Prose>
		<p>
			Before you catch a moving object, you make a prediction. Where it is now matters, but so does
			how it arrived there. You can reach toward where it will be because the recent past contains
			evidence about the next moment. Choosing an action becomes easier when you can anticipate its
			consequences.
		</p>
		<p>
			The two instruments below occupy the same pose. Give both the same push and they move
			differently. A joint can pass through an angle in either direction; a still picture leaves
			that motion out. Apply the torques, then reset and reveal the history. The faint preceding
			poses supply the clue that the two present pictures cannot.
		</p>
		<p>
			This opening experiment is entirely a simulator. Its links move on a horizontal plane, with
			motors turning the joints and no gravity pulling them downward. Damping is switched off here
			so their momentum remains visible. No network has learned anything yet.
		</p>
	</Prose>

	<WorldPlate mode="history" {lab} />

	<Prose>
		<p>
			A picture is an <em>observation</em> of the world. It need not reveal all the information that determines
			what happens next. Here, two recent pictures give evidence about motion, and the motor command supplies
			another part of the explanation. A learner has to bring those pieces together: the recent past,
			the present, and the action being considered.
		</p>
		<p>
			In <ChapterRef slug="reward" />, we learned a policy that chose actions to earn reward. This
			time we will learn their consequences first. A <em>world model</em> predicts how an environment
			changes. When it takes an action as an input, we can try different actions inside the model before
			committing to one in the world. Add a way to score the predicted outcomes and we have the ingredients
			of a planner.
		</p>
		<p>
			That separation makes the knowledge reusable. Knowing what a push does can help with reaching
			one destination, reaching another, or slowing down before arrival. The experience need not
			have been gathered for each of those requests. Learned prediction and separately specified
			objectives are central to LeCun's proposed architecture for autonomous intelligence.<Cite
				id="lecun-2022-ami"
			/> Its memory and hierarchies go much further; here we will build a small, inspectable instance
			of learning a model and then thinking with it.
		</p>
		<h2 class="h2">Learning before wanting</h2>
		<p>
			First we collect experience without a destination. The simulator tries motor commands: keep
			pushing, reverse the torques, release. Damping is now present, so motion gradually loses
			energy when the motors are off. Each observation is a 32 × 32 sensor picture, and each action
			is a pair of numbers controlling the two joint torques. The collection contains no rewards,
			task labels, or demonstrations of how to reach a requested destination.
		</p>
		<p>
			The simulator knows its joint angles and velocities, but training receives only pictures and
			motor commands. This distinction is what makes the experiment interesting. Hand the learner
			the physical state and we have already chosen its representation. Hand it images and it must
			learn a useful way to describe what it sees. Its sensor renders the mechanism alone. Colored
			goal outlines, motion traces, and predicted ghosts are annotations for you; they never enter
			its camera.
		</p>
		<p>
			There are 1,024 pixel values in each observation. Predicting every one would require an answer
			about every visible detail. A useful prediction can be less demanding. To move a cup, for
			example, its position and motion may matter much more than the changing reflection on its
			surface. A representation gives a model room to retain distinctions that support prediction
			without having to reproduce the whole appearance. Our simple drawings keep this problem small
			enough to study in a browser.
		</p>
		<p>
			In <ChapterRef slug="latent" />, reconstruction taught the encoder what to preserve. Here we
			will learn an encoder and a predictor together. The encoder compresses each picture to eight
			numbers, an <em>embedding</em>. The predictor receives two such embeddings and two actions and
			tries to produce the embedding of the next picture:
		</p>
		<Math
			display
			tex={'\\begin{aligned}\\htmlClass{eq-model}{z_t} &= \\htmlClass{eq-model}{E_\\theta}(\\htmlClass{eq-world}{o_t}), \\\\[6pt] \\htmlClass{eq-out}{\\hat z_{t+1}} &= \\htmlClass{eq-model-2}{P_\\phi}(\\htmlClass{eq-model}{z_{t-1}, z_t}, \\htmlClass{eq-world}{a_{t-1}, a_t}).\\end{aligned}'}
		/>
		<p>
			Follow the time indices from left to right. <Math tex={'\\htmlClass{eq-world}{a_{t-1}}'} /> was
			applied between the two pictures we have already seen. <Math
				tex={'\\htmlClass{eq-world}{a_t}'}
			/> is the proposed next action. The two pictures provide evidence about motion; the action history
			helps explain how that motion changed. The hat on <Math
				tex={'\\htmlClass{eq-out}{\\hat z_{t+1}}'}
			/> marks a prediction. Its target is the actual next picture passed through the same encoder.
		</p>
		<p>
			The eight coordinates have no assigned names such as “elbow angle” or “speed.” Each describes
			a picture in a space learned by the encoder; the predictor combines successive descriptions to
			reason about change. Blue marks that representation, violet the predictor, and green its
			answer. The architecture predicts in a learned space on both sides: a <em
				>joint embedding predictive architecture</em
			>, or JEPA.<Cite id="assran-2023-ijepa" />
		</p>
		<p>
			Our training recipe follows LeWorldModel: learn the visual encoder and action-conditioned
			predictor jointly, while constraining the distribution of the encoder's representations.<Cite
				id="maes-2026-lewm"
			/> LeWM uses transformers. These tiny images let us use two small multilayer perceptrons instead,
			so the entire experiment can begin from random weights and train in this tab.
		</p>
	</Prose>

	<ArchitectureDiagram />

	<Prose>
		<p>
			One training example contains three successive pictures. Encode all three. Give the first two
			embeddings and the two intervening actions to the predictor, then compare its answer with the
			third embedding. The prediction loss is their mean squared difference:
		</p>
		<Math
			display
			tex={'\\mathcal{L}_{\\mathrm{pred}} = \\frac{1}{d}\\sum_{j=1}^{d}\\left(\\htmlClass{eq-out}{\\hat z_{t+1,j}}-\\htmlClass{eq-model}{z_{t+1,j}}\\right)^2, \\qquad d=8.'}
		/>
		<p>
			Gradient descent adjusts both networks. It changes how the predictor forecasts and how the
			encoder represents the pictures, including the target picture. That moving target gives the
			encoder freedom to find a predictable description of the world. We also add a distribution
			penalty that discourages describing every picture identically. The next plots show both terms;
			in <PlateRef id="collapse" /> we will remove the constraint and see why it matters.
		</p>
		<p>
			Before pressing Train, inspect one example below. The first two pictures and their motor
			commands pose the question; one of the six pictures is what actually followed. To display an
			answer, we encode those six pictures and find the one nearest to the predicted embedding. This
			is a test of the learned representation, separate from the training objective. The network is
			never trained to choose a letter or to paint these candidate pictures.
		</p>
		<p>
			Now train, keeping the same example in view. Compare <em>Before learning</em> with
			<em>Now</em>, then inspect a few other examples. The questions and alternatives stay fixed
			while the weights change. The score counts correct matches across 64 held-out moments,
			including the ones that are not on screen; these observations never supply a training
			gradient.
		</p>
	</Prose>

	<WorldPlate mode="train" {lab} />
	<UnderTheHood slug="world" block="learn" />

	<Prose>
		<p>
			The loss and the picture test tell us different things. Loss measures numerical agreement
			between a forecast and its target representation. A correct match means the forecast was
			closer to the recorded future than to the five alternatives. That makes improvement something
			you can inspect, although it is still a choice among six pictures, not proof of an exact
			prediction. Individual examples can remain wrong even as the overall score improves.
		</p>
		<p>
			<em>Copy the present</em> asks how far we could get without predicting change at all. It uses
			the current picture's embedding as the forecast and matches it against the same alternatives.
			Nearby moments often look alike, so this can already be a useful guess. Beating it is evidence
			that learning to predict a change helps. This baseline uses the <em>current</em> encoder, so its
			score can change during training even though its copying rule never does.
		</p>
		<p>
			The untrained model need not start at one correct answer in six. Its predictor adds a learned
			change to the present embedding, so even its initial answer can resemble the present. The
			candidate observations are related pictures, too. Starting from random weights does not make
			its choices uniformly random, which is another reason to compare it with the copying rule.
		</p>
		<h2 class="h2">A future made of predictions</h2>
		<p>
			Training always supplied real pictures for the context. To look further ahead, let the first
			predicted embedding stand in for the next observation. Slide the two-frame history forward,
			append another proposed action, and predict again. Repeating this builds a <em>rollout</em>: a
			sequence of consequences computed without advancing the real mechanism.
		</p>
		<p>
			In the next plate, every action choice starts after the same short push. Keep pushing, reverse
			the torques, or release the motors. Release still produces motion because the mechanism is
			already moving. Start with three observations ahead, then try a longer forecast. In each case,
			the ink instrument replays exactly the action sequence given to the predictor. The weights
			stay fixed throughout.
		</p>
		<p>
			The blue drawing is a window into the embedding forecast. A separate <em>readout</em>, fitted
			after freezing the encoder, translates embeddings into joint angles for display. Only this
			display fit uses angle labels for learning. It does not teach the world model or choose
			actions. Its own error is reported beside the rollout error, because a gap between the
			drawings can come from either the predicted embedding or its translation into a pose.
		</p>
	</Prose>

	<WorldPlate mode="forecast" {lab} />

	<Prose>
		<p>
			Watch agreement with the observed movement, especially as you extend the horizon. After the
			first forecast, the predictor is receiving something it produced itself. A small error can
			therefore become part of the next question, and the next. A model that is useful one step
			ahead can drift badly over a long rollout. The clean shape of a blue arm cannot settle this:
			rigid links are built into the drawing, while accurate dynamics have to be learned.
		</p>
		<p>
			This suggests a practical way to use imperfect forecasts. Look ahead to choose a promising
			action, execute a little of the plan, then observe again. The new picture can replace an
			inaccurate imagined present. We will use that loop shortly. First we need to understand a more
			fundamental failure: a model can make its prediction loss small while learning almost nothing
			useful.
		</p>
		<h2 class="h2">A prediction of nothing</h2>
		<p>
			Suppose the encoder returns the same eight numbers for every picture. The predictor can return
			those numbers too. Its prediction error is zero, yet an open instrument, a folded instrument,
			and a desired destination have become indistinguishable. This is
			<em>representation collapse</em>. Because the encoder is learned, low prediction loss alone
			cannot rule it out.
		</p>
		<Math
			display
			tex={'\\mathcal{L} = \\mathcal{L}_{\\mathrm{pred}} + \\htmlClass{eq-knob}{\\lambda}\\,\\htmlClass{eq-op}{\\mathrm{SIGReg}}(\\htmlClass{eq-model}{Z}).'}
		/>
		<p>
			The second term asks for variation across observations. <em
				>Sketched isotropic Gaussian regularization</em
			>, SIGReg, projects a batch of embeddings onto many directions and compares the resulting
			distributions with a standard Gaussian.<Cite id="balestriero-2025-lejepa" /> One constant vector
			would produce a pileup at a single value in every direction, which fails this test. The amber coefficient
			<Math tex={'\\htmlClass{eq-knob}{\\lambda}'} /> controls how strongly this requirement competes
			with prediction.
		</p>
		<p>
			There are now two pressures: make what is preserved predictable, and preserve enough
			differences to maintain the required distribution. The Gaussian describes the collection of
			embeddings across different pictures. Unlike the variational autoencoder, this encoder does
			not draw a random latent sample around each individual picture.
		</p>
		<p>
			Press Compare below. The right-hand model restarts from the same initial weights and sees the
			same training examples, with the distribution term removed. Your trained model stays fixed on
			the left while that run catches up. Read the prediction loss together with the spread and the
			held-out matching score. The experiment is about which distinctions survive, not whether a
			loss curve can be made to descend.
		</p>
	</Prose>

	<WorldPlate mode="collapse" {lab} />
	<UnderTheHood slug="world" block="spread" />

	<Prose>
		<p>
			A representation can contract without becoming perfectly constant. It can also have plenty of
			spread while preserving the wrong information. The distribution plot therefore answers only
			part of the question. The matching test asks whether the future remains distinguishable, and
			control will ask whether the representation supports a useful decision. Compare the runs at
			the same number of updates; their raw latent losses also depend on the different scales the
			encoders have learned.
		</p>
		<p>
			Prediction in a learned space makes this trade possible: the model can retain useful structure
			without solving every detail of future appearance. Feature prediction in video models explores
			the same possibility on richer observations.<Cite id="bardes-2024-vjepa" />
			What survives still has to earn its usefulness. On this page, the next test is whether the model
			can help choose an action toward a goal it was never trained to achieve.
		</p>
		<h2 class="h2">Turn a forecast into a decision</h2>
		<p>
			A world model predicts what could follow an action. It needs a separate instruction about
			which outcome we want. Choose one of the desired poses below. We make a clean picture of it
			and pass that picture through the same encoder:
			<Math
				tex={'\\htmlClass{eq-model}{z_g} = \\htmlClass{eq-model}{E_\\theta}(\\htmlClass{eq-world}{o_g})'}
			/>. A goal image specifies the whole visible pose; two different elbow positions can put the
			tip in the same place, so matching only the tip would be a different task.
		</p>
		<p>
			For any proposed action sequence, roll the model forward and measure how far its final
			predicted embedding lies from the goal embedding. That distance becomes a <em>cost</em>:
		</p>
		<Math
			display
			tex={'C_{\\mathrm{goal}} = \\frac{1}{d}\\left\\lVert\\htmlClass{eq-out}{\\hat z_{t+H}} - \\htmlClass{eq-model}{z_g}\\right\\rVert^2.'}
		/>
		<p>
			Here <Math tex="d=8" /> is the number of embedding coordinates and <Math
				tex={'\\htmlClass{eq-knob}{H}'}
			/> is how many actions we consider ahead. A smaller cost means the model predicts a closer match.
			We also give the planner a mild preference for smaller motor commands:
		</p>
		<Math
			display
			tex={'C = C_{\\mathrm{goal}} + \\frac{\\htmlClass{eq-knob}{\\rho}}{2\\htmlClass{eq-knob}{H}}\\sum_{k=0}^{\\htmlClass{eq-knob}{H}-1}\\lVert \\htmlClass{eq-world}{a_{t+k}}\\rVert^2, \\qquad \\htmlClass{eq-knob}{\\rho}=0.01.'}
		/>
		<p>
			Each action has two coordinates between −1 and 1, so the second term is a weighted mean of
			their squares. Both preferences belong to the planner. Neither was used to train the world
			model. During training we optimized <em>weights</em> to explain recorded experience; during
			planning we search over <em>actions</em>, using those weights unchanged.
		</p>
		<p>
			The search samples candidate torque sequences, keeps the better ones, and samples again near
			them. This is the <em>cross-entropy method</em>. Rehearse lets you inspect three candidates
			from that larger search while the instrument stays still. Step executes only the first action
			of the selected sequence. Run then takes another observation and starts a new search. This
			<em>model predictive control</em> loop is also used with learned visual models such as V-JEPA
			2-AC to plan toward image goals.<Cite id="assran-2025-vjepa2" />
		</p>
	</Prose>

	<WorldPlate mode="plan" {lab} />
	<UnderTheHood slug="world" block="plan" />

	<Prose>
		<p>
			Try Rehearse followed by Step before using Run. The blue first-step prediction stays visible
			after the action, so you can compare what was expected with what occurred. On the next
			iteration the planner begins from the newly observed pictures, even when they disagree with
			its previous forecast. Feedback corrects its starting information; it does not update the
			weights.
		</p>
		<p>
			The alternatives are different action sequences under one model. Their spacing does not show
			the model's confidence. A low predicted cost can identify a useful plan, but search can also
			find a place where the predictor is overoptimistic. Longer foresight adds room to act and more
			opportunities for error, while giving the same search budget a larger problem. The observed
			movement is the final test of the plan.
		</p>
		<h2 class="h2">Change what you want</h2>
		<p>
			The trained model has never been asked to unfold, return, or curl. Those names belong to our
			requests. To change the destination we supply a different goal embedding and score the
			predicted futures again. The facts about how actions change the mechanism can stay in place
			while the preference changes.
		</p>
		<p>
			We can also change what counts as arriving. A photograph gives a pose, but the mechanism may
			be moving quickly as it passes through it. Scoring only the final moment can favor that brief
			match. To encourage it to remain nearby, average the goal error over the last four predicted
			moments instead:
		</p>
		<Math
			display
			tex={'C_{\\mathrm{goal,stay}} = \\frac{1}{\\htmlClass{eq-knob}{K}d}\\sum_{k=\\htmlClass{eq-knob}{H}-\\htmlClass{eq-knob}{K}+1}^{\\htmlClass{eq-knob}{H}} \\left\\lVert\\htmlClass{eq-out}{\\hat z_{t+k}} - \\htmlClass{eq-model}{z_g}\\right\\rVert^2, \\qquad \\htmlClass{eq-knob}{K}=4.'}
		/>
		<p>
			The effort preference stays the same. A fast pass may now score worse than a movement that
			brakes near the destination. This cost encourages settling; the observed pose and time spent
			near it tell us whether settling actually happened.
		</p>
		<p>
			Rehearse once in the next plate, then change the destination or intention. The three predicted
			futures remain exactly the same: only their scores and ranking change. Step tests the
			preferred first action from that small gallery. Rehearse again, or use Run, to search for new
			actions under the new request. Rescoring existing predictions makes the separation between
			knowledge and preference visible without any new learning.
		</p>
	</Prose>

	<WorldPlate mode="transfer" {lab} />

	<Prose>
		<p>
			Now change damping. This alters the mechanism's response to an action while leaving the
			predictor's weights alone. A new goal changed the question; this changes the world that must
			answer it. Compare the first-step prediction with the observed motion. Replanning can use
			fresh pictures to compensate for some mistakes, but learning the altered dynamics would
			require new training experience.
		</p>
		<p>
			There are several kinds of reuse here. The held-out examples test prediction on experience
			that was not used for gradient updates. Changing goals tests whether the same learned dynamics
			can serve different requests. Changing mechanics tests a harder boundary: whether what was
			learned still describes the environment. Success at one does not settle the others. These
			local experiments make those distinctions observable rather than hiding them inside a single
			success score.
		</p>
		<p>
			The architecture is broader than this mechanism. Observations might be camera frames from a
			robot, actions its controls, and a goal an image of an arranged workspace. Applying the recipe
			there requires suitable experience, a representation that retains the needed information, and
			predictions reliable over the planning horizon. It does not mean these eight numbers or these
			trained weights already understand another world.
		</p>
		<p>
			World models also support other ways of choosing actions. Ha and Schmidhuber's
			<em>World Models</em> combined a learned visual representation and recurrent dynamics with a
			trained controller.<Cite id="ha-2018-worldmodels" /> Here, a fresh search makes the decision. The
			common resource is learned predictive knowledge; JEPA and planning at decision time are the particular
			choices this chapter has let us inspect.
		</p>
		<p>
			The representations we began with in <ChapterRef slug="space" /> have acquired another job. They
			let a machine compare actions by consequences that have not happened yet.
			<PlateRef id="history" /> supplied a past, <PlateRef id="train" /> learned from what followed, and
			<PlateRef id="plan" /> used that knowledge to consider a move. By the last plate we could change
			what we wanted while keeping what had been learned. Experience, prediction, preference, and feedback
			each had a distinct part to play.
		</p>
	</Prose>
</ChapterShell>
