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
			The two instruments occupy the same pose. Give both the same push. One opens; the other folds.
			What is missing from the picture?
		</p>
		<p>
			Their past. A joint can pass through an angle in either direction, and a still picture records
			neither. The first plate lets you uncover the preceding moments. This is the actual mechanism,
			before any learning: two links moving on a horizontal plane, driven at their joints, carrying
			momentum. Damping is switched off in this opening experiment. The same pose can have different
			futures because it can have different velocities.
		</p>
	</Prose>

	<WorldPlate mode="history" {lab} />

	<Prose>
		<p>
			In <ChapterRef slug="reward" />, a policy learned which action to take. Here we will first
			learn what an action does. A <em>world model</em> predicts how an environment changes, including
			the consequences of actions. Once we have one, a desired outcome becomes a question we can ask of
			it: which movements might bring the mechanism there?
		</p>
		<p>
			Learning the world and choosing a goal can happen at different times. That separation is a
			central idea in the architecture proposed by LeCun in 2022.<Cite id="lecun-2022-ami" /> The broader
			proposal includes memory and planning at several levels of abstraction. We will build one small
			piece of it, with a short history, a short future, and a world you can inspect completely.
		</p>
		<h2 class="h2">Learning before wanting</h2>
		<p>
			First, collect experience without a destination: observations, actions, and what followed.
			Here the actions are joint torques, or turning forces. Exploration supplies pushes, releases,
			and reversals; damping gradually slows motion. None of these episodes was collected to reach a
			goal. Later, the same knowledge will serve several requests.
		</p>
		<p>
			The model receives the small sensor pictures and motor commands. Joint angles and velocities
			belong to the simulator, not its training inputs. The crisp links, motion traces, and goal
			outlines are drawings for you. <PlateRef id="train" /> tests learning with pictures held out of
			training: can a predicted embedding identify the right next observation among similar alternatives?
		</p>
		<p>
			We need a way to represent each picture. The encoder in <ChapterRef slug="latent" /> was taught
			to preserve enough to reconstruct its input. This encoder is taught to make the next representation
			predictable from recent ones and the intervening actions. Its output is an
			<em>embedding</em>: a short vector whose coordinates are learned along with the predictor.
		</p>
		<Math
			display
			tex={'\\begin{aligned}\\htmlClass{eq-model}{z_t} &= \\htmlClass{eq-model}{E_\\theta}(\\htmlClass{eq-world}{o_t}), \\\\[6pt] \\htmlClass{eq-out}{\\hat z_{t+1}} &= \\htmlClass{eq-model-2}{P_\\phi}(\\htmlClass{eq-model}{z_{t-1}, z_t}, \\htmlClass{eq-world}{a_{t-1}, a_t}).\\end{aligned}'}
		/>
		<p>
			The colors follow the parts of the machine: <Math tex={'\\htmlClass{eq-world}{o, a}'} /> are pictures
			and actions supplied from outside; <Math tex={'\\htmlClass{eq-model}{E_\\theta}'} />
			and its embeddings are blue; <Math tex={'\\htmlClass{eq-model-2}{P_\\phi}'} /> is the violet predictor.
			Its answer, <Math tex={'\\htmlClass{eq-out}{\\hat z}'} />, is green.
		</p>
		<p>
			Read the subscripts carefully. The first action carried the mechanism between the two context
			pictures. The second is the action whose consequence we are predicting. The target comes from
			encoding the picture that followed. No person has to annotate it.
		</p>
		<p>
			This is a <em>joint embedding predictive architecture</em>, or JEPA: learn representations
			through a prediction made in representation space.<Cite id="assran-2023-ijepa" /> Our small, action-conditioned
			version follows LeWorldModel's joint training idea, with much smaller networks and images.<Cite
				id="maes-2026-lewm"
			/> It is a LeWM-inspired educational model, trained from random weights in this browser.
		</p>
	</Prose>

	<ArchitectureDiagram />

	<WorldPlate mode="train" {lab} />
	<UnderTheHood slug="world" block="learn" />

	<Prose>
		<p>
			Now hold the weights and starting observation fixed. Change only the proposed actions. This is
			the question a world model lets us ask before acting. Let its first prediction become part of
			the context for its second. Continue. The resulting
			<em>rollout</em> is a forecast that must live with its own mistakes. Compare it with the real mechanism
			replaying exactly those actions.
		</p>
		<p>
			The blue ghosts need an explanation. The predictor outputs embeddings, not drawings. After
			training, we fit a separate <em>readout</em> that translates an embedding into a pose for display.
			This readout uses simulator labels, with the encoder frozen. It cannot update the world model or
			choose its actions. Its drawings use rigid links, so a tidy silhouette is not proof of an accurate
			prediction. Look at the agreement with the actual replay.
		</p>
	</Prose>

	<WorldPlate mode="forecast" {lab} />

	<Prose>
		<p>
			A long rollout asks more of the model than a succession of one-step forecasts. With fresh
			observations, yesterday's error can be corrected by today's picture. In a rollout, the next
			input is something the model invented. An error changes the place from which the following
			prediction is made. Lengthening the future gives a planner more room to act, more room to be
			wrong, and a larger search to solve with the same number of candidates.
		</p>
		<h2 class="h2">A prediction of nothing</h2>
		<p>
			There is an even shorter route to low prediction error. Suppose the encoder returns the same
			vector for every picture. The predictor can return that vector too. It will be perfectly
			consistent, and unable to tell an open instrument from a folded one. This failure is called
			<em>representation collapse</em>.
		</p>
		<Math
			display
			tex={'\\begin{aligned}\\mathcal{L}_{\\mathrm{pred}} &= \\htmlClass{eq-op}{\\operatorname{mean}}\\!\\left[(\\htmlClass{eq-out}{\\hat z_{t+1}}-\\htmlClass{eq-model}{z_{t+1}})^2\\right], \\\\[6pt] \\mathcal{L} &= \\mathcal{L}_{\\mathrm{pred}} + \\htmlClass{eq-knob}{\\lambda}\\,\\htmlClass{eq-op}{\\mathrm{SIGReg}}(\\htmlClass{eq-model}{Z}).\\end{aligned}'}
		/>
		<p>
			The second term supplies a competing requirement. <em
				>Sketched isotropic Gaussian regularization</em
			>, SIGReg, compares projections of a batch of embeddings with a standard Gaussian
			distribution.<Cite id="balestriero-2025-lejepa" /> A constant vector fails that test. The network
			must preserve distinctions while making its representations predictable. Both uses of the encoder
			receive gradients; the future embedding is learned too. The amber weight <Math
				tex={'\\htmlClass{eq-knob}{\\lambda}'}
			/> sets how strongly this constraint competes with prediction.
		</p>
		<p>
			The Gaussian here describes embeddings across different observations. It is not the random
			cloud sampled around each observation in the variational autoencoder chapter. And a spread of
			vectors does not, on its own, tell us what information survived. Prediction, a held-out
			readout, and control provide different checks.
		</p>
	</Prose>

	<WorldPlate mode="collapse" {lab} />
	<UnderTheHood slug="world" block="spread" />

	<Prose>
		<p>
			Compare the runs using the same experience and training budget. The unregularized model may
			lose distinctions without reaching a perfectly constant output. Read loss beside spread and
			usefulness. Separately trained encoders can choose different scales, so a smaller raw latent
			error is not, by itself, a better world model.
		</p>
		<p>
			Why make a prediction in this space at all? Because the representation can preserve what helps
			prediction without reproducing every visible detail. That possibility motivates feature
			prediction in video models too.<Cite id="bardes-2024-vjepa" /> It is a possibility to test, not
			a guarantee that an encoder will ignore everything we consider irrelevant. Here, the practical question
			is whether the information it retains supports a new movement.
		</p>
		<h2 class="h2">Give the model a destination</h2>
		<p>
			Place the warm outline where you want the instrument to arrive. It specifies the whole pose:
			two different elbow positions can put the tip in the same place. Encode a clean picture of
			that pose with the same encoder, <Math
				tex={'\\htmlClass{eq-model}{z_g} = \\htmlClass{eq-model}{E_\\theta}(\\htmlClass{eq-world}{o_g})'}
			/>. Now a candidate sequence of torques can be scored by how close its final predicted
			embedding comes to the goal embedding.
		</p>
		<Math
			display
			tex={'C_{\\mathrm{goal}} = \\frac{1}{d}\\left\\lVert\\htmlClass{eq-out}{\\hat z_{t+H}} - \\htmlClass{eq-model}{z_g}\\right\\rVert^2.'}
		/>
		<Math
			display
			tex={'C = C_{\\mathrm{goal}} + \\frac{\\htmlClass{eq-knob}{\\rho}}{2\\htmlClass{eq-knob}{H}}\\sum_{k=0}^{\\htmlClass{eq-knob}{H}-1}\\lVert \\htmlClass{eq-world}{a_{t+k}}\\rVert^2.'}
		/>
		<p>
			Here, <Math tex="d" /> is the number of embedding coordinates and <Math
				tex={'\\htmlClass{eq-knob}{H}'}
			/>
			is the number of actions considered ahead. The second term prefers smaller motor commands: each
			action has two coordinates bounded between −1 and 1, and
			<Math tex={'\\htmlClass{eq-knob}{\\rho} = 0.01'} /> weights their mean square. This preference is
			supplied to the planner; it did not train the world model. The <em>cost</em> says which
			outcome we prefer. The world model predicts consequences. The
			<em>planner</em> searches for actions with a low predicted cost. Keep those three jobs apart: none
			of the weights changes while a movement is being considered.
		</p>
		<p>
			Our search samples torque sequences, keeps the better candidates, and samples again around
			them. This is the <em>cross-entropy method</em>. Only the first action is executed. Another
			picture then replaces the forecast of the present, and the search begins again. This loop is
			<em>model predictive control</em>; latent world models such as V-JEPA 2 use it for visual
			goals too.<Cite id="assran-2025-vjepa2" />
		</p>
	</Prose>

	<WorldPlate mode="plan" {lab} />
	<UnderTheHood slug="world" block="plan" />

	<Prose>
		<p>
			Notice the pause before a move. The actual instrument stands still while candidate futures are
			evaluated. Their traces are alternative actions under one learned model. They are not a
			probability distribution over everything that could happen. A candidate can receive a low cost
			because it is useful, or because the predictor is wrong about it. Replaying the chosen action
			is the test.
		</p>
		<p>
			Now ask for a slightly different thing. A picture specifies a pose, but not how fast the
			joints are moving through it. To encourage arrival and remaining nearby, score several final
			moments against the same goal instead of scoring only the last one:
		</p>
		<Math
			display
			tex={'C_{\\mathrm{goal,stay}} = \\frac{1}{\\htmlClass{eq-knob}{K}d}\\sum_{k=\\htmlClass{eq-knob}{H}-\\htmlClass{eq-knob}{K}+1}^{\\htmlClass{eq-knob}{H}} \\left\\lVert\\htmlClass{eq-out}{\\hat z_{t+k}} - \\htmlClass{eq-model}{z_g}\\right\\rVert^2.'}
		/>
		<p>
			We replace the goal term with this average over the last four predictions,
			<Math tex={'\\htmlClass{eq-knob}{K} = 4'} />; the effort term and its weight stay the same.
			Rehearse once in the next plate, then change the destination or intention. The displayed
			futures stay fixed; only their scores and ranking change. A movement that passes through the
			outline may now be worse than one that settles nearby. This small gallery isolates preference
			from prediction. Rehearse again to search for new actions under the new request.
		</p>
	</Prose>

	<WorldPlate mode="transfer" {lab} />

	<Prose>
		<p>
			A close pass and a settled arrival are different tests. These few destinations, from one
			starting pose, give us a small experiment in control. A different initialization, a more
			distant pose, or a longer search can expose failures. The observed movement, including what
			happens after reaching the outline, is the evidence.
		</p>
		<p>
			Changing the destination gives the planner a new question. Changing damping changes the answer
			the physical world gives to an action. Compare the frozen model's forecast with the new
			mechanism. Fresh observations can help the next plan recover from an error, but they do not
			rewrite the dynamics in its weights. Learning from new experience is a separate operation.
		</p>
		<p>
			A <em>world model</em> names a broad family, not one training recipe. Earlier systems learned
			a compressed visual representation, a recurrent dynamics model, and a controller trained to
			use them.<Cite id="ha-2018-worldmodels" /> This page instead compares actions at decision time,
			using a jointly trained representation and predictor. The small world makes the distinction visible;
			it does not establish how far either approach generalizes.
		</p>
		<p>
			We have arrived at another use for the representations this book began learning in
			<ChapterRef slug="space" />. They can support a decision about something that has not happened
			yet. <PlateRef id="history" /> needed a past to distinguish two futures.
			<PlateRef id="plan" /> needed a goal to choose between them. Experience trained the model; search
			chose actions; feedback supplied another observation. We changed what we wanted without having to
			start learning the world again.
		</p>
	</Prose>
</ChapterShell>
