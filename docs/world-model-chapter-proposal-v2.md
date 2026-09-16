# Before the Move — second proposal

Research and design review · September 16, 2026 · Proposed chapter, not an implemented or validated learning experiment.

**I would change the central world.** Use a small two-link kinetic instrument: two slender arms, visible bearings, a fine tip, and a fading trace of its motion on the book’s warm paper. It learns from pictures and recorded torques. Later, a reader shows it a desired pose and watches it consider movements before choosing one.

The main discovery becomes **learn how a world changes, then reuse that knowledge when what you want changes**. Reaching, arriving gently, and changing the destination are different requests made of the same frozen model. Braking remains a lovely moment, but no longer carries the entire chapter.

This supersedes the puck recommendation in [the first proposal](./world-model-chapter-proposal.md). That document retains the detailed code audit and integration inventory. The present proposal changes the world, visual direction, narrative emphasis, sensor sizing, and validation plan. No application code has changed.

## Why this is the stronger direction

The instrument has an attractive silhouette before it has learned anything. Its articulation gives motion character without requiring textures, elaborate scenery, or a complicated simulator. A full ghost pose makes the desired outcome legible. Momentum makes history and anticipation necessary. Two independent torque inputs make the control problem manageable while allowing curved, coordinated movement.

Use a fully actuated, damped planar mechanism moving on a horizontal plane. Start without contact, obstacles, or gravity. Prefer coupled two-link dynamics; if an independent-joint approximation proves necessary, describe it accurately. Do not introduce chaotic or underactuated pendulum behavior merely for visual drama. Links can overlap in projection; their geometry and joint marks must remain distinguishable in the actual sensor raster.

Call it a **kinetic instrument**, not a robot artist. A trace reveals movement. It does not promise handwriting, picture generation, or accurate ornamental drawing. A later sequence of explicitly supplied poses could leave a beautiful cumulative trace, but that is an optional extension after reliable single-goal control. Supplied waypoints would not constitute learned hierarchical planning.

This gives the research ideas a quiet, concrete expression. Experience supplies predictive knowledge; the goal arrives afterward; planning uses that knowledge; observation corrects the next decision. Appreciation for the work should be visible in those choices. The chapter needs neither a tribute nor speculation about what any particular researcher would enjoy.

## What the additional reading changes

LeCun’s 2022 position paper separates knowledge of the world from objectives and describes optimizing candidate actions through a predictive model. That suggests making a **change of intention with unchanged weights** the chapter’s decisive experiment. Our short-horizon toy represents a small part of that program; it does not implement its proposed memory or learned planning hierarchy. [A Path Towards Autonomous Machine Intelligence, §§2.1, 3.1.2, 3.2, 4.7](https://openreview.net/forum?id=BZ5a1r-kVsf)

I-JEPA and V-JEPA motivate learning in representation space, while illustrating that architecture and training tasks influence what representations preserve. The ability to avoid reconstructing every detail is not a guarantee of useful invariance. An attractive latent plot cannot substitute for testing control. [I-JEPA](https://arxiv.org/html/2301.08243v3), [V-JEPA, §2](https://arxiv.org/html/2404.08471v1)

V-JEPA 2 makes the observation–prediction–action loop particularly relevant, while using a substantially different pretraining and action-conditioning recipe. LeWM remains our methodological anchor: jointly learned visual representations and action-conditioned predictions. Its Reacher benchmark specifies the full arm configuration, reinforcing the choice of a whole ghost pose over a fingertip target. Its large, SAC-collected corpus does not establish that our small exploratory browser corpus will suffice. [V-JEPA 2, §§3–4](https://arxiv.org/html/2506.09985v1), [LeWM, Appendix E](https://arxiv.org/html/2603.19312v3)

The supplied research report led to two useful additional checks. **LeVJEPA**, submitted August 27, learns representations by aligning views of the same temporal window. Its causal attention restricts access to future observations; it does not provide our action-conditioned future predictor. Its pretraining efficiency results are not browser benchmarks. [LeVJEPA, §§3, 6](https://arxiv.org/html/2608.27395v1)

**When Does LeJEPA Learn a World Model?** gives conditional identifiability results, not a blanket guarantee for our instrument. The clean theorem assumes Gaussian latent variables and particular stationary transitions; its own real Reacher trajectories violate those assumptions. Periodic joint angles and torque-driven dynamics require empirical validation. Nor is its alignment objective identical to LeWM’s action-conditioned prediction objective. [Paper, §§3, 6.2, 7](https://arxiv.org/html/2605.26379v1)

The report is useful research context. Its proposals for several architecture modes, a different browser framework, and a multimillion-parameter model would expand this into a different project. Retain Svelte and jax-js, which the course already uses. Treat all proposed latency and memory figures as estimates until measured. Do not reuse its ambiguous instruction to normalize SIGReg inputs, or assume that ordinary microbatch gradient accumulation reproduces the full-batch distribution statistic.

## The visual language

Keep the book’s Newsreader prose, restrained Inter controls, numbered plates, and quiet paper bands. The instrument should feel native to those pages. Avoid a separate robotics dashboard.

- **Form:** two slender, slightly tapered links; small concentric bearings; a precise tip. Material comes from restrained contrast and geometry, without metallic gloss or decorative hardware.
- **Color:** neutral ink for the actual mechanism and motion trace; cool blue for learned forecasts; a warm dashed full-pose outline for the goal. Shape and line style reinforce color.
- **Composition:** one generous stage and an asymmetric open gesture. No permanent grids, rainbow embeddings, hovering metric cards, or clouds of candidate paths.
- **Rhythm:** the real arm holds still while a few possible futures appear. One plan is selected, motion resumes, and the next observation arrives. Deliberation has visible time.
- **Controls:** one primary action per plate. Goal and foresight matter to the reader; candidate counts and solver settings belong under the hood.

The accompanying motion study is explicitly scripted. It tests silhouette, composition, target legibility, and the distinction between physical motion and rehearsal. It contains no training, learned prediction, or performance claim. Its smooth interpolation is not the proposed controller. The final chapter must earn its motions through measured learned planning, without replacing them with splines or an invisible inverse-kinematics controller.

The motion trace, ghost goal, and forecast overlays stay outside the model’s observations. A small “what the model sees” view makes that boundary inspectable. This also avoids asking the model to predict accumulating ink.

## Six plates, one instrument

| Plate                                    | Experience                                                                                                                                                         | Discovery                                                                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| I — The missing moment                   | Two identical poses have opposite angular velocities. Reveal their preceding frames, apply the same torques, and replay. This is openly a simulator demonstration. | One picture need not specify the state. History carries information.                                                         |
| II — Learn the consequences              | Move the joints, collect varied exploratory episodes, inspect picture/action pairs, and train the model. There is no goal or reward in this collection task.       | Experience can train a predictive model before a particular task is chosen.                                                  |
| III — Let a prediction continue          | Freeze the model. Replay recorded actions through it and through the simulator. Compare short and longer forecasts using a separately fitted display readout.      | A one-step forecast with fresh observations differs from a rollout using its own predictions.                                |
| IV — When everything looks the same      | A compact matched comparison removes the distribution regularizer. Show actual representation spread and predictive usefulness alongside loss.                     | A low prediction loss can hide lost distinctions. Avoiding collapse is one requirement, not a certificate of useful physics. |
| V — Before the move                      | Supply a complete ghost pose. Hold the world still, reveal a few candidate futures and their costs, select an action, then observe and replan.                     | Prediction, preference, search, and feedback do different jobs.                                                              |
| VI — The same knowledge, a new intention | Change the desired pose while weights stay frozen. Compare matching a pose near the end with matching it over several final moments. Then change damping.          | New goals can reuse a model; new mechanics can invalidate it. Replanning and further learning are different responses.       |

The regularizer comparison remains visible but does not dominate the opening. Run both variants from matched ordinary random initializations and the same corpus. Show the measured result even if the unregularized run does not visibly collapse on schedule. Do not manufacture a dramatic all-zero initialization.

The last plate should first deliver a satisfying new-goal success, then show the boundary of that reuse. The ending can connect naturally to the earlier chapters: a representation need not reconstruct a picture to support a decision; search can use learned transitions; a probe reveals recoverable information rather than proving general understanding.

Possible opening prose:

> The two instruments occupy the same pose. Give both the same push. One opens; the other folds. What is missing from the picture?

Possible final idea:

> We changed the destination without changing the model. Changing the mechanism was a different matter.

These are provisional lines, not a finished chapter.

## The experiment we should actually build

**Observations and experience.** Begin with 32×32 grayscale images, comparing 48×48 if thin links or joint overlaps become ambiguous. Rendering quality and sensor information are separate requirements: the reader sees a crisp high-resolution mechanism; the network sees a small raster generated from the same scene. Choose frame spacing jointly with speed so displacement is visible without skipping important transitions.

Collect a few thousand to tens of thousands of local transitions, selecting the final budget by measurement. Vary initial angles, velocities, and torque sequences. Include sustained pushes, reversals, and release rather than only independent action noise. Keep complete episodes or generation seeds separate for training and evaluation. A reader’s short manual session can illustrate data collection; it should not be the sole source of coverage.

**Model.** Use a shared per-frame encoder, recent latent/action history, and a small action-conditioned predictor. Start with two or three observations and latent widths 4, 8, and 16. A 32×32 flattened MLP encoder `1024 → 64 → 8`, with predictor `20 → 64 → 64 → 8`, is approximately 72,000 parameters. That is a sizing example for two aligned latent/action slots, not the selected architecture. Compare with a compact spatial encoder if needed. Keep the final embedding unconstrained by unit normalization.

Make alignment explicit: `P(z[t−1], z[t], a[t−1], a[t]) → ẑ[t+1]`. The first action is the recorded torque between the context frames; the second is the proposed next torque. During rollout, shift both histories together and insert the predicted embedding.

The small MLP architecture, concatenated action conditioning, sensor resolution, and history length are explicit departures from the paper’s transformers. The correct description is **a LeWM-inspired educational model trained from scratch in the browser**.

**Learning.** Preserve next-latent prediction plus SIGReg, with gradients through both uses of the shared encoder. No pretrained teacher, EMA branch, target detachment, reconstruction loss, state labels, or reward labels train the world model. The distribution regularizer acts across independent examples at each sequence position, including target frames. Preserve the pinned implementation’s projection, quadrature, and reduction semantics when reducing its computational budget. Ordinary averaging of microbatch SIGReg losses changes the statistic. [Official training objective](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py), [official SIGReg](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py)

**Planning.** Freeze the learned weights. Encode a clean goal image, predict candidate torque sequences, and score their future embeddings against that goal. Use bounded cross-entropy-method search, initially measuring 64–128 candidates and three refinement rounds at a few fixed horizons. Execute one torque pair, acquire another observation, and replan. This small first-action MPC loop deliberately differs from the paper’s full solver budget and execution cadence. Batch candidate evaluation and keep the predictor cheap; planning repeats it many times.

The visible traces represent alternative action sequences. A deterministic predictor does not turn them into a calibrated uncertainty distribution.

A static goal image specifies pose, not velocity. A terminal-window cost can encourage arrival and remaining nearby:

\[
C(a_{t:t+H-1})=\frac{1}{K}\sum_{k=H-K+1}^{H}\left\|\hat z_{t+k}-E(o_g)\right\|^2.
\]

This is a proposed cost-design experiment, not a guarantee of settling. Measure physical pose error and dwell separately. Include both joints: one tip position can correspond to different elbow configurations. If an action-effort penalty or coarser action parameterization improves motion, make it an explicit planner choice; do not hide it in an extra training objective.

**Showing predictions.** The predictor produces embeddings. After freezing it, fit a small pose readout locally on separate simulator-labeled examples, with no gradients back into the encoder. Its only role is to draw diagnostic ghosts. It must never feed the planner or its cost. Evaluate it on true observations and on predicted latent rollouts. If angles are decoded and rendered with known rigid-link geometry, disclose that this display imposes valid geometry; it does not prove the embedding predicted a physically valid state.

**Abstraction, if earned.** An optional paired experiment changes appearance while preserving mechanics, then changes damping while preserving appearance. A nuisance must reach the actual sensor and vary independently of physical state. Measure its effect on forecasts and control; do not promise invariance. This belongs after the central experiment works, not as another mandatory subsystem.

## Evidence before committing to the chapter

After concept review, develop a visual prototype and a small learning prototype in parallel. The first establishes the actual rendered mechanism and physical motion; the second establishes that useful control is possible within a browser-sized budget. Integrate them before writing the polished narrative.

| Gate                  | Required evidence                                                                                                                                                                                                                                                                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Visual quality        | The real renderer looks composed when paused, during exploration, and during failed control. Goal pose, actual motion, and predictions are distinguishable on desktop, mobile, and both themes. Beauty cannot depend on a successful seed.                                                                                                          |
| Numerical correctness | Validate simulator timestep/substep convergence and near energy conservation with torque and damping disabled. The joint objective and gradients agree with small reference fixtures. SIGReg’s axes, scale, and random-direction behavior match the pinned source. No privileged simulator data enters learned planning.                            |
| Information           | Sensor frames distinguish relevant poses; history distinguishes velocity. Held-out full-pose probes and action-sensitivity checks establish useful information beyond noncollapse. Use circular angle errors where appropriate.                                                                                                                     |
| Dynamics              | Evaluate matched held-out action sequences at several rollout lengths. Separate representation, display-readout, and latent-dynamics errors. Include overlap, wraparound, and joint-limit cases where applicable.                                                                                                                                   |
| Planning              | Compare three diagnostic stages: known dynamics with physical pose cost; known dynamics with learned embedding cost; learned dynamics with learned embedding cost. This separates search limits, representation geometry, and prediction error. These diagnostic planners are not the public learned controller or guaranteed performance ceilings. |
| Reuse                 | Report new-goal reach and settle rates across fixed held-out starts and multiple training seeds. Set pose tolerance, dwell, and action budgets before selecting showcase runs. Show failures as well as successes.                                                                                                                                  |
| Browser cost          | Measure compile time, physical batch memory, training time to useful behavior, planning latency, and responsiveness on a named integrated-GPU device. Treat roughly one minute of active training as a design aspiration, not a claim. Make deliberation visible if continuous control is too expensive.                                            |

The arm still has only two positional degrees of freedom. It does not remove the possible mismatch between a small physical manifold and a Gaussian embedding target. Sweep latent width, observation stride, and data coverage before increasing model size. A new theory citation does not resolve that empirical issue.

If these gates fail, revise the world, model, or ambition openly. Do not preserve the headline by quietly introducing coordinate inputs, pretrained weights, simulator-based action search, or scripted successful motion. If reaching works but settling does not, review that result before changing the promised finale.

## Fit with the existing application

Keep a page-owned lab shared across the plates, one training/planning worker, fixed computational shapes, and the existing jax-js array-lifecycle discipline. The course already demonstrates local JIT training, pause/reset behavior, compact plots, and under-the-hood code. Use those conventions instead of adding a new framework or architecture selector. Train every learned component locally from random initialization, including the diagnostic readout.

The eventual integration still needs a `/world` route, world-model modules, plate and bibliography entries, code lessons, lab packaging, and revised transitions from flow and the epilogue. The first proposal records those locations. The new chapter should feel like the final step of the book’s existing argument, with its familiar interaction language and an unusually memorable physical object.

**Recommended decision:** proceed with the kinetic instrument and the frozen-model/new-intention story as the concept to validate. Approve the world and narrative first; select the final architecture, performance claims, and chapter prose only after the two prototypes meet the evidence gates.
