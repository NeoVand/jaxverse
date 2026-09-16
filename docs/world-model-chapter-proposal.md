# Before the Move — proposed final chapter

Research and design proposal · September 16, 2026 · Implementation has not begun.

**Recommendation.** Chapter 11 should follow one small physical world from experience to prediction to planning. My preferred world is a puck on a slippery tabletop. The reader trains an action-conditioned model from rendered pictures, then watches a planner use it to bring the puck into a dock. The decisive moment is a push _away_ from the dock: braking now to arrive later.

Working title: **Before the Move**. Kicker: **World models**. Proposed deck: “Train a model of a little world from pictures and actions. Then give it a destination, and let it plan a few steps ahead.” Aim for roughly 20 minutes of reading and interaction, subject to measured training time.

**Research basis.** I read the main text and relevant appendices of [LeWorldModel v3, June 3, 2026](https://arxiv.org/html/2603.19312v3), and audited the authors’ implementation at commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. The research also covered SIGReg’s basis in [LeJEPA](https://arxiv.org/html/2511.08544v3) and the older [World Models](https://worldmodels.github.io/) approach. Application study covered the authoring guide, six chapter narratives—latent, language, reward, taste, Rook, flow—shared UI, training workers, chapter registries, code lessons, and downloadable labs. I inspected latent, reward, and flow in the running app, exercised live autoencoder training, and observed flow inference. This is research and design evidence, not a benchmark of the proposed model.

**What matters from LeWM.** The paper learns an encoder and an action-conditioned temporal predictor from pixel/action trajectories. Prediction happens in representation space. A Gaussian-distribution regularizer counters collapsed representations. Control uses candidate latent rollouts and a goal image. The reported system is approximately 15 million parameters, evaluated on specific control environments; its reported speedup is a benchmark comparison, not a browser expectation. Its weaker Two-Room planning is particularly relevant to our tiny world: the suggested low-dimensional prior mismatch is a hypothesis, and strong state probes also point toward possible dynamics/planning problems. We should call our substantially smaller architecture a _LeWM-inspired educational model_, with departures listed explicitly. [Paper, methods and evaluation appendices](https://arxiv.org/html/2603.19312v3)

LeJEPA supplies the distribution-matching machinery. Its theory has assumptions about representations and downstream tasks; it does not establish that our toy will learn physics. Its Gaussian target describes a distribution of deterministic embeddings across observations, not a random cloud sampled for each observation as in the book’s VAE chapter. Finite random projections encourage that distribution; they do not certify it. [LeJEPA, sections 3–5](https://arxiv.org/html/2511.08544v3)

JEPA means _Joint Embedding Predictive Architecture_. SIGReg means _Sketched Isotropic Gaussian Regularization_. The pinned LeWM configuration makes the scale and our proposed simplifications concrete:

| Component          | Official configuration                                  | Proposed browser starting point            |
| ------------------ | ------------------------------------------------------- | ------------------------------------------ |
| Observation        | 224×224 image                                           | 24×24 grayscale image                      |
| Encoder            | ViT-Tiny, patch size 14, no pretrained weights          | Small MLP or spatial convolutional encoder |
| Embedding          | 192 dimensions                                          | Sweep 2–32 dimensions                      |
| Temporal predictor | Six transformer layers, 16 attention heads, 0.1 dropout | Small history-conditioned MLP              |
| Action input       | Learned action embedding and adaptive normalization     | Concatenated action history                |
| Projection         | Hidden BatchNorm/GELU between linear layers             | Unrestricted final linear output           |

These are architecture changes, not merely a smaller training batch. The source default also uses λ=0.09 and 100 maximum epochs; those configuration defaults should not be confused with every reported experiment. [Model configuration](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/model/lewm.yaml), [training configuration](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/lewm.yaml)

The historical framing should be brief. A world model is a broader idea than JEPA. Ha and Schmidhuber’s system used a VAE, recurrent probabilistic dynamics, and a separately trained controller; our proposal uses online action search with frozen learned dynamics. Neither architecture is the definition of the entire field. [World Models, agent architecture](https://worldmodels.github.io/)

**Why this belongs at the end of this book.** Three earlier chapters give it a natural entrance:

| Existing chapter                                          | Connection to make                                                                                                                           |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [The Hidden Map](../src/routes/latent/+page.svelte)       | Revisit what the learning objective asks a representation to preserve. Reconstruction and predictive usefulness ask different questions.     |
| [Learning from Reward](../src/routes/reward/+page.svelte) | Its REINFORCE learner acts without a learned transition model. Here we explicitly learn one, then use it to compare possible actions.        |
| [Rook](../src/routes/rook/+page.svelte)                   | Preserve its distinction between recoverable information and proof of understanding. Separate representation, dynamics, and decision-making. |

The house style is one experiment unfolding across plates, controlled comparisons, consequences the reader can see, and failures that change the explanation. Keep the serif prose, numbered full-width plates, compact controls, semantic equation colors, and quiet “Under the hood” lessons. Use the current components and registries: some examples in `AUTHORING.md` lag the implementation, including literal plate numbering and learning-rate notation. Follow the current book’s `η` for learning rate.

**The world.** A circular puck moves on a bounded tabletop under two-dimensional thrust, momentum, drag, and wall rebounds. Begin with outer walls only. A maze adds long-horizon search and collision complexity before the central lesson works. A pendulum would echo the reward chapter, but makes learning and interpreting the representation harder; a pushing task is a possible later extension.

The reader can steer manually, then ask the page to collect exploratory trajectories. Collection uses varied starts and action sequences, including boundary encounters. The corpus is frozen before each training experiment. This separates collecting experience from learning from it; an explicit later action can collect a new corpus after changing the physics.

The network receives low-resolution frames and recorded actions. Position, velocity, simulator equations, rewards, and goal markers are excluded from its training inputs. Simulator coordinates are allowed in evaluation and in a clearly separated display readout. Keep the goal overlay outside the observation raster, and show a small “what the model sees” inset so this boundary is inspectable.

History is essential: two pucks at the same location can be moving in opposite directions. Choose rendering resolution and timestep together so motion is actually visible in successive frames. A temporal predictor cannot recover information that the images have erased.

**Proposed six-plate reading sequence.**

| Plate                                   | Reader experience                                                                                                                                                                                           | What it establishes                                                                                                          |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| I — The missing moment                  | Two identical still pictures; predict what happens after the same push. Reveal the preceding frames and replay. This is explicitly the simulator, before learning.                                          | An observation is not necessarily a complete state. Recent history changes what can be predicted.                            |
| II — Experience becomes a model         | Steer briefly, collect a corpus, inspect frame/action pairs, and press Train. A compact diagram connects the real arrays to encoder, predictor, and next-frame target.                                      | Self-supervision from consequences; actions are recorded information even though no rewards or state labels train the model. |
| III — A perfect prediction of nothing   | Compare matched runs with and without the regularizer. Show prediction error beside representation spread and usefulness. Let the reader inspect random one-dimensional projections of the embedding cloud. | Low loss can be a shortcut. Anti-collapse is necessary to investigate, but is not sufficient evidence of useful dynamics.    |
| IV — Let the prediction continue        | Freeze a checkpoint, fit the display readout locally, and compare a predicted trajectory with the actual replay of identical actions. Extend the forecast horizon.                                          | One-step prediction with fresh observations differs from a rollout that feeds on its own predictions.                        |
| V — Before the move                     | Place the dock. Reveal sampled plans, the better candidates, and the selected first action. Execute it, observe again, and replan. Try a shorter horizon.                                                   | A model predicts consequences; a cost expresses the goal; a planner selects actions; feedback supplies new evidence.         |
| VI — A new destination, a changed world | Move the dock with learning paused. Then change drag while keeping the weights frozen. Compare forecasts and actual outcomes; optionally collect new experience and retrain.                                | A new goal can reuse a model. Changed dynamics can invalidate it. Replanning and learning are different repairs.             |

Each plate earns one concept. Collection and training share one apparatus. The collapse comparison is compact, with detailed controls under the hood. No additional full-page latent atlas is needed. The primary planning control is foresight; numerical solver settings belong in the code lesson.

The intended delight comes from visible consequences: opposite futures hidden in identical pictures, a low-error model that has thrown distinctions away, candidate paths narrowing into a decision, and braking before arrival. These are proposed observations to validate, not animations to script as successful learning.

**The model to prototype.** Start with 24×24 grayscale images, two or three observation frames, and the correspondingly aligned action history. Benchmark a small flattened-image MLP against a compact convolutional encoder. Preserve spatial location rather than globally averaging it away. Use a small MLP over the latent/action history for the first predictor. This explicitly replaces the research model’s transformers and action-conditioning blocks. An unrestricted final linear output avoids imposing a unit-length constraint on embeddings. [Official model and rollout wiring](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py)

An illustrative configuration is `576 → 64 → 8` for the encoder and `(2×8 + 2×2) → 64 → 64 → 8` for the predictor: approximately 43,500 parameters. It is a sizing example, not a selected architecture. Test latent widths 2, 4, 8, 16, and 32; a wider embedding is not automatically better for this world. Start with batch sizes 64/128 and a few thousand to tens of thousands of locally generated transitions. Select the final corpus and model from measurements.

Preserve the joint objective:

\[
z_t=E_\theta(o_t),\qquad
\hat z_{t+1}=P_\phi(z_{t-h+1:t},a_{t-h+1:t})
\]

\[
\mathcal L=\operatorname{mean}_{b,t,j}(\hat z_{t+1,j}-z_{t+1,j})^2
+\lambda\,\mathrm{SIGReg}(Z).
\]

Use teacher-forced next-step targets over short sequences. Both occurrences of the shared encoder receive gradients; there is no detached target branch, EMA encoder, pretrained visual backbone, or reconstruction term in the world-model objective. Regularize all encoded frames, including targets, independently across the batch at each time position. Free-running rollouts belong in evaluation and planning initially; adding a rollout training term later would be an explicit experiment. [Official training code](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py#L16-L41)

Implement actual SIGReg rather than a variance-only substitute. The pinned code uses fresh unit directions, mean sine/cosine statistics, 17 integration knots on `[0,3]`, Gaussian weighting, and a batch-size multiplier. Start with 32/64 directions against a larger reference count. Preserve reductions when changing the budget; otherwise the same λ means something different. Compare ordinary random initializations: exact zero collapse is stationary for this smooth statistic, so “inflate an all-zero model” would be a misleading demonstration. [Official SIGReg implementation](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py#L10-L35)

The experiment should expose separately the prediction term, regularizer, and held-out spread. Plot projected embeddings on fixed axes within a comparison; automatic rescaling can hide collapse. Do not rank separately trained encoders solely by their raw latent MSE, since their representations and scales can differ.

**Planning a dock honestly.** A single goal image specifies where the puck should appear; it cannot by itself require zero velocity. Make the actual task visible: “Enter this circle and remain inside for the next few moments.” Encode a clean goal image and penalize several final predicted frames against it:

\[
C(a_{t:t+H-1})=\frac1K\sum_{k=H-K+1}^{H}
\|\hat z_{t+k}-E(o_g)\|_2^2.
\]

This terminal-window cost is our proposed toy extension. It encourages settling without supplying velocity to the planner. Independently measure actual docking by distance and a specified dwell duration; the embedding cost does not guarantee the physical success criterion.

Start CEM measurements at 64–128 candidate action sequences, three refinement rounds, and a small set of fixed horizons. The planner samples actions, predicts their consequences, keeps elites, and refines the action distribution. Freeze model weights throughout. Execute one action, then replace imagined history with fresh observations and plan again. These choices deliberately reduce the [official solver’s budget](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/solver/cem.yaml); their adequacy is unmeasured.

Show one real puck and a restrained set of candidate traces. Distinguish candidate alternatives from uncertainty: a deterministic predictor’s plan bundle is not a calibrated distribution of possible futures. If planning needs deliberation time, visibly pause the simulated clock instead of implying continuous real-time control.

**How we can draw a prediction without cheating.** The world model outputs embeddings, not images. After freezing it, train a small position readout in the browser using separate simulator-labeled samples. Draw its predictions as labeled diagnostic ghost positions and report its held-out error. A few actual observation thumbnails keep the comparison anchored to what happened. The readout receives no gradients back into the world model and never supplies a planner input or score.

This creates three explicit software boundaries: simulator for actual observations, learned model for candidate futures, and readout for display. Test that the planner cannot import the simulator transition function or coordinate readout. If the readout is inaccurate, its ghost trace must not be presented as an exact picture of the latent prediction. The latent-error comparison remains available alongside it.

A pixel decoder would introduce another learning problem and obscure the main lesson. Leave it out of the first version. All learned components in this proposal start locally from random initialization, including the diagnostic readout; the chapter has no required shipped checkpoint or remote inference path.

**Scientific wording to maintain.** Say “predicts consequences in this environment,” “information recoverable from the embedding,” and “goal-conditioned planning.” Avoid assertions that it understands physics in general, learns universal causal laws, ignores all irrelevant detail, or guarantees noncollapse. An action-conditioned model can still rely on correlations and fail outside its experience. Its deterministic predictor can average ambiguous futures.

A teleport or appearance perturbation can be an optional surprise experiment: compare matched trajectories under a frozen model and measure next-embedding error. Call the result prediction error, not consciousness or calibrated uncertainty. Use appearance changes that actually reach the model’s input. Never make “recoloring does not matter” a promised conclusion without measuring it. Keep the main ending about changed drag, because it distinguishes altered dynamics from an isolated state jump.

**Feasibility gate before writing the final chapter.** Existing jax-js supports the needed operators, and a small CPU primitive check produced finite gradients through sine/cosine statistics. Existing workers already provide JIT training and local GPU inference. That does not establish full-objective correctness, learning quality, or browser speed. [jax-js upstream](https://github.com/ekzhang/jax-js)

The first implementation milestone, after approval, should be a disposable local research lab with numerical fixtures and a reproducible results table:

| Question                        | Evidence required                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Is the objective faithful?      | Compare MSE, SIGReg values, and gradients against pinned reference fixtures with identical projection directions. Include the target encoder gradient path.                                                                                                                                                                                         |
| Does it learn dynamics?         | Hold out whole episodes and seeds; compare one-step and longer forecasts against persistence. Test altered/removed actions and reduced history.                                                                                                                                                                                                     |
| Does it retain distinctions?    | Held-out spread and effective rank alongside loss, plus a separately evaluated position readout. Compare regularized and unregularized runs fairly.                                                                                                                                                                                                 |
| Can it control?                 | Reaching and docking success on a fixed unseen goal/start set, across at least three training seeds. Compare random actions and an identically budgeted planner using known dynamics as a diagnostic reference. Disclose its privileged state access and use the same physical success criterion; finite-budget search is not a guaranteed ceiling. |
| Does the model help?            | The same planning setup with a no-action or shuffled-action model, plus the no-SIGReg comparison. Low training error alone does not pass.                                                                                                                                                                                                           |
| Does it fit a reader’s session? | Measure first compilation, time to useful forecasts/control, planning latency, memory, and total warmup on named devices. A useful result within a few minutes is a design target, not a current claim.                                                                                                                                             |
| Is the page robust?             | Pause/reset, rapid controls, leaving/revisiting the chapter, device loss, keyboard use, touch, reduced motion, and a modest integrated GPU.                                                                                                                                                                                                         |

Fix the evaluation split and report failures as well as successful seeds. Set distance/dwell criteria and time budgets before choosing the final run. Include training/data budgets in every comparison. Begin by checking that the oracle can solve the intended docking task within the proposed horizon; otherwise learning is not the only difficulty.

If the tiny pixel model fails these gates, revise the environment, latent width, data coverage, or model size. Do not silently substitute coordinate inputs, a pretrained encoder, simulator-based planning, or an extra training objective and retain the original claim. If reaching is reliable but docking is not, bring that measured tradeoff back before changing the centerpiece.

**Integration after the experiment succeeds.** Reuse the serialization, disposal, and scheduling patterns in [the diffusion lab](../src/lib/diffusion/lab.svelte.ts), while giving this chapter a page-owned lab shared by its plates. Keep training and planning jobs serialized on one worker. Prewarm only a few fixed horizon shapes; use two-dimensional dense operations where the existing runtime warns about large backward intermediates. Keep GPU arrays out of reactive state.

Proposed new modules live under `src/lib/world/`: simulator, renderer/data collection, model/objective, runtime, planner, worker/RPC wrapper, and shared lab. The chapter route would be `/world`, with components under `src/lib/components/demos/world/`. It needs chapter/plate/glyph/bibliography registries, three code lessons (joint training, SIGReg, latent MPC), claim checks, and `labs/world` packaging. The local model-saving feature is a separate scope choice: existing persistence tracks progress, not a complete resumable training state.

Update the narrative transitions as well as navigation: flow currently declares the book finished, Rook says two chapters remain, and the epilogue describes Fashion-MNIST as the last two chapters. Update the authoring chapter count and README. Validate the finished work with the required Svelte tooling, type/lint checks, focused numerical and learning tests, static build, paper/label checks, and browser review of the actual final path.

The proposed order is **approve the concept → validate the browser experiment → settle architecture and measured claims → write and build the six plates → scientific and visual review**. The next decision is whether the inertial puck and docking payoff are the right final experiment for this book.
