# Lab 11 · Before the Move

A two-link mechanism learns from small grayscale pictures and motor commands. Train an encoder
and an action-conditioned predictor together, freeze them, and search for a sequence of torques
that brings a predicted embedding close to a goal image. No pretrained weights or dataset are
downloaded: the browser generates exploratory episodes, and training starts from random weights.

This is a standalone lab from [jaxverse](https://github.com/NeoVand/jaxverse).

```sh
npm install
npm run dev
```

Open the printed localhost URL. The lab begins with 5,000 training steps; use **Train 5,000 more steps**
to continue. **Consider a move** freezes the weights, fits the diagnostic pose readout if needed,
and searches in latent space. **Apply first action** executes just the first motor command. Repeat
the two to close the feedback loop. Change the goal or the terminal-window cost without retraining.
Both intentions include the same effort preference: 0.01 times the mean squared normalized motor
command. The initial planning horizon is six observations, each separated by 0.24 simulated seconds.

Read the measured errors. A plausible-looking ghost can still predict the wrong pose. The
display readout is trained with separate simulator labels after freezing the encoder; it never
supplies a planner input or a cost. The only model inputs are images and actions. The displayed
joint error is a diagnostic, not the objective optimized by the planner.

`src/main.ts` contains the interface. `src/world/` contains the same model, SIGReg objective,
optimizer, simulator, sensor, corpus, worker, and planner used by the chapter. Try changing the
training seed, regularization weight, or training budget in `main.ts`. Compare predicted error
with the persistence baseline, and check how a longer planning horizon changes the result.
The included goals begin from one starting pose; success there does not establish reliable control
of more distant goals or altered mechanics. Distinguish passing near a goal from remaining there.
This small educational adaptation uses MLPs and a residual predictor; it does not reproduce
LeWorldModel's architecture or reported performance.

WebGPU is preferred; the worker reports the backend it can use. Compilation and training time
depend on your browser and device. Once dependencies are installed, the experiment needs no
network access.

When running from the jaxverse repository rather than a downloaded zip, run `npm run build:labs`
at the repository root first. That command refreshes the generated `src/world/` copy before
packaging, keeping the chapter and lab implementation in sync.
