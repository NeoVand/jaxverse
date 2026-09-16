# Browser world model: measured development results

The shipped model learns useful local control from pixels, but it is not a generally reliable controller. The default seed reaches all three local presets; failures across other seeds, distant targets, and longer open-loop rollouts remain visible. These are development measurements used to choose the chapter's defaults, not results on an untouched control benchmark.

Measurements were collected on 2026-09-16 on Apple M4 using Chromium, WebGPU/Metal, and the repository's jax-js runtime. The [raw historical runs](world-model-benchmark-results.json) include unsuccessful configurations and all recorded trials. Some early runs predate later diagnostics; absent measurements are not zeros.

## What actually trains

- Grayscale 32×32 sensor frames, fixed pixel scaling `(1 − pixel) × 4`, a 128-wide GELU encoder, and eight-dimensional unnormalized embeddings.
- Predictor: `[z[t−1], z[t], a[t−1], a[t]]`, two 128-wide GELU layers, and an eight-dimensional correction added to `z[t]`. There are 152,464 trainable parameters in total. All parameters begin from ordinary random initialization.
- Joint next-embedding MSE plus 0.01×SIGReg. Both source and target uses of the shared encoder receive gradients. The regularizer applies independently across the physical batch at each of the three sequence positions, including the target position. No state labels, pretrained weights, detached targets, pixel reconstruction, rewards, or display readout enter this objective.
- Physical batch 64; 32 fresh unit projection directions per update; 17 quadrature knots on [0,3]. Integration weights, batch multiplier, and reduction axes follow the [pinned LeWM implementation](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py). This preserves that numerical recipe while substantially reducing projection count and changing architecture and data scale.
- Adam, learning rate 0.001, betas (0.9, 0.99), global gradient norm cap 5; 5,000 updates.
- 256 exploratory episodes ×64 actions =16,384 training transitions; 12 separate validation episodes. Two-observation windows are sampled within episodes. The first transition of each episode cannot supply a previous frame/action and therefore is not sampled as a training target. The interval between observations is 0.24 simulated seconds.

The final configuration is larger than the first proposal's 64-wide network. The original shorter observation interval made individual action effects difficult to see at 32×32. Increasing it to 0.24 seconds improved action observability, while making a six-observation planning horizon span 1.44 seconds. The model, live control, and forecast reference simulation all use the same interval. This is a teaching implementation inspired by LeWM, not a reproduction of its architecture, training corpus, or results.

## Prediction and representation checks

Each row trains from scratch with a different seed. A seed changes initialization, exploratory corpus, and diagnostic pose samples. Prediction measurements use a fixed 64-example batch from that run's held-out episodes. They are a small deterministic diagnostic, not an average over the entire validation corpus.

| Seed | Prediction MSE | Persistence MSE | Shuffled current-action MSE | Duplicate-current-history MSE | Mean embedding std | Pose readout RMS (rad) |
| ---- | -------------: | --------------: | --------------------------: | ----------------------------: | -----------------: | ---------------------: |
| 17   |       0.005191 |        0.044600 |                    0.006985 |                      0.038433 |             0.9026 |                 0.1408 |
| 19   |       0.003889 |        0.041928 |                    0.005758 |                      0.034600 |             0.9069 |                 0.1739 |
| 37   |       0.003483 |        0.035553 |                    0.005109 |                      0.028592 |             0.9216 |                 0.1212 |

Persistence predicts the current embedding unchanged. The action intervention shuffles only the proposed next action; the prior action and both images remain correct. The history intervention replaces the previous image with the current image. These interventions support actual use of both history and proposed action in this diagnostic; they do not establish causal sufficiency or performance under every counterfactual action. In particular, the exploratory policy repeats its previous action in about 85.5% of sampled windows.

The diagnostic pose readout is fit _after_ representation learning and with the encoder frozen. It uses 1,024 independently sampled uniform poses, 256 fixed random nonlinear features plus the original latent coordinates, and ridge regression to joint sine/cosine values. Error is measured on 256 separate diagnostic poses. It neither supplies gradients to the world model nor scores any plan. Its drawn arms can be wrong even when a latent prediction is useful, and the second joint contributes more error than the first.

### Matched collapse comparison

Seed 17, the same 5,000 updates, identical initial parameters, identical sampled data order, and separate projection/data random streams:

| Objective        | Held-out prediction MSE | Mean embedding std | Pose readout RMS (rad) |
| ---------------- | ----------------------: | -----------------: | ---------------------: |
| MSE +0.01 SIGReg |                0.005191 |            0.90263 |                0.14080 |
| MSE only         |          0.000000003202 |         0.00003839 |                1.63141 |

The prediction-only model almost erases the differences between observations. Its tiny loss is not evidence of accurate physical prediction. This is one measured matched seed, not a claim that every unregularized run must collapse identically. The historical near-collapse effective-rank number in the raw file used an inappropriate denominator floor; current code computes the scale-invariant participation ratio directly. The comparison above does not use that historical rank value.

### Open-loop error

For seed 37, eight fixed held-out episode windows, recorded actions, and no new observations fed back during the rollout:

| Horizon | Physical time (s) | Endpoint latent MSE | Persistence MSE | Drawn pose RMS (rad) |
| ------- | ----------------: | ------------------: | --------------: | -------------------: |
| 1       |              0.24 |             0.00636 |         0.08008 |               0.0905 |
| 3       |              0.72 |             0.06138 |         0.55571 |               0.2112 |
| 6       |              1.44 |             0.21324 |         1.24473 |               0.5009 |
| 12      |              2.88 |             0.82704 |         1.47271 |               0.8544 |

Drawn pose error combines prediction error and readout error. These endpoint measurements use a different sample from the one-step table. They demonstrate why repeated observation and replanning matter, and why a longer imagined future can become less useful. The raw run is labeled `seed37-effort003`; the effort coefficient only changes planning, so it does not affect these frozen-model predictions.

## Learned control

The planner receives image embeddings and action history, not physical state. It evaluates 64 bounded action sequences per CEM round, makes three rounds, uses two-observation action knots, and executes only the first action before observing again. It returns the best sequence actually evaluated rather than an unevaluated elite mean.

Both intentions use this cost:

`mean((predicted_embedding − goal_embedding)² over scoring window and latent coordinates) + 0.01 × mean(action² over the whole horizon and both motors)`.

Visit scores the final forecast. Hold scores the final `min(4, horizon)` forecasts. Neither score contains velocity, joint angles, readout error, or a true simulator rollout. The action term discourages unnecessary torque; it does not guarantee rest. The planning horizon is six observations for both modes.

All reported local episodes start from `q1=−1.35, q2=1.65, v1=v2=0`. Goals are Unfold `(−0.9,1.0)`, Return `(−1.8,1.8)` (called `turn` in the raw runs), and Curl `(−1.05,2.25)`. A separate distant Curl `(−0.7,2.65)` remains a stress test. Each episode runs 40 physical actions, or 9.6 simulated seconds, with CEM seed `300 + action_index`.

Reaching means circular joint RMS error ≤0.15 rad at some observation. Finite dwell means five consecutive qualifying observations. “Final five” requires all five final observations to qualify. The threshold was retained throughout tuning. None of these criteria asserts zero velocity or indefinite stability.

| Training seed | Visit: local goals reached | Hold: local five-frame dwell | Hold: local final five | Distant Curl hold: dwell / final five |
| ------------- | -------------------------- | ---------------------------- | ---------------------- | ------------------------------------- |
| 17            | 3/3                        | 3/3                          | 3/3                    | no / no                               |
| 19            | 2/3                        | 3/3                          | 2/3                    | yes / yes                             |
| 37            | 2/3                        | 3/3                          | 3/3                    | yes / no                              |
| Total         | 7/9                        | 9/9                          | 8/9                    | 2/3 / 1/3                             |

The seed-19 hold failure is instructive: it spends five frames near Curl and later wanders, ending 0.787 rad away. Seed 37's distant Curl likewise qualifies briefly but ends 0.714 rad away. Reporting only “dwell success” would conceal both failures. Default seed 17's hold final errors are 0.0285, 0.0724, and 0.0582 rad; its visit to Unfold ends at 0.1856 rad after earlier reaching 0.0577 rad.

These three seeds and goals informed development choices. Curl was made gentler after the larger move failed, and the larger goal was retained in the stress suite. A final bounded comparison used effort coefficients 0, 0.003, and 0.01. At 0.003, local visit success remained 7/9 while local finite dwell dropped to 8/9; 0.01 was selected uniformly for both intentions. This is tuning evidence, not an unbiased estimate of a population success rate.

Earlier seed-17 horizon-12 hold runs with zero effort reached finite dwell on only one of three original goals and ended outside tolerance on all three. This changes both horizon and effort relative to the final defaults, so it cannot isolate a horizon effect. The independent [known-dynamics oracle](world-oracle-benchmark.md) uses privileged physical cost and the earlier 0.12-second interval; it establishes search feasibility for its own setting, not a matched upper bound for the learned controller.

## Runtime, checks, and reproduction

Observed 5,000-update training loops took approximately 24–29 seconds on this machine. This excludes corpus generation and fitting the display readout; timings varied with compilation and concurrent development activity. Per-episode median six-step planning latency was roughly 5.4–6.9 ms in the selected runs. These are local observations, not device-independent promises. Peak browser/GPU memory and fallback Wasm training throughput were not measured.

The numerical tests compare SIGReg against an independently written JavaScript quadrature calculation and compare full shared-encoder autodiff gradients with finite differences. They are not a cross-framework PyTorch parity test. Planner tests check goal-window semantics and that returned bounded actions were actually evaluated. Engine tests check matched regularizer experiments and seed-reset corpus consistency. The final core run passed six tests, targeted ESLint, and TypeScript/Svelte checking.

Start Vite, then run the browser harness in another terminal. It imports the same production files used by the chapter:

```sh
npm run dev -- --host 127.0.0.1 --port 5176
node tools/world-bench-browser.mjs '{"steps":5000,"seed":17,"suite":true,"holdHorizon":6,"gentleCurl":true,"effort":0.01}' /tmp/world-seed17.json
node tools/world-bench-browser.mjs '{"steps":5000,"seed":17,"regularization":0,"control":false}' /tmp/world-collapse.json
```

Set `WORLD_BENCH_ORIGIN` if using another port. Playwright Chromium must already be installed. Repeat the first command for seeds 19 and 37 to reproduce the reported local suite. Keep explicit effort values when reproducing historical zero-effort runs, since the shipped default is now 0.01. Browser/compiler/device differences can change floating-point results and CEM choices.
