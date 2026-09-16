# World model validation and architecture experiments

Follow-up measurements from 2026-09-16, using the actual browser core on Apple M4 and Chromium WebGPU/Metal. The shipped model starts from random weights and learns useful local control. It is not a reliable controller for every seed, target, or rollout length.

The [accompanying raw results](world-model-validation-results.json) retain every run in this comparison. Seed 17 is the chapter's development seed; 41 and 73 were selected before this experiment as fresh checks. These three seeds are a small diagnostic suite, not a population success estimate. The goals and thresholds were kept unchanged.

## Method

Each trained run receives 5,000 updates, the same 16,384-transition corpus size, physical batch 64, learning rate 0.001, eight-dimensional embeddings, 0.01 SIGReg weight, and the existing optimizer. Paired seeds preserve simulated trajectories and sampled windows across resolutions. Different input dimensions necessarily change the initial network. The transformer comparison preserves the encoder's exact initial weights and changes only the predictor.

The planner uses six observations, 64 candidates, three CEM rounds, and the same 0.01 action-effort term. Each control trial executes 40 actions (9.6 simulated seconds). Three local goals are tested once with the final-pose objective and once with the hold objective. A seventh trial uses the more distant Curl stress target. A separate zero-update run uses the same planner with untrained weights.

- **Reach:** any observation has circular joint RMS error ≤0.15 rad.
- **Dwell:** five consecutive observations meet that threshold.
- **Final five:** all five final observations meet it. This catches models that arrive briefly and then leave.

The UI stops a run after finite dwell; this suite continues to measure whether the mechanism stays nearby. Simulator state is used to score these diagnostics, never to choose a torque. The separately fitted pose readout draws predictions but does not score plans.

## Results

| Model                         | Parameters | Median training | Median plan | Reach | Hold dwell | Hold final five |
| ----------------------------- | ---------: | --------------: | ----------: | ----: | ---------: | --------------: |
| MLP · 32×32                   |    152,464 |          23.0 s |      5.4 ms |   8/9 |        8/9 |             7/9 |
| MLP · 64×64                   |    545,680 |          28.8 s |      5.9 ms |   7/9 |        6/9 |             3/9 |
| Transformer predictor · 32×32 |    162,768 |         144.6 s |     52.1 ms |   7/9 |        9/9 |             9/9 |

All counts combine three local goals and three seeds. Training times cover the 5,000-update core loop on this machine. Planning times are the median of the per-trial median latencies; cold first-plan compilation is not represented by that number. These are local timing observations, not universal performance guarantees.

The untrained 32×32 control reaches **0/3** local goals in the visit trials and satisfies the final-five hold criterion in **0/3** hold trials. It does briefly satisfy dwell in one hold trial, showing why a momentary success alone is weak evidence.

| Model                         | Seed | Prediction / persistence | Shuffled-action / prediction | No-history / prediction | Readout RMS (rad) | Reach | Hold final five |
| ----------------------------- | ---: | -----------------------: | ---------------------------: | ----------------------: | ----------------: | ----: | --------------: |
| MLP · 32×32                   |   17 |                    0.116 |                         1.35 |                    7.40 |             0.141 |   3/3 |             3/3 |
| MLP · 32×32                   |   41 |                    0.106 |                         1.41 |                    7.72 |             0.129 |   2/3 |             2/3 |
| MLP · 32×32                   |   73 |                    0.119 |                         1.38 |                    6.74 |             0.123 |   3/3 |             2/3 |
| MLP · 64×64                   |   17 |                    0.100 |                         1.33 |                    9.03 |             0.325 |   2/3 |             0/3 |
| MLP · 64×64                   |   41 |                    0.094 |                         1.52 |                    9.23 |             0.243 |   2/3 |             2/3 |
| MLP · 64×64                   |   73 |                    0.139 |                         1.20 |                    5.92 |             0.263 |   3/3 |             1/3 |
| Transformer predictor · 32×32 |   17 |                    0.138 |                         1.36 |                    6.30 |             0.168 |   3/3 |             3/3 |
| Transformer predictor · 32×32 |   41 |                    0.163 |                         1.67 |                    5.04 |             0.108 |   2/3 |             3/3 |
| Transformer predictor · 32×32 |   73 |                    0.160 |                         1.59 |                    5.41 |             0.180 |   2/3 |             3/3 |

A lower prediction/persistence ratio is better within a run. Ratios above one for the action/history interventions show that disturbing those inputs harms prediction. The diagnostic readout error and physical control outcomes are separate checks; prediction MSE alone is insufficient.

## What the tests establish

For the 32×32 MLP, held-out prediction error is about one ninth of the copy-current-embedding baseline. Shuffling the proposed next action or removing the previous observation makes predictions worse. Embedding spread remains substantial; the tiny loss is not obtained by mapping every picture to the same vector. The zero-update control separates the learned controller from the behavior of the search routine alone.

Long open-loop rollouts degrade. At twelve observations, drawn pose errors across the three 32×32 runs range from approximately 0.63 to 1.03 rad; those errors include the display readout's error. Distant goals and sustained holding also fail in some runs. The chapter should describe repeated observation and replanning, rather than imply that an accurate long-term simulator has been learned.

The prediction checks use one fixed 64-example validation batch per seed; rollout checks use eight held-out episode windows. These remain small diagnostics. Raw errors from different learned latent spaces should not be compared in isolation; the persistence ratio, physical readout, and actual control outcomes provide additional evidence.

## Doubling resolution versus display size

The core now accepts a `resolution` option, propagated through corpus generation, reset, comparison, and inference. Both 32×32 and 64×64 train and plan in the browser. The benchmark derives all frame sizes from the model configuration.

Doubling each image dimension quadruples pixel storage. With the current dense encoder, parameters increase from 152,464 to 545,680. The stored Float32 training and validation frames increase from 68.05 to 272.19 MiB; this is calculated frame storage, not measured peak memory. A training pixel batch grows from 0.75 to 3 MiB. Browser/GPU peak memory and Wasm training throughput were not measured.

At the unchanged training budget, 64×64 does not improve control reliability. It may benefit from a different encoder or separate tuning; this experiment does not establish that higher resolution is intrinsically worse. Keep 32×32 as the chapter default for now.

The visible sensor preview is now 128×128 CSS pixels, twice its previous width and height. It displays the exact model input with nearest-neighbor scaling and labels its actual resolution. Enlarging this view does not invent extra visual information or increase training cost.

## Transformer experiment

A transformer is a legitimate choice here. LeWM uses a vision transformer encoder and a causal transformer predictor with action conditioning; its encoder ablation also studies a ResNet. The learned predictive objective does not require our toy implementation to use an MLP. [LeWM §3.1 and Appendix G](https://arxiv.org/html/2603.19312v3), [pinned configuration](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/model/lewm.yaml).

The isolated research variant keeps the current image encoder, two-frame history, aligned actions, residual prediction, target-encoder gradients, and SIGReg objective. Its dynamics predictor has two causal attention blocks, two heads, width 32, FFN width 64, learned positions, and action-conditioned AdaLN with initially closed residual gates. Its eight-coordinate output remains unconstrained. Dropout is off. These are controlled toy choices, not a replication of the paper's network. The [official conditional predictor](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py) informed the design.

The variant has 162,768 parameters: the same 132,232-parameter encoder plus a 30,536-parameter predictor, versus 20,232 in the current MLP predictor. It is kept under `tools/` and is not imported by the chapter. A full vision-transformer encoder is a separate experiment. Doubling resolution does not enlarge the temporal predictor's two-token attention matrix, but a vision transformer with fixed patch size would receive four times as many image tokens.

CPU checks verify exact encoder-initialization matching, parameter count, finite full-objective gradients, opening of the AdaLN gates, action sensitivity after learning, and eager/JIT agreement. Actual WebGPU runs then test learning and control. This compares complete predictor designs, not attention alone: AdaLN, normalization, initialization, and predictor parameter count also differ. Updates are matched, not compute time. Neither architecture has received separate hyperparameter tuning in this comparison; a performance difference is not a general verdict on MLPs versus transformers.

## Bugs fixed during validation

- Pausing at an arbitrary training step could prevent held-out diagnostics from refreshing throughout the next training session. Evaluation now depends on updates since the last measurement, and always measures the final weights. The regression pauses at step 43 and resumes to 5,043; it fails against the old behavior.
- The decorative chapter number caused horizontal overflow at tablet widths. It now moves outside the reading column only when there is enough margin.
- A new 64×64 regression checks corpus pixels and shape through initialization, training, seed reset, and the matched comparison baseline.

## Implementation verification

All 145 unit tests pass, including the pause/resume and 64×64 regressions. Type checking, linting, and the production build pass. The production browser interaction test completes training, forecast/replay, rehearsal, closed-loop control, changing the goal, matched-budget objective comparison, reset, pause, and disposal on navigation. Visual checks cover light and dark themes at 390, 640, and 1280 CSS pixels; the world and noise chapters also have no horizontal overflow at the checked tablet widths.

## Reproduce

Start the dev server on port 5176, or set `WORLD_BENCH_ORIGIN` to its actual origin. Run GPU workloads sequentially to avoid competing for the device.

```sh
node tools/world-bench-browser.mjs '{"resolution":32,"steps":5000,"seed":17,"suite":true,"holdHorizon":6,"gentleCurl":true,"effort":0.01}' /tmp/world32.json
node tools/world-bench-browser.mjs '{"resolution":64,"steps":5000,"seed":17,"suite":true,"holdHorizon":6,"gentleCurl":true,"effort":0.01}' /tmp/world64.json
WORLD_BENCH_MODEL=tools/world-transformer-model.ts node tools/world-bench-browser.mjs '{"resolution":32,"steps":5000,"seed":17,"suite":true,"holdHorizon":6,"gentleCurl":true,"effort":0.01}' /tmp/world-transformer.json
node tools/world-transformer-smoke.mjs
node scripts/world-e2e.mjs http://127.0.0.1:5176
```

Repeat the three model comparisons with seeds 41 and 73. Use `steps:0` for the untrained baseline. The core timing includes training compilation and update-loop waits but excludes corpus generation, readout fitting, and UI evaluation checkpoints; those are not an estimate of total reader wait time. Timings vary with browser, compiler, device, and other activity. Benchmark JSON records model source hashes for runs made after that provenance field was added.
