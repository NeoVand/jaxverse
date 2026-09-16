# Kinetic instrument: known-dynamics planning diagnostic

This is a search-feasibility check, **not a learned-model result**. The diagnostic planner calls the true simulator and scores privileged joint angles. Neither that planner nor its output is used by the chapter's learned controller.

Reproduce with Node 24 or later:

```sh
node tools/world-oracle-bench.ts > /tmp/world-oracle-results.json
```

The script prints compact summaries to stderr and every episode's measurements to stdout. An optional name substring selects one configuration, for example `node tools/world-oracle-bench.ts h12-hold-64x3`.

## Protocol

All settings and criteria were chosen before the run. The search mirrors the initial production CEM: normalized bounded torques, two-observation action knots, zero initial means, initial standard deviation 0.8, a floor of 0.08, the best one-eighth of candidates as elites, and selection of the best sequence actually evaluated. Each step executes only the first torque pair and searches again from the new state. There is no warm start, velocity penalty, action-effort penalty, or corrective controller.

- Three starts: the initial resting pose `(-1.35, 1.65, 0, 0)`; that pose with velocities `(0.5, -0.5)`; and `(-1.7, 1.2, -0.2, 0.4)`. State order is `(q1, q2, v1, v2)`, with q2 relative to q1.
- Three goal poses: `(-0.9, 1.0)`, `(-1.8, 1.8)`, and `(-0.7, 2.65)`.
- Three solver seeds: 7, 19, and 37. The seed at action t is `1000 × seed + t`.
- Forty actions per episode, at 0.12 simulated seconds each: a 4.8-second budget.
- Reaching: circular joint RMS error at or below 0.15 radians.
- Dwell: five consecutive qualifying observations, or 0.6 simulated seconds. This measures remaining near a pose, not zero velocity.
- Terminal cost scores squared pose RMS at the last forecast. Hold cost averages it across the final four forecasts.

Every configuration evaluates the same 27 start/goal/solver-seed combinations. All other simulator parameters are defaults.

## Results

| Horizon | Cost      | Candidates × rounds | Reached | Dwelled | Mean final pose error (rad) | Mean final joint speed norm (rad/s) |
| ------- | --------- | ------------------- | ------- | ------- | --------------------------- | ----------------------------------- |
| 6       | Terminal  | 64 × 3              | 27/27   | 27/27   | 0.0220                      | 0.0682                              |
| 12      | Terminal  | 64 × 3              | 27/27   | 27/27   | 0.0452                      | 0.1544                              |
| 12      | Last four | 64 × 3              | 27/27   | 27/27   | 0.0246                      | 0.0527                              |
| 20      | Last four | 64 × 3              | 27/27   | 19/27   | 0.0806                      | 0.1326                              |
| 12      | Last four | 128 × 4             | 27/27   | 27/27   | 0.0188                      | 0.0573                              |

Every run ended inside the pose tolerance, including the eight horizon-20 dwell failures. Those failures reached too late to accumulate five qualifying observations before the budget expired. Thus final pose alone would conceal a meaningful difference in behavior.

The hold cost at horizon 12 produced lower average terminal motion than terminal-only scoring at that horizon, without directly penalizing velocity. It did not guarantee rest: 25 of its 27 runs ended below a joint-speed norm of 0.15 rad/s. Increasing candidates and rounds reduced average pose error, but did not monotonically improve every metric.

The shortest tested horizon already solved these local moves. A claim that short foresight necessarily fails would be false for this suite. The longest horizon performed worse under the fixed search budget even with exact dynamics; this result cannot be blamed on learned rollout error. More search dimensions and assigning cost further into the future are both relevant changes. The experiment does not separately identify their contributions.

## Visual and physical scope

These three goals are suitable initial presets. They change both the silhouette and endpoint position, while staying within the mechanism's normal workspace. At horizon 12 with the hold cost, actual tip travel ranged from 0.206 to 0.793 world units across the suite. The smallest angular gap to a fully folded, collinear configuration was 0.298 radians, about 17 degrees. The executed trajectories therefore avoided exact overlaid links. These are geometric checks, not a substitute for reviewing the rendered motion.

The simulator intentionally contains no link self-collision or joint limits. Projected links may overlap in the broader exploration corpus or during other plans. Joint marks and distinct link tones retain information in the sensor image; the page must describe the horizontal idealized mechanism without implying a collision-aware robot. Every possible tip remains within radius 0.85, leaving a 0.15-unit margin inside the sensor's ±1 viewport.

CPU planning averaged approximately 4.7 ms (horizon 6), 9–10 ms (horizon 12 at 64 × 3), 25.9 ms (horizon 20), and 22.2 ms (horizon 12 at 128 × 4) on Apple M4 with Node 24.21.0. Other development work was running concurrently; these are contextual timings, not isolated performance benchmarks or predictions of the learned browser planner's cost.

**Implication for the learning experiment:** the small search budget is sufficient for the proposed local goals under a useful physical metric. Use horizon 6 as a short-rollout reaching baseline and horizon 12 with the final-four cost for the gentle-arrival comparison. A learned controller still needs separate evidence: known dynamics with a learned embedding cost, followed by learned dynamics with that cost. The oracle is neither a guaranteed upper bound nor permission to display its motion as learned behavior.
