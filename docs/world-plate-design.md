# Making learned consequences visible

The plates now hold questions fixed while the model's answers change. The MLP, training objective, experience collection, and planner budget are unchanged.

| Plate             | Reader's experiment                                                      | Evidence                                                                                        |
| ----------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| I · History       | Reveal the past of two identical poses.                                  | The same action has different consequences when motion differs.                                 |
| II · Learning     | Switch between the untrained and current model on fixed unseen examples. | Which recorded future is nearest to the predicted embedding, compared with copying the present. |
| III · Forecasting | Change the proposed actions while preserving the starting observation.   | A different forecast, tested by executing the same commands in the simulator.                   |
| IV · Collapse     | Remove the distribution constraint under a matched training budget.      | Read the loss beside representation spread and the same future-matching test.                   |
| V · Planning      | Compare three candidate futures, then execute one action.                | Predicted costs explain the choice; its predicted pose remains beside the observed result.      |
| VI · Reuse        | Change the destination or scoring window after rehearsing.               | Cached predictions stay fixed while preferences change their scores and ranking.                |

Plate II gives the visual test the main area; recorded experience and the loss curve are available in an expandable section. Later plates distinguish changing a request from changing physical damping. A fresh observation helps planning react to errors; it does not retrain the frozen world model.

Plate III starts every action choice after the same 0.96-second push. Its two context pictures and connecting motor command come from that actual simulation. Keeping the push, reversing the torques, and releasing the motors each hold their selected commands for the entire forecast. Release therefore tests coasting from existing momentum, with the usual damping, instead of zero torque applied to a stationary arm. Long learned rollouts can still drift from the observed trajectory.

## The matching diagnostic

The evaluation uses the existing fixed validation batch of 64 transitions. For each transition, the predictor receives two past pictures and their aligned actions. It predicts one future embedding. That vector is compared with the encoded actual next picture and five nearby-looking alternative next pictures from the validation batch. Candidate sets use pixel distance, exclude duplicate pictures, and remain fixed across checkpoints. The actual outcome defines the test but is never passed to the predictor. Six display examples are chosen at evenly spaced batch indices before training, independently of their scores.

These are recorded photographs, not decoded or generated predictions. No labeled pose readout is fitted for this diagnostic. A tie does not count as a correct match. The tie tolerance is relative to the nearest distance: shrinking every embedding cannot create ties simply by crossing an absolute threshold. A regression rescales the real network's latent coordinates by one millionth and checks that its selections remain unchanged. The copy-present baseline uses the current encoder, making the benefit of prediction visible alongside what the representation already retains. This is a small retrieval test, not an estimate of accuracy across arbitrary worlds.

Browser WebGPU checks with the existing settings and 5,000 updates:

| Seed | Before training | After training | Copy present, final encoder |
| ---- | --------------: | -------------: | --------------------------: |
| 17   |           31/64 |          63/64 |                       45/64 |
| 41   |           35/64 |          63/64 |                       49/64 |
| 73   |           36/64 |          61/64 |                       47/64 |

The default seed reaches 39/64 at 100 updates, 51/64 at 500, and 58/64 at 1,000. Four of the six displayed examples change from incorrect to correct. [Raw checkpoints](world-future-matching-results.json) retain the choices and counts.

With only prediction loss at the same 5,000-update budget, the default seed matches 13/64 futures versus 63/64 with regularization, even though its prediction loss reaches 3.20e−9. Both results have zero ties. The small loss alone therefore does not establish useful prediction; small spread alone does not prove that every distinction has vanished. [Paired comparison and distances](world-matching-collapse-results.json).

## Keeping the planning comparison honest

Rehearsal returns the chosen sequence and two alternatives, their latent rollouts, and embeddings of the available goal pictures. The gallery preserves their identities and geometry. Changing a goal or intention recomputes the same latent cost used by the planner, including the unchanged motor-effort preference; it does not run inference or training. Step executes the first action of the best displayed sequence. Run performs a fresh search at each observation. A three-candidate gallery cannot substitute for that larger search.

The predicted and actual poses after an action are display diagnostics. Their difference includes the separately fitted readout's error, which is labeled in the plate. Physical state is not used to rank candidates.

Regression checks cover held-out provenance, candidate stability, unchanged training after evaluations, retained before-learning measurements, and reranking without changing forecasts. The browser flow additionally checks visible learning, action-sensitive forecasts, identical cached paths across goal/intention changes, the retained prediction after acting, and invalidation after reset.

Sensor images invert for dark display, including system dark mode. The canvas backing pixels and model inputs are unchanged. Layouts were inspected at 390, 640, and 1280 CSS pixels in both themes.

Final verification: 148 unit tests pass, Svelte reports zero errors and warnings, lint and production build pass, and the complete production browser flow passes. Browser checks cover manual and system theme preferences and verify unchanged canvas pixels across theme changes.
