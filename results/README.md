# Extended results gallery

The five-page paper is the primary publication artifact. This gallery keeps the broader evidence chain visible for readers who want to understand the project beyond the page limit. These figures are complementary project materials; they are not additional claims beyond the manuscript's stated evaluation boundary.

## Evidence map

| Role | Figure | What it supports |
|---|---|---|
| Motivation | [VINT versus DVN motivation](figures/fig1_motivation_vint_vs_dvn.pdf) | Route progress can remain high while terminal arrival fails. |
| Method | [Imagined-endpoint scoring](figures/fig3_imagined_endpoint_scoring.png) | Candidate trajectories are compared through predicted latent endpoints. |
| Training | [Latent prediction and SIGReg](figures/fig4_lfp_sigreg_training.png) | Future-frame supervision and non-collapse regularization are training-time components. |
| Main offline result | [Go Stanford held-out results](figures/fig5_go_stanford_main_results.png) | The primary action and latent-future prediction comparisons. |
| Cross-domain transfer | [Cross-dataset generalization](figures/fig7_cross_dataset_generalization.png) | Checkpoint-only transfer across visual and platform shifts. |
| Mechanism | [Candidate scoring ablation](figures/fig6c_candidate_scorer_ablation.png) | The learned scorer contributes beyond fixed or random candidate selection. |
| Mechanism | [Latent collapse diagnostics](figures/fig6d_latent_collapse_diagnostics.png) | Representation quality is monitored alongside prediction performance. |
| Deployment | [Online candidate selection](figures/fig_online_candidate_selection_examples.png) | The propose--imagine--select pathway during closed-loop operation. |
| Scenario | [Gazebo overhead context](figures/fig8_overhead_scene_context.png) | Geometry, obstacles and fixed routes in indoor, forest and campus scenes. |
| Closed loop | [Repeated simulation](figures/fig8_repeated_closed_loop_simulation.png) | Five-run deployment outcomes and failure modes across scenarios. |

## Figure files

The gallery contains PNG previews for browser viewing and PDF exports for vector-quality download. The individual overhead panels are also preserved for readers who want to inspect each scenario separately:

- [Indoor obstacle](figures/fig8_2_indoor_obstacle_overhead_trajectory.png)
- [Forest road](figures/fig8_3_forest_overhead_trajectory.png)
- [Campus road](figures/fig8_4_campus_overhead_trajectory.png)
- [Closed-loop trajectories](figures/fig8_closed_loop_simulation_trajectories.png)
- [Training dynamics](figures/figS1_main_training_dynamics.png)

## Compact result tables

The `data/` folder contains small, human-readable CSV summaries selected from the analysis outputs:

- `offline_main.csv`: primary held-out action and latent-future metrics.
- `cross_dataset.csv` and `cross_dataset_checkpoint_eval.csv`: checkpoint-only transfer results.
- `ablation_go_stanford.csv`: candidate-number and SIGReg configuration comparisons.
- `candidate_scorer.csv`: learned, first-candidate, random and oracle selection comparisons.
- `latent_diagnostics.csv`: effective-rank and variance diagnostics for learned representations.
- `simulation_summary.csv` and `simulation_runs.csv`: repeated closed-loop outcomes and per-run records.
- `closed_loop_summary.csv`: compact scene-level trajectory summary.

These are derived analysis tables, not raw sensor archives. Raw datasets, ROS bags and model checkpoints remain outside this repository because they are large, environment-specific and may have separate redistribution constraints.

## Reading order

For a quick project tour, read the motivation figure first, then the method and online-selection figures, followed by the cross-domain and repeated-simulation results. This preserves the intended argument: identify the arrival gap, introduce latent consequence prediction, show that the predicted endpoint affects selection, and then evaluate transfer and closed-loop behavior.
