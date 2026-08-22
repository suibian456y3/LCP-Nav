# Compact result tables

The CSV files in this directory are small derived summaries intended for inspection and lightweight downstream plotting. They are copied from the analysis tables used to prepare the manuscript and are kept separate from the raw datasets and ROS bags.

Column names and metric definitions follow the manuscript. In particular:

- `action_loss` and `lfp_loss` are model-evaluation losses.
- `validated_goal_success` applies the composite closed-loop criterion used in the paper.
- `strict_validated_success_1m` applies the stricter 1 m endpoint check.
- `endpoint_error_m` is the final distance to the demonstration endpoint.
- `path_completion_rate_pct` is the maximum topological progress during a run.

Absolute workstation paths and raw-source pointers have been removed from the release tables. The original analysis workspace retains the complete provenance records.
