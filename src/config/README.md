# Public configurations

The public configuration set is intentionally limited to the LCP-Nav study:

- `dvn_sigreg_go_stanford_15.yaml`: main SIGReg training configuration.
- `dvn_sigreg_go_stanford_eval_smoke.yaml`: lightweight evaluation smoke test.
- `dvn_k1_*`, `dvn_k3_*`: candidate-number and checkpoint-role comparisons.
- `dvn_sigreg_w003_*`, `dvn_sigreg_w006_*`, `dvn_k3_sigreg_w015_*`: SIGReg sensitivity checks.
- `vint_go_stanford_15.yaml`: direct-policy baseline configuration.

Paths under `datasets` are placeholders. Replace them with the local dataset and split locations before running an experiment.
