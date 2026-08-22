# SIGReg DVN Experiment Notes

This folder is a separate modification workspace. The original project path is
not modified by these SIGReg edits.

Changed files live under `train/`:

- `vint_train/models/regularizers.py`
  - Adds a local SIGReg implementation adapted from LeWorldModel.
- `vint_train/models/dvn/dvn.py`
  - Computes `sigreg_loss` on the visual latent sequence
    `(history, current observation, next observations)`.
- `train.py`
  - Passes `sigreg.knots` and `sigreg.num_proj` into DVN.
  - Passes `sigreg.weight` into the DVN train/eval loop.
- `vint_train/training/train_eval_loop.py`
  - Threads `sigreg_weight` through the DVN loop.
- `vint_train/training/train_utils.py`
  - Logs `sigreg_loss`.
  - Adds `sigreg_weight * sigreg_loss` to the DVN total loss.
- `config/dvn.yaml`
  - Adds:
    - `sigreg.weight: 0.09`
    - `sigreg.knots: 17`
    - `sigreg.num_proj: 1024`
  - Points `go_stanford` split paths at the current project.

Known pre-existing issue:

- The active `vint_train/models/dvn/dvn.py` currently returns
  `dist_pred`, `action_pred`, `loss_dyn`, and `sigreg_loss`, but the existing
  DVN training loop expects `action_scores` as well. This mismatch appears to
  pre-date the SIGReg changes, so reproduce or repair the base DVN path before
  attributing runtime failures to the regularizer.
