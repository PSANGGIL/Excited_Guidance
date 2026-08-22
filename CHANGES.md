# v6 changes

- Public execution reduced to `train.py` and `predict.py`.
- Removed `trainer.py`, `predictor.py`, `train.sh`, `pred.sh`, and `pred_run`.
- Prediction now accepts one `--run` argument and automatically resolves the
  saved run config and checkpoint.
- Atom classes are inferred from processed database metadata, not manually
  specified in YAML.
- Train/validation/test atom and property metadata are checked for consistency.
- Validation is now fully conditional; random CFG dropout is used only during
  training.
- Added deterministic validation corruption seeds.
- Added `val_shuffled_condition_total_loss` and `val_condition_gain`.
- Added early stopping on `val_cond_total_loss`.
- Checkpointing now monitors `val_cond_total_loss`.
- Documented the exact structural denoising loss and clarified that S1 is a
  condition rather than a direct regression target.

- Added strict checkpoint/config semantic checks for atom classes, conditioning
  property, transform, charge classes, and bond classes.
- Added preprocessing/normalization transform consistency checks.
- Prediction defaults to the stable `best.ckpt` alias created after training.
