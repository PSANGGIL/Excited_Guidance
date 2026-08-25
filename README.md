# MolGuidance S1-conditioned simple v6

This version exposes only two public commands:

```bash
python train.py --config config.example.yaml
python predict.py --run runs/<RUN_NAME> --target 3.0 --n-mols 100
```

The previous `trainer.py`, `predictor.py`, `train.sh`, `pred.sh`, and `pred_run`
execution layers were removed. Shell process management can still be done
externally with `nohup`, `tmux`, Slurm, or another scheduler when needed.

## Files

- `train.py`: training, resume, checkpointing, logging, and early stopping
- `predict.py`: checkpoint discovery, S1-conditioned sampling, validation, CSV/SDF, and grid output
- `loader.py`: processed-database loading and automatic metadata discovery
- `model.py`: flow-matching model, CFG, loss, and sampling
- `validate_setup.py`: data/config/checkpoint smoke test
- `config.example.yaml`: one configuration file for training

## Automatic atom-type vocabulary

`dataset.atom_map` is no longer specified by the user.

At startup, `loader.py` reads the train processed database and determines the
class-index-to-element mapping in this order:

1. `train_data_processed.pt["atom_map"]`
2. If absent, infer it from per-atom `atomic_numbers` and the one-hot
   `atom_types` tensor

The inferred class order is written into the resolved run `config.yaml`. The
validation and test splits are checked against the train split. A mismatch in
atom-class order causes an immediate error instead of silently reinterpreting
one-hot classes.

A processed database therefore must store either:

```python
{
    "atom_map": ["H", "C", "N", "O", ...],
    "atom_types": one_hot_tensor,
}
```

or:

```python
{
    "atomic_numbers": per_atom_atomic_numbers,
    "atom_types": one_hot_tensor,
}
```

For transformed targets such as `s1_log_os`, the normalization file must also
store the same transform contract used by preprocessing:

```python
{
    "mean": ...,
    "std": ...,
    "target_transform": "log",   # or log10/log1p/identity
    "target_epsilon": ...,
}
```

Training aborts if the processed split and normalization file disagree.

## Training

```bash
python train.py \
  --config config.example.yaml \
  --seed 42
```

Resume model, optimizer, and scheduler state:

```bash
python train.py --resume runs/<RUN_NAME>
```

Initialize only model weights from a checkpoint:

```bash
python train.py \
  --config config.example.yaml \
  --seed-model runs/<RUN_NAME>/checkpoints/last.ckpt
```

Override a YAML value:

```bash
python train.py \
  --config config.example.yaml \
  --set training.trainer_args.devices=1 \
  --set training.batch_size=128
```

## Prediction

Use a run directory. `predict.py` automatically reads its `config.yaml` and
`checkpoints/best.ckpt` (falling back to `last.ckpt`):

```bash
python predict.py \
  --run runs/<RUN_NAME> \
  --target 3.0 \
  --n-mols 100 \
  --device cuda:0
```

An exact checkpoint can also be supplied:

```bash
python predict.py \
  --run runs/<RUN_NAME>/checkpoints/epoch=0123-step=45678.ckpt \
  --target 3.0
```

The checkpoint must remain under `RUN_DIR/checkpoints`, because prediction uses
the resolved `RUN_DIR/config.yaml` saved during training. This prevents a
checkpoint from being combined with an unrelated config.

Default outputs are written under:

```text
generated/YYYYMMDD_HHMMSS/
├── generated.csv
├── generated_grid20.png
└── sdf/
    ├── 00000.sdf
    ├── 00001.sdf
    └── ...
```

Use a fixed output directory with `--output-dir`.

## What the loss actually measures

For each molecule, the loader provides the true final structure

- centered coordinates: `x_1_true`
- atom classes: `a_1_true`
- formal-charge classes: `c_1_true`
- bond classes: `e_1_true`
- normalized S1 condition: `graph.prop`

A random flow time `t` is sampled, and the model receives a corrupted
intermediate structure `z_t` together with the property condition.

With the default configuration, the total loss is:

```text
L_total = 3.0 L_x + 0.4 L_a + 1.0 L_c + 2.0 L_e + 0.05 L_valence
```

where:

- `L_x`: mean squared error between predicted final centered coordinates and
  `x_1_true`
- `L_a`: atom-class cross entropy
- `L_c`: charge-class cross entropy
- `L_e`: bond-class cross entropy
- `L_valence`: log-scaled squared overflow of expected bond-order valence above
  the ground-truth atom/charge valence cap

`L_valence` is computed from differentiable bond probabilities during training.
It does not remove bonds or otherwise rewrite generated molecules. Prediction
also defaults categorical CFG weights (`a`, `c`, and `e`) to `1.0`; stronger
categorical guidance can be requested explicitly but may amplify invalid bonds.

For CTMC categorical features, cross entropy is applied only to entries that
are still masked at time `t`. Already-unmasked entries use target `-100` and are
excluded by `CrossEntropyLoss(ignore_index=-100)`.

The valence auxiliary term is evaluated on all endpoint bond probabilities,
including entries already revealed on the sampled CTMC path. This avoids leaving
some atoms without a chemical-consistency gradient in a batch.

For an existing run whose saved config predates this option, enable it while
resuming with:

```bash
python train.py --resume runs/<RUN_NAME> --set mol_fm.valence_loss_weight=0.05
```

### Important: there is no direct S1 regression loss

The S1 value is an input condition, not a supervised output head. Therefore the
training objective does **not** contain a term such as:

```text
|predicted_S1 - target_S1|
```

`val_cond_total_loss` answers this question:

> Given the correct S1 condition, how well does the model reconstruct/denoise
> the corresponding validation molecule?

It does not by itself prove that newly generated molecules have the requested
S1 value. That requires an independent S1 evaluator applied after generation.

## Corrected validation behavior

Training uses classifier-free dropout:

```text
p_uncond = 0.1
```

so approximately 10% of training molecules use a zero property embedding.

Validation no longer uses that random mixture. It always uses the actual S1
condition and reports:

```text
val_cond_x_loss
val_cond_a_loss
val_cond_c_loss
val_cond_e_loss
val_cond_total_loss
```

The corruption/time random seed is fixed by validation batch, so the same
validation batch is evaluated with the same Monte Carlo corruption at every
epoch. This makes checkpoint selection and early stopping less noisy.

When `training.validation.compare_shuffled_condition: true`, validation also
rolls the S1 targets across molecules and reports:

```text
val_shuffled_condition_total_loss
val_condition_gain
```

with:

```text
val_condition_gain = shuffled-condition loss - correct-condition loss
```

A positive value means the correct S1 condition helps reconstruction compared
with a wrong S1 condition. This is an internal condition-use diagnostic, not an
external S1 accuracy measurement.

### Intuitive validation metrics

With `training.validation.log_intuitive_metrics: true` (the default), the same
validation forward pass also reports human-readable diagnostics without sampling
new molecules:

```text
val_x_axis_rmse
val_x_atom_rms_displacement
val_x_true_bond_length_mae
val_{a,c,e}_masked_accuracy
val_{a,c,e}_masked_perplexity
val_{a,c,e}_masked_true_probability
val_{x,a,c,e,valence}_loss_contribution_percent
val_{expected,selected}_valence_overflow_mean
val_{expected,selected}_valence_overflow_p95
val_{expected,selected}_atom_valence_violation_rate
val_{expected,selected}_molecule_valence_pass_rate
val_condition_gain_percent
```

`expected` uses soft endpoint bond probabilities and is useful for monitoring
training. `selected` uses argmax bond classes and is closer to the categorical
graph that sampling will produce. Coordinate metrics use the processed data's
coordinate unit (normally Angstrom). These diagnostics do not replace raw
generation validity or independent S1 evaluation.

The progress bar shows compact aliases for the three most useful diagnostics:
`val_atom_rms_bar`, `val_bond_acc_bar`, and `val_valence_pass_bar`.

## Early stopping

Early stopping is enabled by default:

```yaml
training:
  early_stopping:
    enabled: true
    monitor: val_cond_total_loss
    mode: min
    patience: 50
    min_delta: 0.0
```

Model checkpoints and early stopping monitor the same fully conditional epoch
metric: `val_cond_total_loss`.

## Required external S1 evaluation

To establish actual conditional generation performance, generate molecules at
several targets and independently calculate or predict their S1 oscillator
strengths. Report at least:

- target vs evaluated S1 Pearson/Spearman correlation
- MAE between target and evaluated S1
- success rate within a chosen tolerance
- validity/uniqueness/novelty by target range
- comparison against unconditional generation
