# Loss and validation review

## 1. Is the S1 target used in validation?

Yes, after this revision.

The validation loader reads the property stored in `val_data_processed.pt`,
selects `dataset.conditioning.property`, and normalizes it with
`train_data_property_normalization.pt`. The normalized value is attached to
each graph as `graph.prop`.

The CFG model now has three explicit modes:

- `mixed`: random conditional/unconditional embedding; training only
- `conditional`: always use `graph.prop`; validation only
- `unconditional`: always use a zero property embedding

`compute_batch_losses(stage="val")` selects `conditional`, so
`val_cond_total_loss` is no longer contaminated by random `p_uncond` dropout.

## 2. What is the target of the loss?

The loss target is the validation molecule's structure, not the scalar S1
value itself.

For a validation molecule with final data

```text
(x_1, a_1, c_1, e_1, y_S1)
```

the model samples a flow time `t`, constructs a corrupted state `z_t`, and
predicts the final structure while receiving `y_S1` as a condition.

The default total loss is:

```text
L_total = 3.0 L_x + 0.4 L_a + 1.0 L_c + 2.0 L_e
```

### Coordinate term

```text
L_x = mean((x_1_pred - x_1_true)^2)
```

Coordinates are centered by molecule in the loader before training.

### Atom, charge, and bond terms

```text
L_a = CrossEntropy(atom_logits, atom_class)
L_c = CrossEntropy(charge_logits, charge_class)
L_e = CrossEntropy(bond_logits, bond_class)
```

For CTMC, categorical entries that have already become unmasked at sampled time
`t` are assigned target `-100` and excluded with
`CrossEntropyLoss(ignore_index=-100)`. Therefore categorical loss is computed
only on entries that are still masked and need reconstruction.

If an entire modality has no masked entries in a batch, `_safe_feature_loss`
returns differentiable zero instead of NaN.

## 3. Does this validate S1 generation accuracy?

No. This is an important distinction.

There is no S1 prediction head and no direct term such as:

```text
L_S1 = |S1_pred - S1_target|
```

A lower `val_cond_total_loss` means that the correct S1 condition helps the
model explain or reconstruct molecules from the validation distribution. It
does not prove that a newly generated molecule has the requested S1 oscillator
strength.

Actual target accuracy must be measured after generation with an independent
S1 evaluator, such as the intended quantum-chemical workflow or a separately
validated surrogate model.

## 4. Added condition-use diagnostic

When enabled, validation repeats the same corruption/time realization after
rolling S1 targets across molecules in the batch.

```text
val_condition_gain
  = val_shuffled_condition_total_loss
  - val_cond_total_loss
```

Interpretation:

- positive: correct S1 condition is more useful than a wrong condition
- near zero: the model may be ignoring S1
- negative: incorrect conditions perform better, indicating a bug, poor
  conditioning, or unstable training

The same validation random seed is reused for the correct and shuffled passes,
so the comparison is not dominated by different corruption samples.

This metric is still an internal condition-use test, not an S1 accuracy test.

## 5. Early stopping

Early stopping and ModelCheckpoint now monitor the same epoch-level metric:

```text
val_cond_total_loss
```

Default settings:

```yaml
training:
  early_stopping:
    enabled: true
    monitor: val_cond_total_loss
    mode: min
    patience: 50
    min_delta: 0.0
```

Validation metrics are logged with `on_step: false` and `on_epoch: true`, so
checkpointing and early stopping use the full validation epoch rather than the
last validation batch.

## 6. Remaining external validation requirement

For target values such as `0.01, 0.1, 0.5, 1.0, 3.0`, generate sufficiently
large sets and independently evaluate S1 oscillator strength. Recommended
metrics:

```text
MAE(target, evaluated_S1)
Pearson(target, evaluated_S1)
Spearman(target, evaluated_S1)
success rate within tolerance
validity by target
uniqueness by target
novelty by target
```

Without this step, the correct claim is "S1-conditioned structural generation,"
not "generation of molecules verified to have the requested S1 value."
