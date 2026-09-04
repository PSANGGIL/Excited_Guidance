# Full-size 5-class curriculum model recovery record

Recorded: 2026-09-04 UTC

## Immutable identity

```yaml
model_id: os_cfg_curriculum_fullsize_restore_e1m_20260901_091826
training_status: early_stopped
stopped_after_epoch: 533
source_branch: codex/condition-negative-curriculum
source_commit: f6eebe3529139f2f56b5e4b61a78b81e3e58024a
source_remote: origin/codex/condition-negative-curriculum
current_documentation_branch: codex/kekule-bond-representation
documentation_branch_commit_at_recording: d1c5c6023133526f63db83b049e8ac4f9b06cfec
run_dir: runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826
run_config: runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/config.yaml
run_config_sha256: 4ccd6af4840a85bf1418687275668eb60391ef3a5baca82e92833eda8200c72c
dataset_dir: /home/jovyan/cpu-only-1-datavol-1/psg/generation/STGG-AL/data/data_
bond_representation: aromatic_5class
bond_classes: [none, single, double, triple, aromatic]
target_property: f_osc
target_transform: log10
target_epsilon: 1.0e-6
seed: 42
train_edge_budget: 1000000
eval_edge_budget: 1000000
```

The authoritative configuration is the saved run config, not the currently
checked-out `config.example.yaml`. The current branch uses a new 4-class Kekulé
dataset and is incompatible with this checkpoint unless the saved run config is
used.

## Checkpoints

Best checkpoint:

```yaml
path: runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/checkpoints/epoch=483-step=356224.ckpt
alias: runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/checkpoints/best.ckpt
sha256: 92bc844fcb14e1d58f7cf54052618bd8fd93324af80e6191a42d62811a29bc78
epoch: 483
step_in_metrics: 356223
validation:
  val_cond_total_loss: 1.5438657999038696
  val_condition_gain: 0.017992708832025528
  val_e_masked_accuracy: 0.9902300834655762
  val_selected_molecule_valence_pass_rate: 0.47870001196861267
```

Final checkpoint:

```yaml
path: runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/checkpoints/last.ckpt
sha256: 59d88924324671f6db7af3280da525eeaf0456221b0e0944283a333e23481b11
epoch: 533
step_in_metrics: 393023
train_total_loss: 1.6751571893692017
val_cond_total_loss: 1.5695407390594482
val_condition_gain: 0.013581400737166405
early_stopping_reason: no val_cond_total_loss improvement for 50 validations
```

Other retained top-k checkpoints are `epoch=457-step=337088.ckpt` and
`epoch=529-step=390080.ckpt`.

## Generation record

All strict-success counts require RDKit sanitization, one connected component,
SMILES conversion, and SDF writing. `SampleAnalyzer.frac_valid_mols` uses a
different definition and must not be compared directly.

| Checkpoint | Seed | N | Batch | Guidance x/a/c/e | Sanitize | Strict/SDF | Valence failures | Kekulize failures | Output |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| epoch 113 | 42 | 1,000 | 2 | 1/1/1/1 | - | 1 | - | - | `generated/fullsize_curriculum_e113_n1000_seed42_allguide1` |
| epoch 127 | 43 | 1,000 | 8 | 1/1/1/1 | - | 1 | - | - | `generated/fullsize_curriculum_e127_n1000_seed43_b8_allguide1` |
| epoch 144 | 44 | 1,000 | 8 | 1/1/1/1 | 4 | 4 | 647 | 346 | `generated/fullsize_curriculum_e144_n1000_seed44_b8_allguide1` |
| epoch 296 | 45 | 1,000 | 32 | 1/1/1/1 | 4 | 2 | 801 | 195 | `generated/fullsize_curriculum_best_e296_n1000_seed45_b32_allguide1` |
| epoch 334 | 46 | 1,000 | 32 | 1/1/1/1 | 7 | 5 | 865 | 128 | `generated/fullsize_curriculum_best_e334_n1000_seed46_b32_allguide1` |
| epoch 334 | 47 | 10,000 | 32 | 1/1/1/1 | 69 | 49 | 8,656 | 1,275 | `generated/fullsize_curriculum_best_e334_n10000_seed47_b32_allguide1` |
| epoch 483 | 48 | 1,000 | 32 | 1/1/1/1 | 7 | 6 | 731 | 262 | `generated/fullsize_curriculum_best_e483_n1000_seed48_b32_allguide1` |

The best-supported strict validity estimate is 49/10,000 = 0.49%. Loss
improvement from epoch 334 to 483 did not materially change this rate in the
1,000-sample checks. Valence and Kekulize errors remain the dominant failures.

## xTB/sTDA record for the 10,000-sample run

```yaml
input_valid_molecules: 49
xtb_version: 6.7.1
xtb_environment: /home/jovyan/cpu-only-1-datavol-1/xtb_env
xtb4stda: /home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home/exe/xtb4stda
stda: /home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home/exe/stda
optimization: "--opt loose --cycles 500"
parallel_jobs: 8
xtb_converged: 49
stda_parsed: 49
state_window: 1-10
top1_os_mean: 0.6150428571428571
top1_os_median: 0.5282
top1_os_range: [0.0825, 2.3214]
```

Relevant outputs under the 10,000-sample run `sdf/` directory:

- `stda_top_os_state1-10.csv`
- `stda_top3_os_state1-10.csv`
- `stda_top5_os_state1-10.csv`
- `stda_top5_os_comparison_wide.csv`
- `stda_closest_to_0.5_state1-10.csv`
- `stda_top3_os_grid.png` and per-molecule/page PNG files

## Exact restoration commands

Use the source branch in a clean worktree or switch to it after preserving any
local changes:

```bash
git switch codex/condition-negative-curriculum
git rev-parse HEAD
# Expected: f6eebe3529139f2f56b5e4b61a78b81e3e58024a
```

Reproduce the epoch 483 generation:

```bash
/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python predict.py \
  --run runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/checkpoints/epoch=483-step=356224.ckpt \
  --target 0.5 \
  --n-mols 1000 \
  --max-batch-size 32 \
  --device cuda:0 \
  --seed 48 \
  --guide-w-x 1.0 --guide-w-a 1.0 --guide-w-c 1.0 --guide-w-e 1.0 \
  --output-dir generated/fullsize_curriculum_best_e483_n1000_seed48_b32_allguide1_reproduced \
  --analyze
```

Resume from the final training state only if additional training is explicitly
desired. The original run already met its early-stopping condition:

```bash
/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python train.py \
  --config runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/config.yaml \
  --resume runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/checkpoints/last.ckpt \
  --no-wandb
```

Before restoration, verify integrity with:

```bash
sha256sum \
  runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/config.yaml \
  runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/checkpoints/epoch=483-step=356224.ckpt \
  runs/os_cfg_curriculum_fullsize_restore_e1m_20260901_091826/checkpoints/last.ckpt
```
