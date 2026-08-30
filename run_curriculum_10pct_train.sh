#!/usr/bin/env bash
set -euo pipefail

# Full curriculum training profile measured at about 17.1 GiB peak VRAM
# (roughly 9.3% of the 183-GiB B200).
exec /home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python train.py \
  --config config.example.yaml \
  --run-name os_cfg_curriculum_10pct_train \
  --seed 42 \
  --no-wandb \
  --set training.batch_size=128 \
  --set training.max_num_edges=220000 \
  --set training.max_num_edges_eval=220000 \
  --set training.num_workers=2 \
  --set model_setting.property_embedding_dim=128 \
  --set vector_field.n_vec_channels=8 \
  --set vector_field.n_hidden_scalars=128 \
  --set vector_field.n_hidden_edge_feats=64 \
  --set vector_field.n_molecule_updates=3 \
  --set vector_field.n_message_gvps=2 \
  --set vector_field.n_update_gvps=2 \
  "$@"
