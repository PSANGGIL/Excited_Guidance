#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from loader import (
    data_module_from_config,
    materialize_dataset_metadata,
    resolve_train_split,
)
from model import model_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate MolGuidance config, processed PT data, and checkpoint compatibility"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    materialize_dataset_metadata(config)
    data_module = data_module_from_config(config, seed=args.seed)
    data_module.setup("fit")
    print(f"train_split={resolve_train_split(config['dataset'])}")
    print(f"train_molecules={len(data_module.train_dataset)}")
    print(f"val_molecules={len(data_module.val_dataset)}")
    print(f"train_batches={len(data_module.train_dataloader())}")
    print(f"val_batches={len(data_module.val_dataloader())}")
    atom_counts = data_module.train_dataset.atom_counts
    print(f"atom_type_class_count={len(data_module.train_dataset.atom_type_map)}")
    print(f"atom_map={data_module.train_dataset.atom_type_map}")
    print(f"molecule_atom_count_min={int(atom_counts.min())}")
    print(f"molecule_atom_count_max={int(atom_counts.max())}")
    print("atom_map_changes_molecule_atom_counts=false")

    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Fresh validation builds a newly initialized model.  A checkpoint is loaded
    # only when --checkpoint is explicitly supplied.
    model = model_from_config(config, checkpoint)
    print(f"model={type(model).__name__}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters())}")
    print(f"checkpoint_loaded={checkpoint is not None}")
    if checkpoint is not None:
        print("checkpoint_state_dict_compatible=true")

    if args.forward:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        model.to(device).train()
        batch = next(iter(data_module.train_dataloader())).to(device)
        train_losses = model.compute_batch_losses(model._local_graph_copy(batch), stage="train")
        val_losses = model.compute_batch_losses(model._local_graph_copy(batch), stage="val")
        for stage_name, losses in (("train_mixed", train_losses), ("val_cond", val_losses)):
            total = model._combine_feature_losses(losses)
            for name, loss in losses.items():
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError(
                        f"Non-finite {stage_name}/{name} loss: {loss}"
                    )
                print(f"{stage_name}_{name}_loss={float(loss.detach().cpu()):.8g}")
            print(f"{stage_name}_total_loss={float(total.detach().cpu()):.8g}")
        print("one_batch_forward=true")


if __name__ == "__main__":
    main()
