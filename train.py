#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import math
import shutil
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from loader import (
    data_module_from_config,
    materialize_dataset_metadata,
    resolve_train_split,
)
from model import model_from_config


def parse_scalar(value: str) -> Any:
    return yaml.safe_load(value)


def set_dotted(config: dict, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE: {expression}")
    dotted, raw = expression.split("=", 1)
    keys = dotted.split(".")
    current = config
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = parse_scalar(raw)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the S1-conditioned MolGuidance model")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config. Optional when --resume points to a run directory.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Existing run directory or Lightning checkpoint.",
    )
    parser.add_argument(
        "--seed-model",
        type=Path,
        default=None,
        help="Initialize model weights from a checkpoint without resuming optimizer state.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser.parse_args()


def resolve_resume(resume: Path | None) -> tuple[Path | None, Path | None]:
    """Return (run_dir, checkpoint_path)."""
    if resume is None:
        return None, None
    resume = resume.expanduser().resolve()
    if resume.is_dir():
        checkpoint = resume / "checkpoints" / "last.ckpt"
        if not checkpoint.is_file():
            candidates = sorted((resume / "checkpoints").glob("*.ckpt"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoint found under {resume}")
            checkpoint = candidates[-1]
        return resume, checkpoint
    if resume.is_file():
        run_dir = resume.parent.parent if resume.parent.name == "checkpoints" else None
        return run_dir, resume
    raise FileNotFoundError(f"Resume path not found: {resume}")


def resolve_config_path(
    requested_config: Path | None,
    resume_run_dir: Path | None,
) -> Path:
    if requested_config is not None:
        path = requested_config.expanduser().resolve()
    elif resume_run_dir is not None:
        path = resume_run_dir / "config.yaml"
    else:
        raise ValueError("--config is required unless --resume supplies a run directory")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def unique_run_dir(output_root: Path, base_name: str) -> tuple[str, Path]:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = base_name.strip() or "run"
    run_name = f"{base_name}_{timestamp}"
    run_dir = output_root / run_name
    suffix = 1
    while run_dir.exists():
        run_name = f"{base_name}_{timestamp}_{suffix:02d}"
        run_dir = output_root / run_name
        suffix += 1
    return run_name, run_dir


class FiniteLossCallback(Callback):
    """Stop immediately instead of silently propagating NaN/Inf losses."""

    def on_before_backward(self, trainer, pl_module, loss):
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(
                f"Non-finite loss detected at global_step={trainer.global_step}: {loss}"
            )


def build_checkpoint_callbacks(
    run_dir: Path,
    config: dict,
) -> tuple[ModelCheckpoint, ModelCheckpoint | None]:
    """Build independent best-model and latest-state checkpoint callbacks."""
    checkpoint_cfg = dict(config.get("checkpointing", {}))
    checkpoint_dir = str(run_dir / "checkpoints")
    checkpoint_cfg["dirpath"] = checkpoint_dir
    save_latest = bool(checkpoint_cfg.pop("save_last", True))
    best_callback = ModelCheckpoint(
        save_last=False,
        **checkpoint_cfg,
    )
    latest_callback = None
    if save_latest:
        latest_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="latest-epoch={epoch}-step={step}",
            save_top_k=0,
            save_last=True,
            every_n_epochs=1,
            monitor=None,
        )
    return best_callback, latest_callback


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    resume_run_dir, resume_checkpoint = resolve_resume(args.resume)
    config_path = resolve_config_path(args.config, resume_run_dir)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    for expression in args.overrides:
        set_dotted(config, expression)

    if args.resume is not None and args.seed_model is not None:
        raise ValueError("Use either --resume or --seed-model, not both")

    # Resolve the atom-type vocabulary from the processed PT file.  This only
    # materializes the atom-class order; it does not change molecule atom counts.
    materialize_dataset_metadata(config)

    if args.output_dir is not None:
        config["training"]["output_dir"] = str(args.output_dir.expanduser().resolve())
    output_root = Path(config["training"]["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if resume_run_dir is not None:
        run_dir = resume_run_dir
        run_name = run_dir.name
    else:
        base_name = args.run_name or config.get("wandb", {}).get("name") or "run"
        run_name, run_dir = unique_run_dir(output_root, base_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    checkpoint_callback, latest_checkpoint_callback = (
        build_checkpoint_callbacks(run_dir, config)
    )

    if args.no_wandb or args.debug or config.get("wandb", {}).get("mode") == "disabled":
        logger = CSVLogger(save_dir=str(run_dir), name="lightning_logs")
    else:
        wandb_cfg = dict(config.get("wandb", {}))
        wandb_cfg.pop("save_dir", None)
        wandb_cfg.setdefault("project", "mol-fm")
        wandb_cfg["name"] = run_name
        wandb_cfg["save_dir"] = str(run_dir)
        logger = WandbLogger(config=config, **wandb_cfg)

    data_module = data_module_from_config(config, seed=args.seed)
    # Build datasets before constructing Trainer so fractional validation limits can
    # be checked against the actual edge-aware batch count.
    data_module.setup("fit")
    val_batch_count = len(data_module.val_dataloader())

    seed_model = args.seed_model.expanduser().resolve() if args.seed_model else None
    if seed_model is not None and not seed_model.is_file():
        raise FileNotFoundError(f"Seed checkpoint not found: {seed_model}")
    model = model_from_config(config, seed_model)

    trainer_cfg = dict(config["training"].get("trainer_args", {}))
    trainer_cfg["val_check_interval"] = config["training"]["evaluation"].get(
        "val_loss_interval", 1.0
    )
    trainer_cfg["check_val_every_n_epoch"] = 1
    # A custom batch sampler already handles distributed sharding.
    trainer_cfg["use_distributed_sampler"] = False

    limit_val = trainer_cfg.get("limit_val_batches")
    if isinstance(limit_val, float) and 0.0 < limit_val < 1.0:
        if math.floor(val_batch_count * limit_val) == 0:
            print(
                "Validation fraction would produce zero batches; "
                "using one validation batch instead."
            )
            trainer_cfg["limit_val_batches"] = 1
    if args.debug:
        trainer_cfg["limit_train_batches"] = 10
        trainer_cfg["limit_val_batches"] = 1

    callbacks = [
        checkpoint_callback,
        FiniteLossCallback(),
        TQDMProgressBar(refresh_rate=1 if args.debug else 20),
    ]
    if latest_checkpoint_callback is not None:
        callbacks.append(latest_checkpoint_callback)

    early_cfg = dict(config.get("training", {}).get("early_stopping", {}))
    if early_cfg.pop("enabled", True):
        early_cfg.setdefault("monitor", "val_cond_total_loss")
        early_cfg.setdefault("mode", "min")
        early_cfg.setdefault("patience", 50)
        early_cfg.setdefault("min_delta", 0.0)
        early_cfg.setdefault("check_finite", True)
        early_cfg.setdefault("verbose", True)
        callbacks.append(EarlyStopping(**early_cfg))
    # LearningRateMonitor needs a logger.
    if logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **trainer_cfg)

    if resume_checkpoint is not None:
        training_mode = "resume (model + optimizer + scheduler state)"
        loaded_checkpoint = str(resume_checkpoint)
    elif seed_model is not None:
        training_mode = "seed weights only"
        loaded_checkpoint = str(seed_model)
    else:
        training_mode = "fresh start (no checkpoint loaded)"
        loaded_checkpoint = "none"

    atom_counts = data_module.train_dataset.atom_counts
    print(f"Run directory: {run_dir}")
    print(f"Config: {config_path}")
    print(f"Training mode: {training_mode}")
    print(f"Loaded checkpoint: {loaded_checkpoint}")
    print(f"Train split: {resolve_train_split(config['dataset'])}")
    print(f"Atom-type classes: {len(config['dataset']['atom_map'])}")
    print(f"Atom-type map: {config['dataset']['atom_map']}")
    print(
        "Molecule atom-count range: "
        f"{int(atom_counts.min())}..{int(atom_counts.max())} "
        "(read from node_idx_array; not changed by atom_map)"
    )
    print(f"Train molecules: {len(data_module.train_dataset)}")
    print(f"Validation molecules: {len(data_module.val_dataset)}")
    print(f"Validation batches: {val_batch_count}")
    print("Validation policy: fully conditional; CFG dropout is disabled for validation")
    print(
        "Total loss: "
        + " + ".join(
            f"{model.total_loss_weights[feat]}*L_{feat}"
            for feat in model.canonical_feat_order
        )
        + (
            f" + {model.valence_loss_weight}*L_valence"
            if model.valence_loss_weight > 0.0
            else ""
        )
    )
    print("Early stopping monitor: val_cond_total_loss")

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=(
            str(resume_checkpoint)
            if resume_checkpoint is not None
            else None
        ),
        weights_only=False,
    )

    # Expose a stable prediction path independent of Lightning's filename format.
    if checkpoint_callback.best_model_path:
        best_source = Path(checkpoint_callback.best_model_path).resolve()
        best_alias = (run_dir / "checkpoints" / "best.ckpt").resolve(strict=False)
        if best_source != best_alias:
            best_alias.unlink(missing_ok=True)
            try:
                best_alias.symlink_to(best_source.name)
            except OSError:
                shutil.copy2(best_source, best_alias)
        print(f"Best checkpoint: {best_alias}")


if __name__ == "__main__":
    main()
