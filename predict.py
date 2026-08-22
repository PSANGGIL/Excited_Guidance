#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from loader import materialize_dataset_metadata
from model import (
    ClassifierFreeGuidance,
    SampleAnalyzer,
    model_from_config,
)


REFERENCE_SPLIT_FILES = {
    "train": "train_data_processed.pt",
    "val": "val_data_processed.pt",
    "test": "test_data_processed.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate S1-conditioned molecules from one run directory or checkpoint."
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Training run directory or an explicit .ckpt file.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Raw S1 oscillator-strength target used for every generated molecule.",
    )
    parser.add_argument(
        "--property-values-file",
        type=Path,
        default=None,
        help="Optional .npy file containing one raw target per molecule.",
    )
    parser.add_argument("--n-mols", type=int, default=100)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--n-atoms-per-mol", type=int, default=None)
    parser.add_argument("--number-of-atoms-file", type=Path, default=None)
    parser.add_argument("--n-timesteps", type=int, default=100)
    parser.add_argument("--stochasticity", type=float, default=None)
    parser.add_argument("--hc-thresh", type=float, default=None)
    parser.add_argument("--guide-w-x", type=float, default=2.0)
    parser.add_argument("--guide-w-a", type=float, default=1.0)
    parser.add_argument("--guide-w-c", type=float, default=1.0)
    parser.add_argument("--guide-w-e", type=float, default=1.0)
    parser.add_argument(
        "--dfm-type",
        default="campbell",
        choices=["campbell", "campbell_rate_matrix"],
    )
    parser.add_argument(
        "--guidance-format",
        default="linear",
        choices=["linear", "log"],
    )
    parser.add_argument(
        "--where-to-apply-guide",
        default="probabilities",
        choices=["probabilities", "rate_matrix"],
    )
    #parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: generated/YYYYMMDD_HHMMSS",
    )
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--allow-unsanitized", action="store_true")
    parser.add_argument("--allow-multifragment", action="store_true")
    parser.add_argument("--skip-dataset-membership-check", action="store_true")
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--grid-count", type=int, default=20)
    parser.add_argument("--grid-mols-per-row", type=int, default=4)
    parser.add_argument("--grid-width", type=int, default=450)
    parser.add_argument("--grid-height", type=int, default=350)
    return parser.parse_args()


def resolve_run_artifacts(run: Path) -> tuple[Path, Path, Path]:
    """Return ``(run_dir, config_path, checkpoint_path)`` from one CLI argument."""
    run = run.expanduser().resolve()
    if run.is_dir():
        run_dir = run
        checkpoint = run_dir / "checkpoints" / "best.ckpt"
        if not checkpoint.is_file():
            checkpoint = run_dir / "checkpoints" / "last.ckpt"
        if not checkpoint.is_file():
            candidates = sorted(
                (run_dir / "checkpoints").glob("*.ckpt"),
                key=lambda path: path.stat().st_mtime,
            )
            if not candidates:
                raise FileNotFoundError(f"No checkpoint found under {run_dir}")
            checkpoint = candidates[-1]
    elif run.is_file():
        checkpoint = run
        if run.parent.name != "checkpoints":
            raise ValueError(
                "An explicit checkpoint must be inside RUN_DIR/checkpoints so its "
                "resolved config can be found automatically."
            )
        run_dir = run.parent.parent
    else:
        raise FileNotFoundError(f"Run path not found: {run}")

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Resolved run config not found: {config_path}")
    return run_dir, config_path, checkpoint


def load_config(
    path: Path,
) -> dict[str, Any]:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    if not isinstance(config, dict):
        raise TypeError(
            f"Config must be a YAML mapping: {path}"
        )

    return config


def load_1d_npy(
    path: Path,
    dtype: type,
    expected_length: int,
    option_name: str,
) -> list:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"{option_name} file not found: {path}"
        )

    values = (
        np.asarray(
            np.load(path),
            dtype=dtype,
        )
        .reshape(-1)
        .tolist()
    )

    if len(values) != expected_length:
        raise ValueError(
            f"{option_name} length must equal --n-mols: "
            f"{len(values)} != {expected_length}"
        )

    return values


def load_model_state_dict(
    model: ClassifierFreeGuidance,
    checkpoint_path: Path,
) -> None:
    """
    Load only the model parameters from a trusted Lightning
    checkpoint.

    This avoids the PyTorch weights_only=True default issue
    while not restoring optimizer or scheduler state.
    """
    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint_payload, dict)
        and "state_dict" in checkpoint_payload
    ):
        state_dict = checkpoint_payload["state_dict"]
        hparams = checkpoint_payload.get("hyper_parameters", {})
        semantic_checks = {
            "atom_type_map": list(model.atom_type_map),
            "conditioning_property": model.conditioning_property,
            "property_transform": model.property_transform,
            "n_atom_charges": model.n_atom_charges,
            "n_bond_types": model.n_bond_types,
        }
        for key, expected in semantic_checks.items():
            stored = hparams.get(key)
            if stored is not None and stored != expected:
                raise ValueError(
                    f"Checkpoint/config mismatch for {key}: "
                    f"checkpoint={stored!r}, resolved_run_config={expected!r}"
                )

    elif isinstance(
        checkpoint_payload,
        dict,
    ):
        state_dict = checkpoint_payload

    else:
        raise TypeError(
            "Checkpoint must be a state-dict mapping or "
            "a Lightning checkpoint: "
            f"{checkpoint_path}"
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )


def unresolved_ctmc_masks(
    sampled,
    model,
) -> dict[str, bool]:
    """
    Detect unresolved CTMC mask classes before RDKit
    conversion is trusted.
    """
    graph = sampled.g

    result = {
        "atom": False,
        "charge": False,
        "bond": False,
    }

    if "a_1" in graph.ndata:
        result["atom"] = bool(
            (
                graph.ndata["a_1"].argmax(
                    dim=-1
                )
                == model.n_atom_types
            )
            .any()
            .item()
        )

    if (
        not model.exclude_charges
        and "c_1" in graph.ndata
    ):
        result["charge"] = bool(
            (
                graph.ndata["c_1"].argmax(
                    dim=-1
                )
                == model.n_atom_charges
            )
            .any()
            .item()
        )

    if "e_1" in graph.edata:
        result["bond"] = bool(
            (
                graph.edata["e_1"].argmax(
                    dim=-1
                )
                == model.n_bond_types
            )
            .any()
            .item()
        )

    return result


def canonicalize_smiles(
    smiles: str | None,
) -> str | None:
    """
    Return canonical isomeric SMILES, or None when parsing
    fails.
    """
    if smiles is None:
        return None

    smiles = str(
        smiles
    ).strip()

    if not smiles:
        return None

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return None

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def load_processed_split_smiles(
    split_path: Path,
) -> set[str]:
    payload = torch.load(
        split_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Processed PT must contain a dictionary: "
            f"{split_path}"
        )

    if "smiles" not in payload:
        raise KeyError(
            "Processed PT has no 'smiles' field: "
            f"{split_path}"
        )

    stored_smiles = payload[
        "smiles"
    ]

    if stored_smiles is None:
        raise ValueError(
            "Processed PT contains smiles=None: "
            f"{split_path}"
        )

    canonical_set: set[str] = set()
    invalid_count = 0

    for smiles in stored_smiles:
        canonical = canonicalize_smiles(
            smiles
        )

        if canonical is None:
            invalid_count += 1

        else:
            canonical_set.add(
                canonical
            )

    print(
        f"[dataset] {split_path.name}: "
        f"stored={len(stored_smiles)}, "
        f"canonical_unique={len(canonical_set)}, "
        f"invalid_or_empty={invalid_count}"
    )

    return canonical_set


def load_reference_smiles(
    processed_dir: Path,
) -> dict[str, Any]:
    """
    Load canonical SMILES sets from train, validation,
    and test PT files.
    """
    processed_dir = (
        processed_dir
        .expanduser()
        .resolve()
    )

    split_sets: dict[
        str,
        set[str],
    ] = {}

    split_counts: dict[
        str,
        int,
    ] = {}

    missing_splits: list[str] = []

    for split_name, filename in (
        REFERENCE_SPLIT_FILES.items()
    ):
        split_path = (
            processed_dir
            / filename
        )

        if not split_path.is_file():
            print(
                "[warning] Reference split not found: "
                f"{split_path}"
            )

            split_sets[
                split_name
            ] = set()

            split_counts[
                split_name
            ] = 0

            missing_splits.append(
                split_name
            )

            continue

        split_set = (
            load_processed_split_smiles(
                split_path
            )
        )

        split_sets[
            split_name
        ] = split_set

        split_counts[
            split_name
        ] = len(split_set)

    if len(missing_splits) == len(
        REFERENCE_SPLIT_FILES
    ):
        raise FileNotFoundError(
            "No train/val/test processed PT files were "
            "found under: "
            f"{processed_dir}"
        )

    combined: set[str] = set()

    for split_set in split_sets.values():
        combined.update(
            split_set
        )

    return {
        "processed_dir": str(
            processed_dir
        ),
        "splits": split_sets,
        "split_counts": split_counts,
        "all": combined,
        "all_count": len(combined),
        "missing_splits": missing_splits,
    }


def get_membership(
    canonical_smiles: str,
    reference_info: dict[str, Any] | None,
) -> dict[str, bool | None]:
    if reference_info is None:
        return {
            "dataset_membership_checked": False,
            "in_train_dataset": None,
            "in_val_dataset": None,
            "in_test_dataset": None,
            "in_any_dataset": None,
            "is_novel": None,
        }

    split_sets = reference_info[
        "splits"
    ]

    in_train = (
        canonical_smiles
        in split_sets.get(
            "train",
            set(),
        )
    )

    in_val = (
        canonical_smiles
        in split_sets.get(
            "val",
            set(),
        )
    )

    in_test = (
        canonical_smiles
        in split_sets.get(
            "test",
            set(),
        )
    )

    in_any = (
        canonical_smiles
        in reference_info["all"]
    )

    return {
        "dataset_membership_checked": True,
        "in_train_dataset": in_train,
        "in_val_dataset": in_val,
        "in_test_dataset": in_test,
        "in_any_dataset": in_any,
        "is_novel": not in_any,
    }


def set_optional_sdf_property(
    molecule: Chem.Mol,
    name: str,
    value: Any,
) -> None:
    if value is not None:
        molecule.SetProp(
            name,
            str(value),
        )


def write_single_molecule_sdf(
    molecule: Chem.Mol,
    generation_index: int,
    output_sdf_dir: Path,
) -> Path:
    """
    Write one valid molecule as an individual SDF file.

    Examples:
        generation_index = 0   -> 00000.sdf
        generation_index = 7   -> 00007.sdf
        generation_index = 125 -> 00125.sdf
    """
    output_sdf_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sdf_path = (
        output_sdf_dir
        / f"{generation_index:05d}.sdf"
    )

    writer = Chem.SDWriter(
        str(sdf_path)
    )

    if writer is None:
        raise RuntimeError(
            f"Failed to create SDWriter: {sdf_path}"
        )

    writer.SetKekulize(
        False
    )

    try:
        writer.write(
            molecule
        )

    finally:
        writer.close()

    if (
        not sdf_path.is_file()
        or sdf_path.stat().st_size == 0
    ):
        sdf_path.unlink(
            missing_ok=True
        )

        raise IOError(
            f"Failed to write SDF file: {sdf_path}"
        )

    return sdf_path


def draw_generated_molecule_grid(
    rows: list[dict[str, Any]],
    output_png: Path,
    max_molecules: int,
    mols_per_row: int,
    sub_image_size: tuple[int, int],
) -> Path | None:
    if max_molecules <= 0:
        print(
            "[warning] --grid-count <= 0; "
            "grid image was skipped"
        )

        return None

    if mols_per_row <= 0:
        raise ValueError(
            "--grid-mols-per-row must be positive"
        )

    if min(sub_image_size) <= 0:
        raise ValueError(
            "--grid-width and --grid-height "
            "must be positive"
        )

    molecules: list[Chem.Mol] = []
    legends: list[str] = []

    for row in rows:
        if (
            not row.get("valid")
            or not row.get(
                "canonical_smiles"
            )
        ):
            continue

        molecule = Chem.MolFromSmiles(
            row["canonical_smiles"]
        )

        if molecule is None:
            continue

        rdDepictor.Compute2DCoords(
            molecule
        )

        molecules.append(
            molecule
        )

        legends.append(
            f"({row['generation_index']})"
        )

        if len(molecules) >= max_molecules:
            break

    if not molecules:
        print(
            "[warning] No valid molecules were "
            "available for the grid image"
        )

        return None

    output_png = (
        output_png
        .expanduser()
        .resolve()
    )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Draw.MolsToGridImage(
        molecules,
        legends=legends,
        molsPerRow=mols_per_row,
        subImgSize=sub_image_size,
        useSVG=False,
    )

    image.save(
        str(output_png)
    )

    print(
        f"Saved molecule grid: {output_png} "
        f"({len(molecules)} molecules)"
    )

    return output_png


def validate_arguments(args: argparse.Namespace) -> None:
    if args.n_mols <= 0:
        raise ValueError("--n-mols must be positive")
    if args.max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive")
    if args.n_timesteps < 2:
        raise ValueError("--n-timesteps must be at least 2")
    if args.n_atoms_per_mol is not None and args.n_atoms_per_mol <= 0:
        raise ValueError("--n-atoms-per-mol must be positive")
    if args.target is None and args.property_values_file is None:
        raise ValueError("Specify --target or --property-values-file")
    if args.target is not None and args.property_values_file is not None:
        raise ValueError("Use either --target or --property-values-file, not both")
    if args.number_of_atoms_file is not None and args.n_atoms_per_mol is not None:
        raise ValueError(
            "Use either --n-atoms-per-mol or --number-of-atoms-file, not both"
        )

def main() -> None:
    args = parse_args()
    validate_arguments(args)

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA is unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    run_dir, config_path, checkpoint = resolve_run_artifacts(args.run)
    config = load_config(config_path)
    materialize_dataset_metadata(config)

    model = model_from_config(config, seed_ckpt=None)
    load_model_state_dict(model, checkpoint)

    processed_dir = Path(config["dataset"]["processed_data_dir"]).expanduser().resolve()
    conditioning = config["dataset"].get("conditioning", {})
    property_name = conditioning.get("property")
    if not property_name:
        raise ValueError("The run config does not define dataset.conditioning.property")

    dataset_name = config["dataset"].get("dataset_name", "csvmol")
    properties_handle_method = config.get("model_setting", {}).get(
        "properties_handle_method", "concatenate_sum"
    )
    normalization_file = processed_dir / "train_data_property_normalization.pt"
    if not normalization_file.is_file():
        raise FileNotFoundError(f"Normalization file not found: {normalization_file}")

    target = args.target
    multiple_properties = None
    if args.property_values_file is not None:
        multiple_properties = load_1d_npy(
            args.property_values_file,
            float,
            args.n_mols,
            "--property-values-file",
        )

    number_of_atoms = None
    if args.number_of_atoms_file is not None:
        number_of_atoms = load_1d_npy(
            args.number_of_atoms_file,
            int,
            args.n_mols,
            "--number-of-atoms-file",
        )
        if any(value <= 0 for value in number_of_atoms):
            raise ValueError("All atom counts must be positive")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("generated") / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.expanduser().resolve()
    output_sdf_dir = output_dir / "sdf"
    output_csv = output_dir / "generated.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Conditioning property: {property_name}")
    print(f"Database-derived atom classes: {model.atom_type_map}")
    print(f"Output directory: {output_dir}")

    if args.skip_dataset_membership_check:
        reference_info = None

        print(
            "[info] Dataset membership check "
            "is disabled"
        )

    else:
        reference_info = (
            load_reference_smiles(
                processed_dir
            )
        )

        print(
            "[info] Reference dataset summary:\n"
            + json.dumps(
                {
                    "processed_dir": (
                        reference_info[
                            "processed_dir"
                        ]
                    ),
                    "split_counts": (
                        reference_info[
                            "split_counts"
                        ]
                    ),
                    "all_count": (
                        reference_info[
                            "all_count"
                        ]
                    ),
                    "missing_splits": (
                        reference_info[
                            "missing_splits"
                        ]
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    model.to(
        device
    ).eval()

    guide_w = {
        "x": args.guide_w_x,
        "a": args.guide_w_a,
        "c": args.guide_w_c,
        "e": args.guide_w_e,
    }

    molecules = []

    n_batches = math.ceil(
        args.n_mols
        / args.max_batch_size
    )

    with torch.inference_mode():
        for batch_idx in range(
            n_batches
        ):
            start = len(
                molecules
            )

            batch_size = min(
                args.max_batch_size,
                args.n_mols - start,
            )

            batch_properties = (
                multiple_properties[
                    start:start + batch_size
                ]
                if multiple_properties is not None
                else None
            )

            batch_atom_counts = (
                number_of_atoms[
                    start:start + batch_size
                ]
                if number_of_atoms is not None
                else None
            )

            common = {
                "device": device,
                "n_timesteps": (
                    args.n_timesteps
                ),
                "stochasticity": (
                    args.stochasticity
                ),
                "high_confidence_threshold": (
                    args.hc_thresh
                ),
                "properties_for_sampling": (
                    target
                ),
                "property_name": (
                    property_name
                ),
                "normalization_file_path": str(
                    normalization_file
                ),
                "properties_handle_method": (
                    properties_handle_method
                ),
                "multilple_values_to_one_property": (
                    batch_properties
                ),
                "guide_w": guide_w,
                "dfm_type": (
                    args.dfm_type
                ),
                "guidance_format": (
                    args.guidance_format
                ),
                "where_to_apply_guide": (
                    args.where_to_apply_guide
                ),
                "dataset_name": (
                    dataset_name
                ),
            }

            if args.n_atoms_per_mol is None:
                batch = (
                    model.sample_random_sizes(
                        batch_size,
                        number_of_atoms=(
                            batch_atom_counts
                        ),
                        **common,
                    )
                )

            else:
                fixed_atom_counts = torch.full(
                    (batch_size,),
                    args.n_atoms_per_mol,
                    dtype=torch.long,
                    device=device,
                )

                batch = model.sample(
                    fixed_atom_counts,
                    **common,
                )

            if len(batch) != batch_size:
                raise RuntimeError(
                    f"Sampler returned {len(batch)} "
                    f"molecules; expected {batch_size}"
                )

            molecules.extend(
                batch
            )

            print(
                f"Batch {batch_idx + 1}/"
                f"{n_batches}: generated "
                f"{len(molecules)}/"
                f"{args.n_mols}"
            )

    if not molecules:
        raise RuntimeError(
            "Model returned no molecules"
        )

    output_sdf_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    # canonical SMILES -> first valid generation index
    seen_generated: dict[str, int] = {}

    valid_count = 0
    unique_valid_count = 0
    generated_duplicate_count = 0
    in_train_count = 0
    in_val_count = 0
    in_test_count = 0
    in_any_count = 0
    novel_count = 0

    for index, sampled in enumerate(
        molecules
    ):
        raw_target = (
            multiple_properties[index]
            if multiple_properties is not None
            else target
        )

        masks = unresolved_ctmc_masks(
            sampled,
            model,
        )

        has_unresolved_mask = any(
            masks.values()
        )

        rdkit_mol_created = (
            sampled.rdkit_mol is not None
        )

        sanitize_ok = False
        single_component = False
        smiles_ok = False
        written_to_sdf = False
        canonical_smiles = None
        sdf_path = None
        error = ""

        is_unique_generated: bool | None = None

        duplicate_of_generation_index: (
            int | None
        ) = None

        membership = {
            "dataset_membership_checked": (
                reference_info is not None
            ),
            "in_train_dataset": None,
            "in_val_dataset": None,
            "in_test_dataset": None,
            "in_any_dataset": None,
            "is_novel": None,
        }

        try:
            if has_unresolved_mask:
                unresolved = [
                    name
                    for name, present in masks.items()
                    if present
                ]

                raise ValueError(
                    "unresolved CTMC mask: "
                    + ", ".join(unresolved)
                )

            if not rdkit_mol_created:
                raise ValueError(
                    "rdkit_mol is None"
                )

            candidate = Chem.Mol(
                sampled.rdkit_mol
            )

            try:
                Chem.SanitizeMol(
                    candidate
                )

                sanitize_ok = True

            except Exception:
                if not args.allow_unsanitized:
                    raise

            fragments = Chem.GetMolFrags(
                candidate
            )

            single_component = (
                len(fragments) == 1
            )

            if (
                not single_component
                and not args.allow_multifragment
            ):
                raise ValueError(
                    "molecule contains "
                    f"{len(fragments)} "
                    "disconnected fragments"
                )

            canonical_smiles = (
                Chem.MolToSmiles(
                    candidate,
                    canonical=True,
                    isomericSmiles=True,
                )
            )

            smiles_ok = bool(
                canonical_smiles
            )

            if not smiles_ok:
                raise ValueError(
                    "MolToSmiles returned "
                    "an empty string"
                )

            if canonical_smiles in seen_generated:
                is_unique_generated = False

                duplicate_of_generation_index = (
                    seen_generated[
                        canonical_smiles
                    ]
                )

            else:
                is_unique_generated = True

            membership = get_membership(
                canonical_smiles,
                reference_info,
            )

            # SDF title line.
            candidate.SetProp(
                "_Name",
                f"{index:05d}",
            )

            candidate.SetProp(
                "generation_index",
                str(index),
            )

            candidate.SetProp(
                "target_raw_property",
                str(raw_target),
            )

            candidate.SetProp(
                "num_atoms",
                str(sampled.num_atoms),
            )

            candidate.SetProp(
                "sanitize_ok",
                str(sanitize_ok),
            )

            candidate.SetProp(
                "single_component",
                str(single_component),
            )

            candidate.SetProp(
                "canonical_smiles",
                canonical_smiles,
            )

            candidate.SetProp(
                "is_unique_generated",
                str(is_unique_generated),
            )

            set_optional_sdf_property(
                candidate,
                "duplicate_of_generation_index",
                duplicate_of_generation_index,
            )

            for (
                property_name_in_sdf,
                property_value,
            ) in membership.items():
                set_optional_sdf_property(
                    candidate,
                    property_name_in_sdf,
                    property_value,
                )

            sdf_path = write_single_molecule_sdf(
                molecule=candidate,
                generation_index=index,
                output_sdf_dir=output_sdf_dir,
            )

            written_to_sdf = True
            valid_count += 1

            if is_unique_generated:
                seen_generated[
                    canonical_smiles
                ] = index

                unique_valid_count += 1

            else:
                generated_duplicate_count += 1

            if membership[
                "in_train_dataset"
            ]:
                in_train_count += 1

            if membership[
                "in_val_dataset"
            ]:
                in_val_count += 1

            if membership[
                "in_test_dataset"
            ]:
                in_test_count += 1

            if membership[
                "in_any_dataset"
            ]:
                in_any_count += 1

            if membership[
                "is_novel"
            ]:
                novel_count += 1

        except Exception as exc:
            if (
                sdf_path is not None
                and not written_to_sdf
            ):
                sdf_path.unlink(
                    missing_ok=True
                )

                sdf_path = None

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        rows.append(
            {
                "generation_index": index,
                "target_raw_property": raw_target,
                "num_atoms": sampled.num_atoms,
                "valid": written_to_sdf,
                "rdkit_mol_created": (
                    rdkit_mol_created
                ),
                "has_unresolved_mask": (
                    has_unresolved_mask
                ),
                "atom_mask_remaining": (
                    masks["atom"]
                ),
                "charge_mask_remaining": (
                    masks["charge"]
                ),
                "bond_mask_remaining": (
                    masks["bond"]
                ),
                "sanitize_ok": sanitize_ok,
                "single_component": (
                    single_component
                ),
                "smiles_ok": smiles_ok,
                "written_to_sdf": (
                    written_to_sdf
                ),
                "sdf_filename": (
                    sdf_path.name
                    if sdf_path is not None
                    else None
                ),
                "sdf_path": (
                    str(sdf_path)
                    if sdf_path is not None
                    else None
                ),
                "smiles": canonical_smiles,
                "canonical_smiles": (
                    canonical_smiles
                ),
                "is_unique_generated": (
                    is_unique_generated
                ),
                "duplicate_of_generation_index": (
                    duplicate_of_generation_index
                ),
                **membership,
                "error": error,
            }
        )

    fieldnames = list(
        rows[0].keys()
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        csv_writer.writeheader()

        csv_writer.writerows(
            rows
        )

    grid_path = None
    if not args.no_grid:
        grid_path = draw_generated_molecule_grid(
            rows=rows,
            output_png=output_dir / f"generated_grid{args.grid_count}.png",
            max_molecules=args.grid_count,
            mols_per_row=args.grid_mols_per_row,
            sub_image_size=(args.grid_width, args.grid_height),
        )

    summary = {
        "requested": args.n_mols,
        "generated": len(molecules),
        "valid_written": valid_count,
        "valid_fraction": (
            valid_count
            / len(molecules)
        ),
        "generated_unique_valid": (
            unique_valid_count
        ),
        "generated_duplicate_valid": (
            generated_duplicate_count
        ),
        "target_raw_property": target,
        "property_name": property_name,
        "checkpoint": str(checkpoint),
        "output_sdf_dir": str(
            output_sdf_dir
        ),
        "sdf_files_written": valid_count,
        "sdf_filename_format": (
            "str(generation_index).zfill(5) "
            "+ '.sdf'"
        ),
        "output_csv": str(output_csv),
        "grid_image": (
            str(grid_path)
            if grid_path is not None
            else None
        ),
        "atom_count_source": (
            "fixed --n-atoms-per-mol"
            if args.n_atoms_per_mol is not None
            else "--number-of-atoms-file"
            if number_of_atoms is not None
            else "train_data_n_atoms_histogram.pt"
        ),
        "dataset_membership_check": {
            "enabled": (
                reference_info is not None
            ),
            "comparison_key": (
                "canonical isomeric SMILES"
            ),
            "processed_dir": str(
                processed_dir
            ),
            "reference_split_counts": (
                reference_info[
                    "split_counts"
                ]
                if reference_info is not None
                else None
            ),
            "reference_all_count": (
                reference_info[
                    "all_count"
                ]
                if reference_info is not None
                else None
            ),
            "missing_splits": (
                reference_info[
                    "missing_splits"
                ]
                if reference_info is not None
                else []
            ),
            "in_train_dataset": (
                in_train_count
            ),
            "in_val_dataset": (
                in_val_count
            ),
            "in_test_dataset": (
                in_test_count
            ),
            "in_any_dataset": (
                in_any_count
            ),
            "novel_valid": novel_count,
        },
        "sampling": {
            "n_timesteps": (
                args.n_timesteps
            ),
            "guide_w": guide_w,
            "dfm_type": (
                args.dfm_type
            ),
            "guidance_format": (
                args.guidance_format
            ),
            "where_to_apply_guide": (
                args.where_to_apply_guide
            ),
        },
    }

    if args.analyze:
        try:
            summary["analysis"] = (
                SampleAnalyzer()
                .analyze(molecules)
            )

        except Exception as exc:
            summary["analysis_error"] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
