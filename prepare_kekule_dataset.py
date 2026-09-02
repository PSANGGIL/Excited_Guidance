#!/usr/bin/env python3
"""Create a separate processed dataset with aromatic bonds Kekulized.

The source tensors are never modified. Bond classes become:
0=no bond, 1=single, 2=double, 3=triple. RDKit receives the original atom
order so coordinates, atom features, molecule IDs, and properties stay aligned.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from rdkit import Chem


INPUT_BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
OUTPUT_BOND_LABELS = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}


def _atom_symbols(atom_map) -> list[str]:
    if isinstance(atom_map, (list, tuple)):
        return [str(value) for value in atom_map]
    if isinstance(atom_map, dict) and all(
        isinstance(value, int) for value in atom_map.values()
    ):
        return [
            str(symbol)
            for symbol, _ in sorted(atom_map.items(), key=lambda item: item[1])
        ]
    raise ValueError("Processed data must contain an ordered atom_map")


def kekulize_processed_data(data: dict, source_name: str = "processed data") -> dict:
    """Return a shallow copy whose stored aromatic bonds are single/double."""
    required = (
        "atom_types",
        "atom_charges",
        "bond_types",
        "bond_idxs",
        "node_idx_array",
        "edge_idx_array",
        "atom_map",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{source_name} is missing {missing}")

    symbols = _atom_symbols(data["atom_map"])
    converted = dict(data)
    converted_bonds = data["bond_types"].clone().to(torch.long)
    # The production preprocessing stores one-hot atom classes as bool tensors;
    # argmax is not implemented for bool on CPU.
    atom_classes = data["atom_types"].to(torch.uint8).argmax(dim=-1)

    for mol_idx, (node_range, edge_range) in enumerate(
        zip(data["node_idx_array"], data["edge_idx_array"])
    ):
        node_start, node_end = (int(value) for value in node_range)
        edge_start, edge_end = (int(value) for value in edge_range)
        n_atoms = node_end - node_start
        rw_mol = Chem.RWMol()
        for offset in range(n_atoms):
            atom = Chem.Atom(symbols[int(atom_classes[node_start + offset])])
            atom.SetFormalCharge(int(data["atom_charges"][node_start + offset]))
            rw_mol.AddAtom(atom)

        local_edges = data["bond_idxs"][edge_start:edge_end].to(torch.long).clone()
        if local_edges.numel() and int(local_edges.max()) >= n_atoms:
            local_edges -= node_start
        labels = data["bond_types"][edge_start:edge_end].to(torch.long)
        for edge, label_tensor in zip(local_edges, labels):
            src, dst = int(edge[0]), int(edge[1])
            label = int(label_tensor)
            if label not in INPUT_BOND_TYPES:
                raise ValueError(
                    f"{source_name} molecule {mol_idx} has unsupported bond label {label}"
                )
            rw_mol.AddBond(src, dst, INPUT_BOND_TYPES[label])
            if label == 4:
                rw_mol.GetAtomWithIdx(src).SetIsAromatic(True)
                rw_mol.GetAtomWithIdx(dst).SetIsAromatic(True)

        mol = rw_mol.GetMol()
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception as exc:
            raise ValueError(
                f"Kekulization failed for {source_name} molecule {mol_idx}"
            ) from exc

        for output_idx, edge in enumerate(local_edges, start=edge_start):
            bond = mol.GetBondBetweenAtoms(int(edge[0]), int(edge[1]))
            label = OUTPUT_BOND_LABELS.get(bond.GetBondType())
            if label is None:
                raise ValueError(
                    f"Kekulization produced unsupported type {bond.GetBondType()} "
                    f"for {source_name} molecule {mol_idx}"
                )
            converted_bonds[output_idx] = label

    converted["bond_types"] = converted_bonds.to(data["bond_types"].dtype)
    converted["bond_representation"] = "kekule"
    converted["n_bond_types"] = 4
    return converted


def _kekule_edge_marginal(train_data: dict) -> torch.Tensor:
    counts = torch.bincount(train_data["bond_types"].long(), minlength=4).double()
    atom_counts = (
        train_data["node_idx_array"][:, 1] - train_data["node_idx_array"][:, 0]
    ).long()
    possible_edges = (atom_counts * (atom_counts - 1) // 2).sum().item()
    stored_edges = int(train_data["bond_types"].numel())
    if stored_edges > possible_edges:
        raise ValueError("Stored bond count exceeds complete undirected graph size")
    counts[0] += possible_edges - stored_edges
    return (counts / counts.sum()).float()


def convert_dataset(source_dir: Path, output_dir: Path) -> None:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("Output directory must differ from source directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_train = None
    split_files = sorted(source_dir.glob("*_data_processed.pt"))
    if not split_files:
        raise FileNotFoundError(f"No processed split files in {source_dir}")
    for source_file in split_files:
        data = torch.load(source_file, map_location="cpu", weights_only=False)
        converted = kekulize_processed_data(data, source_file.name)
        torch.save(converted, output_dir / source_file.name)
        if source_file.name == "train_data_processed.pt":
            converted_train = converted

    if converted_train is None:
        raise FileNotFoundError("train_data_processed.pt is required")
    old_marginals = torch.load(
        source_dir / "train_data_marginal_dists.pt",
        map_location="cpu",
        weights_only=False,
    )
    p_a, p_c, _, p_c_given_a = old_marginals
    torch.save(
        (p_a, p_c, _kekule_edge_marginal(converted_train), p_c_given_a),
        output_dir / "train_data_marginal_dists.pt",
    )

    for name in (
        "train_data_n_atoms_histogram.pt",
        "train_data_property_normalization.pt",
        "atom_map.json",
        "preprocessing_summary.json",
        "split_indices.json",
    ):
        source_file = source_dir / name
        if source_file.is_file():
            shutil.copy2(source_file, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    convert_dataset(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
