#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Iterator, Sequence

import dgl
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from rdkit import Chem
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Sampler

from model import PROPERTY_MAP, coupled_node_prior, edge_prior


REQUIRED_DATA_KEYS = (
    "positions",
    "atom_types",
    "atom_charges",
    "bond_types",
    "bond_idxs",
    "node_idx_array",
    "edge_idx_array",
)


def _torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _rank_and_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def collate(graphs: Sequence[dgl.DGLGraph]) -> dgl.DGLGraph:
    if not graphs:
        raise ValueError("Cannot collate an empty graph list")
    batched_graph = dgl.batch(graphs)
    if hasattr(graphs[0], "prop"):
        props = torch.stack([graph.prop for graph in graphs])
        batched_graph.prop = props.to(batched_graph.device)
    return batched_graph


def resolve_atom_type_map(atom_map) -> list[str]:
    """Return atom symbols in the exact class-index order used by the data."""
    if isinstance(atom_map, dict):
        values = list(atom_map.values())
        if values and all(isinstance(value, int) for value in values):
            return [
                str(symbol)
                for symbol, _ in sorted(atom_map.items(), key=lambda item: item[1])
            ]
        if atom_map and all(str(key).isdigit() for key in atom_map):
            return [
                str(atom_map[key])
                for key in sorted(atom_map, key=lambda key: int(key))
            ]
        raise ValueError("atom_map must map element->index or index->element")
    if isinstance(atom_map, (list, tuple)):
        return [str(symbol) for symbol in atom_map]
    raise TypeError("atom_map must be a list, tuple, or dict")


def resolve_train_split(dataset_config: dict) -> str:
    if "train_split" in dataset_config:
        return str(dataset_config["train_split"])
    if "use_first_half_training_set" in dataset_config:
        return "train_a" if dataset_config["use_first_half_training_set"] else "train_b"
    return "train"


def _infer_atom_type_map(data_dict: dict, data_file: Path) -> list[str]:
    """Infer the class-index -> element mapping from database-derived metadata.

    Preferred input is ``atom_map`` stored by preprocessing.  If it is absent,
    the mapping is reconstructed from per-atom atomic numbers and the one-hot
    atom class tensor.  This makes the processed database, not YAML, the source
    of truth while preserving the exact class order used for training.
    """
    stored_map = data_dict.get("atom_map")
    if stored_map is not None:
        return resolve_atom_type_map(stored_map)

    atom_types = data_dict.get("atom_types")
    if not torch.is_tensor(atom_types) or atom_types.ndim != 2:
        raise ValueError(f"{data_file} must contain a 2-D atom_types tensor")

    atomic_numbers = None
    for key in ("atomic_numbers", "atom_numbers", "atomic_number"):
        value = data_dict.get(key)
        if torch.is_tensor(value):
            atomic_numbers = value.reshape(-1).to(torch.long)
            break

    if atomic_numbers is None:
        raise KeyError(
            f"Cannot infer element classes from {data_file}. Store either atom_map "
            "or per-atom atomic_numbers in the processed database file."
        )
    if atomic_numbers.numel() != atom_types.shape[0]:
        raise ValueError(
            f"atomic_numbers length does not match atom_types rows in {data_file}: "
            f"{atomic_numbers.numel()} != {atom_types.shape[0]}"
        )

    class_indices = atom_types.argmax(dim=-1)
    periodic_table = Chem.GetPeriodicTable()
    resolved: list[str] = []
    for class_index in range(atom_types.shape[1]):
        values = torch.unique(atomic_numbers[class_indices == class_index]).tolist()
        if len(values) != 1:
            raise ValueError(
                f"Atom class {class_index} in {data_file} maps to atomic numbers "
                f"{values}; each one-hot class must represent exactly one element."
            )
        atomic_number = int(values[0])
        if atomic_number <= 0:
            raise ValueError(f"Invalid atomic number {atomic_number} in {data_file}")
        resolved.append(periodic_table.GetElementSymbol(atomic_number))
    return resolved


def materialize_dataset_metadata(config: dict, split: str | None = None) -> dict:
    """Read model-critical metadata from the processed database in-place.

    Users no longer specify ``dataset.atom_map``.  The train split determines
    the element vocabulary and class order.  Any legacy explicit atom map is
    accepted only as a consistency check and never overrides the database.
    """
    if "dataset" not in config or not isinstance(config["dataset"], dict):
        raise KeyError("Config must contain a dataset mapping")

    dataset_config = config["dataset"]
    processed_dir = Path(dataset_config["processed_data_dir"]).expanduser().resolve()
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")

    resolved_split = split or resolve_train_split(dataset_config)
    data_file = processed_dir / f"{resolved_split}_data_processed.pt"
    if not data_file.is_file():
        raise FileNotFoundError(f"Processed split file not found: {data_file}")

    data_dict = _torch_load(data_file)
    atom_types = data_dict.get("atom_types")
    if not torch.is_tensor(atom_types) or atom_types.ndim != 2:
        raise ValueError(f"{data_file} must contain a 2-D atom_types tensor")

    resolved_map = _infer_atom_type_map(data_dict, data_file)
    if len(resolved_map) != int(atom_types.shape[1]):
        raise ValueError(
            "atom_types class dimension does not match the database-derived atom map: "
            f"{atom_types.shape[1]} != {len(resolved_map)} ({resolved_map})"
        )

    legacy_map = dataset_config.get("atom_map")
    if legacy_map not in (None, "", "auto", "from_data", "from-data"):
        if resolve_atom_type_map(legacy_map) != resolved_map:
            raise ValueError(
                "Legacy dataset.atom_map conflicts with the processed database: "
                f"config={resolve_atom_type_map(legacy_map)}, database={resolved_map}"
            )

    dataset_config["processed_data_dir"] = str(processed_dir)
    dataset_config["atom_map"] = resolved_map
    dataset_config["metadata_source"] = str(data_file)

    conditioning = dataset_config.setdefault("conditioning", {})
    for key in ("target_transform", "target_epsilon", "raw_property_column"):
        if data_dict.get(key) is not None:
            conditioning[key] = data_dict[key]
    if data_dict.get("property_names") is not None:
        conditioning["property_names"] = list(data_dict["property_names"])
    return config


class MoleculeDataset(torch.utils.data.Dataset):
    """Load MolGuidance processed PT files without changing graph semantics."""

    def __init__(self, split: str, dataset_config: dict, prior_config: dict):
        super().__init__()
        self.split = split
        self.dataset_config = copy.deepcopy(dataset_config)
        self.prior_config = copy.deepcopy(prior_config)
        self.conditioning = self.dataset_config.get("conditioning", {})
        self.conditioning_enabled = bool(self.conditioning.get("enabled", False))
        self.target_property = self.conditioning.get("property")
        self.normalize_property = bool(self.conditioning.get("normalize", True))
        self.n_bond_types = int(self.dataset_config.get("n_bond_types", 5))
        self.n_atom_charges = int(self.dataset_config.get("n_atom_charges", 6))
        self.charge_offset = int(self.dataset_config.get("charge_offset", 2))
        self.atom_type_map = resolve_atom_type_map(self.dataset_config["atom_map"])
        self.validate_data = bool(self.dataset_config.get("validate_processed_data", True))

        processed_data_dir = Path(
            self.dataset_config["processed_data_dir"]
        ).expanduser().resolve()
        if not processed_data_dir.is_dir():
            raise FileNotFoundError(
                f"Processed data directory not found: {processed_data_dir}"
            )
        self.processed_data_dir = processed_data_dir

        marginal_path = processed_data_dir / "train_data_marginal_dists.pt"
        if not marginal_path.is_file():
            raise FileNotFoundError(f"Marginal distribution file not found: {marginal_path}")
        marginal_data = _torch_load(marginal_path)
        if not isinstance(marginal_data, (tuple, list)) or len(marginal_data) != 4:
            raise ValueError(
                "train_data_marginal_dists.pt must contain "
                "(p_a, p_c, p_e, p_c_given_a)"
            )
        p_a, p_c, p_e, p_c_given_a = marginal_data
        self._inject_marginals(p_a, p_c, p_e, p_c_given_a)

        self.data_file = processed_data_dir / f"{split}_data_processed.pt"
        if not self.data_file.is_file():
            raise FileNotFoundError(f"Processed split file not found: {self.data_file}")
        data_dict = _torch_load(self.data_file)
        missing = [key for key in REQUIRED_DATA_KEYS if key not in data_dict]
        if missing:
            raise KeyError(f"Missing keys in {self.data_file}: {missing}")

        self.positions = data_dict["positions"]
        self.atom_types = data_dict["atom_types"]
        self.atom_charges = data_dict["atom_charges"]
        self.bond_types = data_dict["bond_types"]
        self.bond_idxs = data_dict["bond_idxs"]
        self.node_idx_array = data_dict["node_idx_array"]
        self.edge_idx_array = data_dict["edge_idx_array"]
        self.dataset = self.dataset_config.get("dataset_name", "csvmol")
        self.data_metadata = {
            key: data_dict.get(key)
            for key in (
                "property_names",
                "raw_property_column",
                "target_transform",
                "target_epsilon",
                "atom_map",
                "geometry_source",
                "add_hydrogens",
            )
        }

        self.properties = None
        self.property_idx: int | None = None
        self.norm_params = None
        if self.conditioning_enabled and self.target_property:
            self._configure_property(data_dict)

        self.atom_counts = (
            self.node_idx_array[:, 1].to(torch.long)
            - self.node_idx_array[:, 0].to(torch.long)
        )
        if self.validate_data:
            self._validate_processed_data()

    def _inject_marginals(self, p_a, p_c, p_e, p_c_given_a) -> None:
        for feature in ("a", "c", "e"):
            if feature not in self.prior_config:
                raise KeyError(f"Missing prior_config entry for feature {feature!r}")
            self.prior_config[feature].setdefault("kwargs", {})
        if self.prior_config["a"]["type"] == "marginal":
            self.prior_config["a"]["kwargs"]["p"] = p_a
        if self.prior_config["e"]["type"] == "marginal":
            self.prior_config["e"]["kwargs"]["p"] = p_e
        if self.prior_config["c"]["type"] == "marginal":
            self.prior_config["c"]["kwargs"]["p"] = p_c
        if self.prior_config["c"]["type"] == "c-given-a":
            self.prior_config["c"]["kwargs"]["p_c_given_a"] = p_c_given_a

    def _configure_property(self, data_dict: dict) -> None:
        if "properties" not in data_dict:
            raise ValueError(f"Properties not found in {self.data_file}")
        self.properties = data_dict["properties"]
        if self.properties.ndim > 1:
            property_names = data_dict.get("property_names")
            if (
                isinstance(property_names, (list, tuple))
                and self.target_property in property_names
            ):
                self.property_idx = list(property_names).index(self.target_property)
            elif self.target_property in PROPERTY_MAP:
                self.property_idx = int(PROPERTY_MAP[self.target_property])
            elif self.properties.shape[-1] == 1:
                self.property_idx = 0
            else:
                raise ValueError(
                    f"Cannot resolve property {self.target_property!r} in property "
                    f"tensor with shape {tuple(self.properties.shape)}"
                )

        if self.normalize_property:
            norm_file = self.processed_data_dir / "train_data_property_normalization.pt"
            if not norm_file.is_file():
                raise FileNotFoundError(f"Normalization file not found: {norm_file}")
            self.norm_params = _torch_load(norm_file)
            if "mean" not in self.norm_params or "std" not in self.norm_params:
                raise KeyError("Normalization file must contain mean and std")

            data_transform = data_dict.get("target_transform")
            norm_transform = self.norm_params.get("target_transform")
            if data_transform is not None:
                if norm_transform is None:
                    raise KeyError(
                        "train_data_property_normalization.pt must store "
                        "target_transform when processed data uses a transformed target"
                    )
                if str(data_transform).lower() != str(norm_transform).lower():
                    raise ValueError(
                        "Property transform mismatch between processed data and "
                        f"normalization metadata: {data_transform!r} != {norm_transform!r}"
                    )

            data_epsilon = data_dict.get("target_epsilon")
            norm_epsilon = self.norm_params.get("target_epsilon")
            if data_epsilon is not None:
                if norm_epsilon is None:
                    raise KeyError(
                        "train_data_property_normalization.pt must store target_epsilon"
                    )
                if not math.isclose(
                    float(data_epsilon),
                    float(norm_epsilon),
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError(
                        "Property epsilon mismatch between processed data and "
                        f"normalization metadata: {data_epsilon!r} != {norm_epsilon!r}"
                    )

    def _validate_processed_data(self) -> None:
        n_molecules = int(self.node_idx_array.shape[0])
        if self.node_idx_array.ndim != 2 or self.node_idx_array.shape[1] != 2:
            raise ValueError("node_idx_array must have shape [N, 2]")
        if self.edge_idx_array.ndim != 2 or self.edge_idx_array.shape != self.node_idx_array.shape:
            raise ValueError("edge_idx_array must have shape [N, 2]")
        if self.positions.ndim != 2 or self.positions.shape[-1] != 3:
            raise ValueError("positions must have shape [total_atoms, 3]")
        if not bool(torch.isfinite(self.positions).all()):
            bad_count = int(torch.count_nonzero(~torch.isfinite(self.positions)))
            raise ValueError(f"positions contains {bad_count} NaN/Inf values")
        if self.atom_types.ndim != 2:
            raise ValueError("atom_types must be a 2-D one-hot tensor")
        if self.atom_types.shape[1] != len(self.atom_type_map):
            raise ValueError(
                "atom_types class dimension does not match dataset.atom_map: "
                f"{self.atom_types.shape[1]} != {len(self.atom_type_map)}"
            )
        total_atoms = self.positions.shape[0]
        if self.atom_types.shape[0] != total_atoms or self.atom_charges.shape[0] != total_atoms:
            raise ValueError("positions, atom_types, and atom_charges lengths differ")
        if self.bond_idxs.ndim != 2 or self.bond_idxs.shape[1] != 2:
            raise ValueError("bond_idxs must have shape [total_bonds, 2]")
        if self.bond_types.shape[0] != self.bond_idxs.shape[0]:
            raise ValueError("bond_types and bond_idxs lengths differ")
        if n_molecules == 0:
            raise ValueError(f"Split {self.split!r} contains no molecules")

        node_starts = self.node_idx_array[:, 0].to(torch.long)
        node_ends = self.node_idx_array[:, 1].to(torch.long)
        edge_starts = self.edge_idx_array[:, 0].to(torch.long)
        edge_ends = self.edge_idx_array[:, 1].to(torch.long)
        if torch.any(node_starts < 0) or torch.any(node_ends < node_starts):
            raise ValueError("Invalid node index intervals")
        if torch.any(edge_starts < 0) or torch.any(edge_ends < edge_starts):
            raise ValueError("Invalid edge index intervals")
        if int(node_ends.max()) > total_atoms:
            raise ValueError("node_idx_array exceeds flattened atom tensor")
        if int(edge_ends.max()) > self.bond_idxs.shape[0]:
            raise ValueError("edge_idx_array exceeds flattened bond tensor")
        if n_molecules > 1:
            if not torch.equal(node_starts[1:], node_ends[:-1]):
                raise ValueError("node_idx_array intervals are not contiguous")
            if not torch.equal(edge_starts[1:], edge_ends[:-1]):
                raise ValueError("edge_idx_array intervals are not contiguous")

        active_counts = self.atom_types.to(torch.int16).sum(dim=-1)
        if not bool(torch.all(active_counts == 1)):
            bad_count = int(torch.count_nonzero(active_counts != 1))
            raise ValueError(f"atom_types contains {bad_count} non one-hot rows")

        charge_indices = self.atom_charges.to(torch.long) + self.charge_offset
        if charge_indices.numel() and (
            int(charge_indices.min()) < 0
            or int(charge_indices.max()) >= self.n_atom_charges
        ):
            raise ValueError(
                "Formal charge outside configured one-hot range: "
                f"raw=[{int(self.atom_charges.min())}, {int(self.atom_charges.max())}], "
                f"offset={self.charge_offset}, classes={self.n_atom_charges}"
            )
        if self.bond_types.numel() and (
            int(self.bond_types.min()) < 0
            or int(self.bond_types.max()) >= self.n_bond_types
        ):
            raise ValueError(
                f"bond_types values must be in [0, {self.n_bond_types - 1}]"
            )

        if self.properties is not None and self.properties.shape[0] != n_molecules:
            raise ValueError("Property count does not match molecule count")
        if self.properties is not None:
            selected_properties = self.properties
            if self.property_idx is not None:
                selected_properties = selected_properties[..., self.property_idx]
            if not bool(torch.isfinite(selected_properties).all()):
                bad_count = int(
                    torch.count_nonzero(~torch.isfinite(selected_properties))
                )
                raise ValueError(
                    f"Conditioning properties contain {bad_count} NaN/Inf values"
                )
        if self.properties is not None and self.normalize_property:
            mean, std = self._normalization_values()
            if not torch.isfinite(torch.as_tensor(mean)) or not torch.isfinite(torch.as_tensor(std)):
                raise ValueError("Normalization mean/std must be finite")
            if float(torch.as_tensor(std)) <= 0:
                raise ValueError("Normalization std must be positive")

    def _normalization_values(self):
        mean = self.norm_params["mean"]
        std = self.norm_params["std"]
        if torch.is_tensor(mean) and mean.ndim > 0:
            if self.property_idx is not None:
                mean, std = mean[self.property_idx], std[self.property_idx]
            elif mean.numel() == 1:
                mean, std = mean.reshape(()), std.reshape(())
            else:
                raise ValueError(
                    "Vector normalization requires a resolved target property index"
                )
        return mean, std

    def __len__(self) -> int:
        return int(self.node_idx_array.shape[0])

    def _property_value(self, idx: int) -> torch.Tensor:
        value = self.properties[idx]
        if self.property_idx is not None:
            value = value[self.property_idx]
        value = value.float()
        if self.normalize_property:
            mean, std = self._normalization_values()
            value = (value - mean) / std
        return value

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        node_start = int(self.node_idx_array[idx, 0])
        node_end = int(self.node_idx_array[idx, 1])
        edge_start = int(self.edge_idx_array[idx, 0])
        edge_end = int(self.edge_idx_array[idx, 1])

        positions = self.positions[node_start:node_end].float()
        atom_types = self.atom_types[node_start:node_end].float()
        raw_charges = self.atom_charges[node_start:node_end].long()
        if positions.shape[0] == 0:
            raise ValueError(f"Molecule {idx} has zero atoms")
        positions = positions - positions.mean(dim=0, keepdim=True)

        bond_types = self.bond_types[edge_start:edge_end].long()
        bond_idxs = self.bond_idxs[edge_start:edge_end].long()
        n_atoms = int(positions.shape[0])
        adjacency = torch.zeros((n_atoms, n_atoms), dtype=torch.long)
        if bond_idxs.numel():
            # Official data are molecule-local. Some custom preprocessors stored
            # flattened/global indices, which are accepted after subtracting node_start.
            if int(bond_idxs.max()) >= n_atoms:
                bond_idxs = bond_idxs - node_start
            if int(bond_idxs.min()) < 0 or int(bond_idxs.max()) >= n_atoms:
                raise IndexError(f"Invalid bond indices for molecule {idx}")
            if torch.any(bond_idxs[:, 0] == bond_idxs[:, 1]):
                raise ValueError(f"Self bond found in molecule {idx}")
            adjacency[bond_idxs[:, 0], bond_idxs[:, 1]] = bond_types
            adjacency[bond_idxs[:, 1], bond_idxs[:, 0]] = bond_types

        upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
        upper_edge_labels = adjacency[upper_edge_idxs[0], upper_edge_idxs[1]]
        lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))
        edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
        edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

        edge_one_hot = one_hot(
            edge_labels.to(torch.int64), num_classes=self.n_bond_types
        ).float()
        charge_one_hot = one_hot(
            raw_charges + self.charge_offset, num_classes=self.n_atom_charges
        ).float()

        graph = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)
        graph.edata["e_1_true"] = edge_one_hot
        graph.ndata["x_1_true"] = positions
        graph.ndata["a_1_true"] = atom_types
        graph.ndata["c_1_true"] = charge_one_hot

        if self.conditioning_enabled and self.target_property:
            graph.prop = self._property_value(idx).to(graph.device)

        destination = {"x": positions, "a": atom_types, "c": charge_one_hot}
        prior_node_features = coupled_node_prior(
            dst_dict=destination, prior_config=self.prior_config
        )
        for feature, value in prior_node_features.items():
            graph.ndata[f"{feature}_0"] = value

        upper_edge_mask = torch.zeros(graph.num_edges(), dtype=torch.bool)
        upper_edge_mask[: upper_edge_idxs.shape[1]] = True
        graph.edata["e_0"] = edge_prior(upper_edge_mask, self.prior_config["e"])
        return graph


class SizeAwareBatchSampler(Sampler[list[int]]):
    """Group equal-size molecules and enforce both molecule and edge limits."""

    def __init__(
        self,
        dataset: MoleculeDataset,
        batch_size: int,
        max_num_edges: int,
        shuffle: bool,
        seed: int = 42,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_num_edges <= 0:
            raise ValueError("max_num_edges must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.max_num_edges = int(max_num_edges)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.size_to_indices = {
            int(size): torch.where(dataset.atom_counts == size)[0].tolist()
            for size in torch.unique(dataset.atom_counts)
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batch_capacity(self, n_atoms: int) -> int:
        directed_edges = max(n_atoms * (n_atoms - 1), 1)
        return max(1, min(self.batch_size, self.max_num_edges // directed_edges))

    def _all_batches(self) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        size_items = sorted(self.size_to_indices.items())
        if self.shuffle and len(size_items) > 1:
            order = torch.randperm(len(size_items), generator=generator).tolist()
            size_items = [size_items[index] for index in order]

        for n_atoms, original_indices in size_items:
            indices = list(original_indices)
            if self.shuffle and len(indices) > 1:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in permutation]
            capacity = self._batch_capacity(n_atoms)
            for start in range(0, len(indices), capacity):
                batch = indices[start : start + capacity]
                if self.drop_last and len(batch) < capacity:
                    continue
                batches.append(batch)

        if self.shuffle and len(batches) > 1:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in permutation]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        rank, world_size = _rank_and_world_size()
        batches = self._all_batches()
        # Match DistributedSampler semantics: every rank receives the same
        # number of steps so DDP cannot hang on the final synchronization.
        per_rank = math.ceil(len(batches) / max(world_size, 1))
        total_size = per_rank * max(world_size, 1)
        if batches and len(batches) < total_size:
            repeats = total_size - len(batches)
            batches = batches + [batches[i % len(batches)] for i in range(repeats)]
        batches = batches[rank:total_size:world_size]
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        total_batches = 0
        for n_atoms, indices in self.size_to_indices.items():
            capacity = self._batch_capacity(n_atoms)
            if self.drop_last:
                total_batches += len(indices) // capacity
            else:
                total_batches += math.ceil(len(indices) / capacity)
        _, world_size = _rank_and_world_size()
        return math.ceil(total_batches / max(world_size, 1))


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_config: dict,
        dm_prior_config: dict,
        batch_size: int,
        num_workers: int = 0,
        distributed: bool = False,
        max_num_edges: int = 40000,
        max_num_edges_eval: int | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.distributed = distributed
        self.dataset_config = copy.deepcopy(dataset_config)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prior_config = copy.deepcopy(dm_prior_config)
        self.max_num_edges = int(max_num_edges)
        self.max_num_edges_eval = int(max_num_edges_eval or max_num_edges)
        self.seed = int(seed)
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit") and not hasattr(self, "train_dataset"):
            self.train_dataset = self.load_dataset(resolve_train_split(self.dataset_config))
            self.val_dataset = self.load_dataset("val")
            self._validate_split_compatibility(self.train_dataset, self.val_dataset)
        if stage in (None, "test", "predict") and not hasattr(self, "test_dataset"):
            self.test_dataset = self.load_dataset("test")
            if hasattr(self, "train_dataset"):
                self._validate_split_compatibility(self.train_dataset, self.test_dataset)

    @staticmethod
    def _validate_split_compatibility(reference: MoleculeDataset, other: MoleculeDataset) -> None:
        """Prevent silent class/property mismatches between train/val/test splits."""
        other_map = other.data_metadata.get("atom_map")
        if other_map is not None and resolve_atom_type_map(other_map) != reference.atom_type_map:
            raise ValueError(
                f"Atom-class order differs between {reference.split} and {other.split}: "
                f"{reference.atom_type_map} != {resolve_atom_type_map(other_map)}"
            )

        for key in ("property_names", "raw_property_column", "target_transform", "target_epsilon"):
            ref_value = reference.data_metadata.get(key)
            other_value = other.data_metadata.get(key)
            if ref_value is not None and other_value is not None and ref_value != other_value:
                raise ValueError(
                    f"Property metadata {key!r} differs between {reference.split} "
                    f"and {other.split}: {ref_value!r} != {other_value!r}"
                )

    def load_dataset(self, split: str) -> MoleculeDataset:
        return MoleculeDataset(
            split,
            self.dataset_config,
            prior_config=copy.deepcopy(self.prior_config),
        )

    def _loader(
        self,
        dataset: MoleculeDataset,
        batch_size: int,
        max_num_edges: int,
        shuffle: bool,
    ) -> DataLoader:
        batch_sampler = SizeAwareBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            max_num_edges=max_num_edges,
            shuffle=shuffle,
            seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dataset,
            batch_size=self.batch_size,
            max_num_edges=self.max_num_edges,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            max_num_edges=self.max_num_edges_eval,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._loader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            max_num_edges=self.max_num_edges_eval,
            shuffle=False,
        )


def _configured_device_count(devices) -> int:
    if isinstance(devices, int):
        return max(devices, 1)
    if isinstance(devices, (list, tuple)):
        return max(len(devices), 1)
    if isinstance(devices, str):
        value = devices.strip().lower()
        if value in {"auto", "-1"}:
            return max(torch.cuda.device_count(), 1)
        if "," in value:
            return len([item for item in value.split(",") if item.strip()])
        try:
            return max(int(value), 1)
        except ValueError:
            return 1
    return 1


def data_module_from_config(config: dict, seed: int = 42) -> MoleculeDataModule:
    materialize_dataset_metadata(config)
    training = config["training"]
    devices = training.get("trainer_args", {}).get("devices", 1)
    return MoleculeDataModule(
        dataset_config=config["dataset"],
        dm_prior_config=config["mol_fm"]["prior_config"],
        batch_size=int(training["batch_size"]),
        num_workers=int(training.get("num_workers", 0)),
        distributed=_configured_device_count(devices) > 1,
        max_num_edges=int(training.get("max_num_edges", 40000)),
        max_num_edges_eval=int(
            training.get("max_num_edges_eval", training.get("max_num_edges", 40000))
        ),
        seed=seed,
    )
