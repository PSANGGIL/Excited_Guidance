import torch

from prepare_kekule_dataset import _kekule_edge_marginal, kekulize_processed_data


def _processed_ring(labels):
    n_atoms = 6
    return {
        "atom_types": torch.ones(n_atoms, 1),
        "atom_charges": torch.zeros(n_atoms, dtype=torch.long),
        "bond_types": torch.tensor(labels, dtype=torch.long),
        "bond_idxs": torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
            dtype=torch.long,
        ),
        "node_idx_array": torch.tensor([[0, n_atoms]], dtype=torch.long),
        "edge_idx_array": torch.tensor([[0, 6]], dtype=torch.long),
        "atom_map": ["C"],
    }


def test_benzene_aromatic_labels_become_alternating_single_double():
    source = _processed_ring([4] * 6)
    converted = kekulize_processed_data(source)

    assert source["bond_types"].tolist() == [4] * 6
    assert sorted(converted["bond_types"].tolist()) == [1, 1, 1, 2, 2, 2]
    assert converted["bond_representation"] == "kekule"
    assert converted["n_bond_types"] == 4


def test_existing_kekule_bonds_remain_unchanged():
    source = _processed_ring([1, 2, 1, 2, 1, 2])
    converted = kekulize_processed_data(source)
    assert torch.equal(converted["bond_types"], source["bond_types"])


def test_edge_marginal_includes_implicit_no_bond_pairs():
    converted = kekulize_processed_data(_processed_ring([4] * 6))
    marginal = _kekule_edge_marginal(converted)
    assert torch.allclose(marginal, torch.tensor([9 / 15, 3 / 15, 3 / 15, 0.0]))
