import unittest

import dgl
import torch
from torch.nn import functional as F

from model import FlowMol


def diagnostic_harness() -> FlowMol:
    model = FlowMol.__new__(FlowMol)
    torch.nn.Module.__init__(model)
    model.exclude_charges = False
    model.charge_offset = 2
    # These tests use hydrogen only: every formal-charge class is capped at one.
    model.register_buffer(
        "_valence_caps",
        torch.ones((1, 6), dtype=torch.float32),
        persistent=False,
    )
    return model


def hydrogen_graph(n_atoms: int) -> tuple[dgl.DGLGraph, torch.Tensor]:
    src = []
    dst = []
    upper = []
    for atom_i in range(n_atoms):
        for atom_j in range(n_atoms):
            if atom_i == atom_j:
                continue
            src.append(atom_i)
            dst.append(atom_j)
            upper.append(atom_i < atom_j)
    graph = dgl.graph((src, dst), num_nodes=n_atoms)
    graph.edata["ue_mask"] = torch.tensor(upper, dtype=torch.bool)
    graph.edata["e_1_true"] = F.one_hot(
        torch.ones(len(src), dtype=torch.long),
        num_classes=5,
    ).float()
    graph.ndata["a_1_true"] = torch.ones((n_atoms, 1))
    graph.ndata["c_1_true"] = F.one_hot(
        torch.full((n_atoms,), 2),
        num_classes=6,
    ).float()
    graph.ndata["x_1_true"] = torch.zeros((n_atoms, 3))
    return graph, graph.edata["ue_mask"]


def confident_single_bond_logits(n_upper_edges: int) -> torch.Tensor:
    logits = torch.full((n_upper_edges, 5), -20.0)
    logits[:, 1] = 20.0
    return logits


class IntuitiveMetricTests(unittest.TestCase):
    def test_selected_valence_passes_for_hydrogen_molecule(self):
        model = diagnostic_harness()
        graph, upper = hydrogen_graph(2)
        logits = confident_single_bond_logits(int(upper.sum()))

        state = model._valence_diagnostic_state(graph, logits, upper)

        self.assertTrue(torch.allclose(state["selected_overflow"], torch.zeros(2)))

    def test_overbonded_hydrogen_has_positive_loss_and_gradient(self):
        model = diagnostic_harness()
        graph, upper = hydrogen_graph(3)
        logits = confident_single_bond_logits(int(upper.sum())).requires_grad_()

        loss = model._expected_valence_loss(graph, logits, upper)
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_intuitive_metrics_report_argmax_valence_failure(self):
        model = diagnostic_harness()
        graph, upper = hydrogen_graph(3)
        edge_logits = confident_single_bond_logits(int(upper.sum()))
        predictions = {
            "x": torch.zeros((3, 3)),
            "a": torch.full((3, 1), 10.0),
            "c": F.one_hot(torch.full((3,), 2), num_classes=6).float() * 10.0,
            "e": edge_logits,
        }
        targets = {
            "x": graph.ndata["x_1_true"],
            "a": torch.zeros(3, dtype=torch.long),
            "c": torch.full((3,), 2, dtype=torch.long),
            "e": torch.ones(int(upper.sum()), dtype=torch.long),
        }

        metrics = model._build_intuitive_validation_metrics(
            graph,
            predictions,
            targets,
            upper,
        )

        self.assertEqual(float(metrics["val_selected_molecule_valence_pass_rate"]), 0.0)
        self.assertEqual(float(metrics["val_selected_atom_valence_violation_rate"]), 1.0)
        self.assertEqual(float(metrics["val_e_masked_accuracy"]), 1.0)


if __name__ == "__main__":
    unittest.main()
