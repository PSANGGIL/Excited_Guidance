import tempfile
import unittest
from pathlib import Path

import dgl
import torch
from torch.nn import functional as F

from model import (
    CFGVectorField,
    ClassifierFreeGuidance,
    EndpointVectorField,
    FlowMol,
    InterpolantScheduler,
    build_edge_idxs,
)
from train import build_checkpoint_callbacks


def small_endpoint_vector_field(convs_per_update: int) -> EndpointVectorField:
    scheduler = InterpolantScheduler(
        canonical_feat_order=["x", "a", "c", "e"],
        schedule_type="linear",
    )
    return EndpointVectorField(
        n_atom_types=2,
        canonical_feat_order=["x", "a", "c", "e"],
        interpolant_scheduler=scheduler,
        n_charges=2,
        n_bond_types=5,
        n_vec_channels=4,
        n_hidden_scalars=16,
        n_hidden_edge_feats=12,
        n_recycles=1,
        n_molecule_updates=3,
        convs_per_update=convs_per_update,
        n_message_gvps=1,
        n_update_gvps=1,
        separate_mol_updaters=True,
        message_norm=10,
        rbf_dmax=8,
        rbf_dim=4,
    )


def small_complete_graph() -> tuple[dgl.DGLGraph, torch.Tensor]:
    edges = build_edge_idxs(3)
    graph = dgl.graph((edges[0], edges[1]), num_nodes=3)
    graph.ndata["x_t"] = torch.randn(3, 3)
    graph.ndata["a_t"] = F.one_hot(
        torch.tensor([0, 1, 0]),
        num_classes=2,
    ).float()
    graph.ndata["c_t"] = F.one_hot(
        torch.tensor([0, 0, 1]),
        num_classes=2,
    ).float()
    graph.edata["e_t"] = F.one_hot(
        torch.tensor([0, 1, 2, 0, 1, 2]),
        num_classes=5,
    ).float()
    upper = torch.zeros(graph.num_edges(), dtype=torch.bool)
    upper[: graph.num_edges() // 2] = True
    return graph, upper


class UpdaterExecutionTests(unittest.TestCase):
    def test_updater_indices_cover_every_configured_updater(self):
        for convs_per_update in (1, 2):
            vector_field = small_endpoint_vector_field(convs_per_update)
            observed = [
                vector_field.molecule_updater_index(conv_idx)
                for conv_idx in range(len(vector_field.conv_layers))
            ]
            observed = [index for index in observed if index is not None]
            self.assertEqual(observed, [0, 1, 2])

    def test_every_separate_updater_receives_gradient(self):
        for convs_per_update in (1, 2):
            with self.subTest(convs_per_update=convs_per_update):
                vector_field = small_endpoint_vector_field(convs_per_update)
                graph, upper = small_complete_graph()
                output = vector_field(
                    graph,
                    t=torch.tensor([0.4]),
                    node_batch_idx=torch.zeros(3, dtype=torch.long),
                    upper_edge_mask=upper,
                    remove_com=True,
                )
                loss = sum(value.square().mean() for value in output.values())
                loss.backward()

                modules = (
                    list(vector_field.node_position_updaters)
                    + list(vector_field.edge_updaters)
                )
                for module_idx, module in enumerate(modules):
                    gradients = [
                        parameter.grad
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ]
                    self.assertTrue(
                        any(gradient is not None for gradient in gradients),
                        f"updater module {module_idx} was not connected to loss",
                    )
                    self.assertTrue(
                        all(
                            gradient is None or bool(torch.isfinite(gradient).all())
                            for gradient in gradients
                        )
                    )


class CFGProbabilityTests(unittest.TestCase):
    def test_condition_margin_penalizes_insufficient_condition_gain(self):
        correct = torch.tensor(1.0, requires_grad=True)
        shuffled = torch.tensor(1.02, requires_grad=True)
        loss = ClassifierFreeGuidance.condition_margin_loss(
            correct, shuffled, margin=0.05
        )
        self.assertAlmostEqual(float(loss), 0.03, places=6)
        loss.backward()
        self.assertEqual(float(correct.grad), 1.0)
        self.assertEqual(float(shuffled.grad), -1.0)

    def test_condition_margin_is_zero_after_required_gain(self):
        loss = ClassifierFreeGuidance.condition_margin_loss(
            torch.tensor(1.0), torch.tensor(1.1), margin=0.05
        )
        self.assertEqual(float(loss), 0.0)

    def test_weight_one_recovers_conditional_distribution(self):
        uncond = torch.tensor([[1000.0, -1000.0, -1200.0]])
        cond = torch.tensor([[-900.0, 900.0, -1100.0]])
        temperature = 0.05
        expected = F.softmax(cond / temperature, dim=-1)

        for guidance_format in ("linear", "log"):
            actual = CFGVectorField.guided_probabilities_from_logits(
                uncond,
                cond,
                guide_weight=1.0,
                temperature=temperature,
                guidance_format=guidance_format,
            )
            self.assertTrue(bool(torch.isfinite(actual).all()))
            self.assertTrue(torch.allclose(actual.sum(-1), torch.ones(1)))
            self.assertTrue(torch.allclose(actual, expected))

    def test_extrapolative_guidance_remains_a_simplex(self):
        uncond = torch.tensor([[1000.0, -1000.0, -1200.0]])
        cond = torch.tensor([[-900.0, 900.0, -1100.0]])
        actual = CFGVectorField.guided_probabilities_from_logits(
            uncond,
            cond,
            guide_weight=3.0,
            temperature=0.01,
            guidance_format="linear",
        )
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(bool((actual >= 0).all()))
        self.assertTrue(torch.allclose(actual.sum(-1), torch.ones(1)))


class BondMetricTests(unittest.TestCase):
    def test_bonded_and_per_class_metrics_expose_no_bond_imbalance(self):
        model = FlowMol.__new__(FlowMol)
        torch.nn.Module.__init__(model)
        labels = torch.tensor([0, 1, 2, 3, 4, 1])
        predicted = torch.tensor([0, 1, 0, 3, 4, 2])
        logits = F.one_hot(predicted, num_classes=5).float() * 10.0

        metrics = model._bond_classification_metrics(logits, labels)

        self.assertAlmostEqual(float(metrics["val_e_bonded_precision"]), 1.0)
        self.assertAlmostEqual(float(metrics["val_e_bonded_recall"]), 0.8)
        self.assertAlmostEqual(
            float(metrics["val_e_single_recall"]),
            0.5,
        )
        self.assertEqual(float(metrics["val_e_double_f1"]), 0.0)
        self.assertIn("val_e_macro_f1", metrics)


class CheckpointCallbackTests(unittest.TestCase):
    def test_latest_checkpoint_is_independent_from_top_k(self):
        config = {
            "checkpointing": {
                "every_n_epochs": 1,
                "monitor": "val_cond_total_loss",
                "mode": "min",
                "save_last": True,
                "save_top_k": 3,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            best, latest = build_checkpoint_callbacks(Path(directory), config)

        self.assertFalse(best.save_last)
        self.assertEqual(best.save_top_k, 3)
        self.assertIsNotNone(latest)
        self.assertTrue(latest.save_last)
        self.assertEqual(latest.save_top_k, 0)
        self.assertIsNone(latest.monitor)


if __name__ == "__main__":
    unittest.main()
