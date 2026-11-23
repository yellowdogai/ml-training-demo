"""The ONNX export and the checks that gate publishing.

  python -m unittest discover tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "task"))

from onnx_export import export_onnx, onnx_config, predict_onnx, verify_onnx  # noqa: E402
from tabular_model import TabularRegressor, activation_name  # noqa: E402


def trained_like_model() -> tuple[TabularRegressor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = TabularRegressor(8, [16, 16]).eval()
    x = torch.randn(200, 8) * 3 + 5
    model.x_mean.copy_(x.mean(0))
    model.x_std.copy_(x.std(0))
    model.y_mean.fill_(2.0)
    model.y_std.fill_(1.5)
    with torch.no_grad():
        y = model(x) + 0.1 * torch.randn(200)
    return model, x, y


def rmse(model, x, y) -> float:
    with torch.no_grad():
        return torch.sqrt(torch.mean((model(x) - y) ** 2)).item()


class OnnxExportTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.model, self.x, self.y = trained_like_model()
        self.path = export_onnx(self.model, 8, self.directory)

    def test_graph_has_the_served_interface(self):
        graph = onnx.load(str(self.path))
        self.assertEqual([o.version for o in graph.opset_import if o.domain in ("", "ai.onnx")], [17])
        (inp,), (out,) = graph.graph.input, graph.graph.output
        self.assertEqual((inp.name, out.name), ("input", "output"))
        self.assertEqual(inp.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        self.assertEqual(out.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        in_dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
        out_dims = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
        self.assertEqual(in_dims, ["batch", 8])
        self.assertEqual(out_dims, ["batch", 1])
        self.assertTrue({n.domain for n in graph.graph.node} <= {"", "ai.onnx"}, "no custom ops")

    def test_checks_pass_and_report_parity(self):
        parity = verify_onnx(self.path, self.model, self.x, self.y, rmse(self.model, self.x, self.y))
        self.assertLess(parity["max_abs_diff_vs_pytorch"], 1e-4)
        self.assertEqual(parity["batch_sizes_run"], [1, 1000])

    def test_scaling_is_inside_the_graph(self):
        # Raw units in, target units out: matches the PyTorch model, which carries its own scaling.
        rows = self.x[:3].tolist()
        with torch.no_grad():
            expected = self.model(self.x[:3]).tolist()
        for got, want in zip(predict_onnx(self.path, rows), expected):
            self.assertAlmostEqual(got, want, places=4)

    def test_a_wrong_rmse_fails_the_checks(self):
        with self.assertRaisesRegex(ValueError, "RMSE from ONNX"):
            verify_onnx(self.path, self.model, self.x, self.y, rmse(self.model, self.x, self.y) + 1e-3)

    def test_an_export_that_disagrees_with_pytorch_fails_the_checks(self):
        with torch.no_grad():
            self.model.y_mean += 0.01  # the PyTorch side changes; the exported graph does not
        with self.assertRaisesRegex(ValueError, "differ by up to"):
            verify_onnx(self.path, self.model, self.x, self.y, rmse(self.model, self.x, self.y))

    def test_config_entry_uses_kserve_spelling(self):
        self.assertEqual(
            onnx_config(8),
            {
                "file": "model.onnx",
                "opset": 17,
                "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 8]}],
                "outputs": [{"name": "output", "datatype": "FP32", "shape": [-1, 1]}],
            },
        )

    def test_activation_is_read_from_the_network(self):
        self.assertEqual(activation_name(self.model), "relu")


if __name__ == "__main__":
    unittest.main()
