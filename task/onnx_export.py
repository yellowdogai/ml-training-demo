"""Export the trained model to ONNX, and check the export before it is published.

model.onnx holds the whole computation, scaling included, so a standard model
server (Triton, MLServer) can serve it over the KServe v2 protocol without this
project's code: raw features in, in the order of config.json, and the target out
in its own units.
"""

import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn

from tabular_model import TabularRegressor

ONNX_FILE = "model.onnx"
OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "output"


class ServedModel(nn.Module):
    """The model as served: [batch, features] raw units in, [batch, 1] target units out."""

    def __init__(self, model: TabularRegressor):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TabularRegressor.forward scales, runs the MLP and unscales; only the shape changes here.
        return self.model(x).unsqueeze(-1)


def onnx_config(n_features: int) -> dict:
    """The "onnx" entry of config.json, with KServe v2 datatypes and -1 for the batch axis."""
    return {
        "file": ONNX_FILE,
        "opset": OPSET,
        "inputs": [{"name": INPUT_NAME, "datatype": "FP32", "shape": [-1, n_features]}],
        "outputs": [{"name": OUTPUT_NAME, "datatype": "FP32", "shape": [-1, 1]}],
    }


def export_onnx(model: TabularRegressor, n_features: int, directory: Path) -> Path:
    """Write model.onnx from the model on CPU, in eval mode, with a dynamic batch axis."""
    served = ServedModel(model.cpu()).eval()
    path = directory / ONNX_FILE
    with torch.no_grad():
        # The TorchScript exporter: it emits opset 17 directly, using standard ops only
        # (Gemm, Relu, Sub, Div, Mul, Add, Squeeze, Unsqueeze). torch is pinned in run_task.sh.
        torch.onnx.export(
            served,
            (torch.zeros(2, n_features),),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    return path


def verify_onnx(
    path: Path, model: TabularRegressor, x_test: torch.Tensor, y_test: torch.Tensor, expected_rmse: float,
    atol: float = 1e-4,
) -> dict:
    """Check the export; raises ValueError on any failure. Returns the parity numbers."""
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = x_test.detach().cpu().numpy().astype(np.float32)
    y = y_test.detach().cpu().numpy().astype(np.float64)

    onnx_out = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
    with torch.no_grad():
        torch_out = ServedModel(model.cpu()).eval()(torch.from_numpy(x)).numpy()
    if onnx_out.shape != (len(x), 1) or onnx_out.dtype != np.float32:
        raise ValueError(f"ONNX output is {onnx_out.dtype} {onnx_out.shape}, expected float32 ({len(x)}, 1)")
    max_abs_diff = float(np.max(np.abs(onnx_out - torch_out)))
    if max_abs_diff > atol:
        raise ValueError(f"ONNX and PyTorch differ by up to {max_abs_diff:.2e} on the test split (limit {atol})")

    for batch in (1, 1000):
        rows = np.resize(x, (batch, x.shape[1]))
        out = session.run([OUTPUT_NAME], {INPUT_NAME: rows})[0]
        if out.shape != (batch, 1):
            raise ValueError(f"batch {batch} gave output shape {out.shape}, expected ({batch}, 1)")

    onnx_rmse = math.sqrt(float(np.mean((onnx_out[:, 0].astype(np.float64) - y) ** 2)))
    if abs(onnx_rmse - expected_rmse) > atol:
        raise ValueError(f"RMSE from ONNX is {onnx_rmse:.6f}, metrics.json says {expected_rmse:.6f}")

    return {
        "test_rows": len(x),
        "max_abs_diff_vs_pytorch": max_abs_diff,
        "onnx_test_rmse": onnx_rmse,
        "metrics_test_rmse": expected_rmse,
        "rmse_abs_diff": abs(onnx_rmse - expected_rmse),
        "batch_sizes_run": [1, 1000],
        "onnxruntime_version": onnxruntime.__version__,
    }


def predict_onnx(path: Path, rows: list[list[float]]) -> list[float]:
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run([OUTPUT_NAME], {INPUT_NAME: np.asarray(rows, dtype=np.float32)})[0][:, 0].tolist()
