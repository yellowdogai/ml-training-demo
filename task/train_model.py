"""Download California Housing, train a regression model, publish it to Hugging Face.

Runs as a YellowDog task (see run_task.sh) and needs nothing from YellowDog, so
it also runs on a laptop. Omit --hf-repo to train without publishing.
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, get_token
from sklearn.model_selection import train_test_split

from onnx_export import ONNX_FILE, export_onnx, onnx_config, predict_onnx, verify_onnx
from dataset import find_archive, load_california_housing
from tabular_model import TabularRegressor, activation_name, save_model

FEATURE_DESCRIPTIONS = {
    "MedInc": "Median income in the block group, in tens of thousands of US dollars",
    "HouseAge": "Median house age in the block group, in years",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Block group population",
    "AveOccup": "Average number of household members",
    "Latitude": "Block group latitude, in degrees",
    "Longitude": "Block group longitude, in degrees",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", help="Hugging Face repository, org/name. Omit to skip publishing.")
    parser.add_argument("--public", action="store_true", help="Create the repository as public (default private).")
    parser.add_argument("--hf-revision", default="main", help="Branch to publish to. A sweep run uses its own, so main is untouched.")
    parser.add_argument("--run-id", default="local", help="Recorded in the metrics and the commit message.")
    parser.add_argument("--output-dir", type=Path, default=Path("model-output"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15, help="Stop after this many epochs with no improvement.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="cuda fails if no GPU is usable, rather than using the CPU."
    )
    return parser.parse_args()


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((prediction - target) ** 2)).item()


def r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual = torch.sum((target - prediction) ** 2)
    total = torch.sum((target - target.mean()) ** 2)
    return (1 - residual / total).item()


def train(args: argparse.Namespace, x_train, y_train, x_val, y_val) -> tuple[TabularRegressor, int]:
    model = TabularRegressor(x_train.shape[1], args.hidden_sizes).to(x_train.device)
    model.x_mean.copy_(x_train.mean(dim=0))
    model.x_std.copy_(x_train.std(dim=0))
    model.y_mean.fill_(y_train.mean())
    model.y_std.fill_(y_train.std())

    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaled_target = (y_train - model.y_mean) / model.y_std

    best_rmse, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in torch.randperm(len(x_train), device=x_train.device).split(args.batch_size):
            optimiser.zero_grad()
            loss = torch.nn.functional.mse_loss(model.forward_scaled(x_train[batch]), scaled_target[batch])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_rmse = rmse(model(x_val), y_val)
        print(f"epoch {epoch:3d}  val_rmse {val_rmse:.4f}", flush=True)

        if val_rmse < best_rmse:
            best_rmse, best_state, best_epoch = val_rmse, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def model_card(repo: str, config: dict, metrics: dict, samples: list[dict]) -> str:
    rows = "\n".join(
        f"| `{f['name']}` | {f['description']} | {f['min']:.2f} | {f['max']:.2f} |" for f in config["features"]
    )
    names = [f["name"] for f in config["features"]]
    target = config["target"]["name"]
    n_features = len(names)
    sample_rows = ",\n".join(f"    {[s['inputs'][n] for n in names]}" for s in samples)
    sample_outputs = ", ".join(f"{s[target]:.4f}" for s in samples)
    kserve_data = ", ".join(str(samples[0]["inputs"][n]) for n in names)
    return f"""---
library_name: pytorch
pipeline_tag: tabular-regression
tags:
- tabular-regression
- onnx
- yellowdog
datasets:
- california-housing
metrics:
- rmse
- r2
---

# {repo}

A proof of concept: a small PyTorch regression model trained as a
[YellowDog](https://yellowdog.ai) task on the California Housing dataset. Given
eight properties of a census block group, it predicts `{config["target"]["name"]}`:
{config["target"]["description"]}. The prices are from the 1990 US census, so this
demonstrates a pipeline and is not a valuation tool.

## Results

| Test metric | Value |
|---|---|
| RMSE | {metrics["test_rmse"]:.3f} |
| R² | {metrics["test_r2"]:.3f} |
| RMSE of always predicting the mean (baseline) | {metrics["baseline_test_rmse"]:.3f} |

Trained on: {trained_on(metrics)}.

## Inputs

| Feature | Description | Train min | Train max |
|---|---|---|---|
{rows}

`config.json` holds this schema and the architecture: a {config["activation"]} MLP
with hidden layers {config["hidden_sizes"]}. `model.safetensors` holds the weights
and the scaling, so a caller passes raw units in the order above.

## Inference

`{ONNX_FILE}` (ONNX opset {config["onnx"]["opset"]}) is the whole model, including
the input and output scaling. Send raw units in the feature order above, as float32
shaped `[batch, {n_features}]`, and you get `{target}` back in its own units, shaped
`[batch, 1]`. No code from this repository is needed.

```python
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download

session = onnxruntime.InferenceSession(hf_hub_download("{repo}", "{ONNX_FILE}"))
rows = np.array([
{sample_rows}
], dtype=np.float32)  # {", ".join(names)}
print(session.run(["output"], {{"input": rows}})[0][:, 0])
# [{sample_outputs}]
```

### KServe v2

Served by Triton or MLServer under the name `yd-training`, one row is:

```
POST /v2/models/yd-training/infer
```

```json
{{"inputs": [{{"name": "input", "datatype": "FP32", "shape": [1, {n_features}], "data": [{kserve_data}]}}]}}
```

and the response's `outputs[0].data` is `[{samples[0][target]:.4f}]` ({config["target"]["description"]}).
For N rows, send `"shape": [N, {n_features}]` with the rows flattened, row by row, into `data`.
"""


def publish(args: argparse.Namespace) -> None:
    """Publish in one commit, so a failed upload cannot leave a half-updated model."""
    api = HfApi()  # uses HF_TOKEN, or the login saved by 'hf auth login'
    api.create_repo(args.hf_repo, repo_type="model", private=not args.public, exist_ok=True)
    if args.hf_revision != "main":
        api.create_branch(args.hf_repo, branch=args.hf_revision, repo_type="model", exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.hf_repo,
        repo_type="model",
        revision=args.hf_revision,
        folder_path=args.output_dir,
        commit_message=f"Training run {args.run_id}",
    )
    print(f"published {commit.commit_url}", flush=True)


def node_location() -> dict | None:
    """Where the task ran, from the variables run_task.sh sets from EC2 instance metadata."""
    location = {
        "region": os.environ.get("NODE_REGION"),
        "availability_zone": os.environ.get("NODE_ZONE"),
        "instance_type": os.environ.get("NODE_INSTANCE_TYPE"),
    }
    return location if any(location.values()) else None


def trained_on(metrics: dict) -> str:
    device = metrics.get("gpu_name") or "CPU"
    node = metrics.get("node")
    if node and node.get("availability_zone"):
        return f"{device}, {node.get('instance_type') or 'an instance'} in {node['availability_zone']}"
    return device


def main() -> int:
    args = parse_args()
    if args.hf_repo and not get_token():
        print("no Hugging Face token: set HF_TOKEN or run 'hf auth login'", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"--device cuda, but torch {torch.__version__} can't use a GPU here", file=sys.stderr)
        return 3

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(device) if args.device == "cuda" else None
    print(f"training on {gpu_name or 'CPU'} with torch {torch.__version__}", flush=True)

    # The repository's copy of the dataset, via the Hub on a node: no download from figshare.
    dataset = load_california_housing(args.data_dir, find_archive(args.hf_repo))
    print(f"downloaded {dataset.data.shape[0]} rows, {dataset.data.shape[1]} features", flush=True)

    x_rest, x_test, y_rest, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.1, random_state=args.seed
    )
    x_train, x_val, y_train, y_val = train_test_split(x_rest, y_rest, test_size=1 / 9, random_state=args.seed)
    x_train, y_train = (torch.tensor(a, dtype=torch.float32, device=device) for a in (x_train, y_train))
    x_val, y_val = (torch.tensor(a, dtype=torch.float32, device=device) for a in (x_val, y_val))
    x_test, y_test = (torch.tensor(a, dtype=torch.float32, device=device) for a in (x_test, y_test))

    started = time.monotonic()
    model, best_epoch = train(args, x_train, y_train, x_val, y_val)
    train_seconds = time.monotonic() - started

    with torch.no_grad():
        prediction = model(x_test)
    baseline = rmse(y_train.mean().expand_as(y_test), y_test)
    metrics = {
        "run_id": args.run_id,
        "device": args.device,
        "gpu_name": gpu_name,
        "node": node_location(),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1) if gpu_name else None,
        "torch_version": torch.__version__,
        "best_epoch": best_epoch,
        "train_seconds": round(train_seconds, 1),
        "rows": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "test_rmse": rmse(prediction, y_test),
        "test_r2": r2(prediction, y_test),
        "baseline_test_rmse": baseline,
    }
    print(json.dumps(metrics, indent=2), flush=True)

    if metrics["test_rmse"] >= baseline:
        print("the model does not beat predicting the mean, not publishing it", file=sys.stderr)
        return 1

    config = {
        "model_type": "tabular_regressor",
        "hidden_sizes": args.hidden_sizes,
        "features": [
            {
                "name": name,
                "description": FEATURE_DESCRIPTIONS[name],
                "min": x_train[:, i].min().item(),
                "max": x_train[:, i].max().item(),
            }
            for i, name in enumerate(dataset.feature_names)
        ],
        "target": {
            "name": "MedHouseVal",
            "description": "median house value in the block group, in hundreds of thousands of US dollars",
        },
        # What is served: enough to rebuild or serve the network from the repository alone.
        "activation": activation_name(model),
        "architectures": ["TabularRegressor"],
        "onnx": onnx_config(len(dataset.feature_names)),
    }
    save_model(model.cpu(), config, args.output_dir)

    onnx_path = export_onnx(model, len(dataset.feature_names), args.output_dir)
    try:
        parity = verify_onnx(onnx_path, model, x_test, y_test, metrics["test_rmse"])
    except Exception as error:  # any failed check stops the publish
        print(f"ONNX export failed its checks, not publishing: {error}", file=sys.stderr)
        return 1
    print("ONNX checks passed: " + json.dumps(parity), flush=True)
    metrics["onnx"] = parity

    # Fixed rows from the dataset, in raw units, with what the exported graph returns for them.
    # Rounded first, so the outputs below are exact for the inputs as printed.
    sample_inputs = [{n: round(float(v), 4) for n, v in zip(dataset.feature_names, row)} for row in dataset.data[:3]]
    sample_outputs = predict_onnx(onnx_path, [list(s.values()) for s in sample_inputs])
    samples = [{"inputs": s, config["target"]["name"]: y} for s, y in zip(sample_inputs, sample_outputs)]
    print("ONNX samples: " + json.dumps(samples), flush=True)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        model_card(args.hf_repo or "california-housing-regressor", config, metrics, samples), encoding="utf-8"
    )
    print(f"saved model to {args.output_dir}", flush=True)

    if args.hf_repo:
        publish(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
