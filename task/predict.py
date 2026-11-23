"""Run a prediction from a published model, or a local directory, with named inputs.

  predict.py --model org/name --input '{"MedInc": 8.3, "HouseAge": 41, ...}'

With no --input it uses the sample below. The input is checked against the
schema in config.json, so a missing or unknown feature is an error rather than
a silently misaligned column.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from tabular_model import load_model

SAMPLE = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322.0,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face repository, or a local directory.")
    parser.add_argument("--input", help="JSON object of feature name to value.")
    parser.add_argument("--revision", default="main", help="Branch of the Hugging Face repository.")
    args = parser.parse_args()

    directory = Path(args.model)
    if not directory.is_dir():
        directory = Path(snapshot_download(args.model, revision=args.revision, allow_patterns=["config.json", "model.safetensors"]))
    model, config = load_model(directory)

    inputs = json.loads(args.input) if args.input else SAMPLE
    names = [feature["name"] for feature in config["features"]]
    missing, unknown = set(names) - set(inputs), set(inputs) - set(names)
    if missing or unknown:
        print(f"bad input: missing {sorted(missing)}, unknown {sorted(unknown)}", file=sys.stderr)
        return 2

    with torch.no_grad():
        value = model(torch.tensor([[float(inputs[name]) for name in names]], dtype=torch.float32)).item()
    print(json.dumps({"inputs": inputs, config["target"]["name"]: value}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
