"""The regression model, and how it is saved to and loaded from a directory.

The published repository holds only data: weights in safetensors and the
architecture and input schema in config.json. Loading it runs no code from the
Hub, which is why this class is not uploaded alongside the weights.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


class TabularRegressor(nn.Module):
    """An MLP that carries its own scaling, so callers pass and get raw units."""

    def __init__(self, n_features: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for hidden in hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("x_mean", torch.zeros(n_features))
        self.register_buffer("x_std", torch.ones(n_features))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward_scaled(self, x: torch.Tensor) -> torch.Tensor:
        """Raw features in, scaled target out. This is what training optimises."""
        return self.net((x - self.x_mean) / self.x_std).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_scaled(x) * self.y_std + self.y_mean


def activation_name(model: TabularRegressor) -> str:
    """The activation between the Linear layers, read from the network, e.g. "relu"."""
    names = {type(layer).__name__.lower() for layer in model.net if not isinstance(layer, nn.Linear)}
    if len(names) != 1:
        raise ValueError(f"expected one activation type between the layers, found {sorted(names)}")
    return names.pop()


def save_model(model: TabularRegressor, config: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), directory / WEIGHTS_FILE)
    (directory / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_model(directory: Path) -> tuple[TabularRegressor, dict]:
    config = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
    model = TabularRegressor(len(config["features"]), config["hidden_sizes"])
    model.load_state_dict(load_file(directory / WEIGHTS_FILE))
    model.eval()
    return model, config
