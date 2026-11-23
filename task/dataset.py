"""The California Housing dataset, from the copy in this repository.

scikit-learn downloads it from figshare, which refuses some cloud networks (HTTP
403 from some AWS regions). So the project keeps the same archive in
dataset/cal_housing.tgz, train.py copies it to the Hugging Face repository's "data"
branch, and the node loads it from there. It is checked against scikit-learn's own
checksum, then handed to scikit-learn, so the data and its processing are exactly
what fetch_california_housing gives.
"""

import hashlib
import tarfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing

ARCHIVE_NAME = "cal_housing.tgz"
# scikit-learn's checksum for this archive (sklearn.datasets._california_housing.ARCHIVE).
SHA256 = "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"
# Where the repository keeps it, relative to this file. Not present on a node.
REPO_COPY = Path(__file__).resolve().parent.parent / "dataset" / ARCHIVE_NAME
HUB_BRANCH = "data"


def check_archive(archive: Path) -> Path:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{archive} has SHA-256 {digest}, not the California Housing archive's {SHA256}")
    return archive


def load_california_housing(data_dir: Path, archive: Path):
    """fetch_california_housing's result, built from a local copy of its archive."""
    data_dir.mkdir(parents=True, exist_ok=True)
    # The cache file fetch_california_housing looks for, built the way it builds it.
    cache = data_dir / "cal_housing_py3.pkz"
    if not cache.exists():
        with tarfile.open(check_archive(archive), mode="r:gz") as f:
            rows = np.loadtxt(f.extractfile("CaliforniaHousing/cal_housing.data"), delimiter=",")
        joblib.dump(rows[:, [8, 7, 2, 3, 4, 5, 6, 1, 0]], cache, compress=6)
    return fetch_california_housing(data_home=str(data_dir), download_if_missing=False)


def find_archive(hf_repo: str | None) -> Path:
    """The repository's copy when running from a checkout, otherwise the Hub's "data" branch."""
    if REPO_COPY.exists():
        return REPO_COPY
    if not hf_repo:
        raise FileNotFoundError(f"no {ARCHIVE_NAME}: run from a checkout, or pass --hf-repo")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(hf_repo, ARCHIVE_NAME, revision=HUB_BRANCH))
