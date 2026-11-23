"""Train on a cloud GPU with YellowDog and publish to Hugging Face: submit the task and follow it.

The task downloads the data, trains on a GPU and publishes the model to Hugging
Face. It runs on the worker pool configure.py made, which it finds by worker tag.

  train.py
  train.py --epochs 60 --seed 7
  train.py --sweep '[{"learning-rate": 0.003}, {"hidden-sizes": [128, 128]}, {}]'
  train.py --dry-run

A sweep submits one task per entry, each with those training settings on top of
the others. Sweep runs publish to their own branch of the repository
(sweep-<run id>), never to main, so they can't overwrite the published model.
With [pool] max_nodes above 1 in yd-training.toml they run in parallel, each on
its own node, wherever YellowDog finds capacity.

Settings can come from .env; see .env.example. HF_TOKEN is set as an environment
variable on each task. yd-training.ipynb drives the same functions from a notebook.
"""

import argparse
import base64
import io
import json
import os
import sys
import tarfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from yellowdog_client import PlatformClient
from yellowdog_client.model import (
    RunSpecification,
    Task,
    TaskGroup,
    TaskSearch,
    TaskStatus,
    WorkRequirement,
    WorkRequirementStatus,
)

from yd_common import NAMESPACE, WORKER_TAG, make_client, missing_env

TASK_DIR = Path(__file__).resolve().parent / "task"
TASK_FILES = [
    "run_task.sh", "train_model.py", "dataset.py", "tabular_model.py", "onnx_export.py", "predict.py", "requirements.txt",
]
DATASET = Path(__file__).resolve().parent / "dataset" / "cal_housing.tgz"

# The code travels inside the task, so this needs no object store. The bundle is
# removed from the environment before anything else runs.
BOOTSTRAP = 'printf %s "$TASK_BUNDLE" | base64 -d | tar xz && unset TASK_BUNDLE && exec bash run_task.sh'

POLL_SECONDS = 20
FINISHED = {WorkRequirementStatus.COMPLETED, WorkRequirementStatus.FAILED, WorkRequirementStatus.CANCELLED}

# task/train_model.py options a run or a sweep may set. Values must be numbers (or a
# list of whole numbers for hidden-sizes): they reach the node through the shell.
TRAIN_OPTIONS = {
    "epochs": int,
    "seed": int,
    "patience": int,
    "batch-size": int,
    "learning-rate": float,
    "hidden-sizes": list,
}


@dataclass
class Run:
    """One training task: its settings, and where it publishes."""

    run_id: str
    settings: dict = field(default_factory=dict)
    revision: str = "main"

    @property
    def train_args(self) -> str:
        parts = []
        for name, value in self.settings.items():
            values = value if isinstance(value, list) else [value]
            parts += [f"--{name}", *(str(v) for v in values)]
        return " ".join(parts)

    def log_url(self, repo: str) -> str:
        return f"https://huggingface.co/{repo}/blob/logs/logs/{self.run_id}.txt"


def check_settings(settings: dict) -> dict:
    for name, value in settings.items():
        kind = TRAIN_OPTIONS.get(name)
        if kind is None:
            raise ValueError(f"unknown training option '{name}'; use one of {', '.join(TRAIN_OPTIONS)}")
        if kind is list:
            if not (isinstance(value, list) and value and all(isinstance(v, int) and v > 0 for v in value)):
                raise ValueError(f"{name} must be a list of positive whole numbers, such as [64, 64]")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or (kind is int and not isinstance(value, int)):
            raise ValueError(f"{name} must be a {'whole number' if kind is int else 'number'}")
    return settings


def plan_runs(run_id: str, base: dict | None = None, sweep: list[dict] | None = None) -> list[Run]:
    """One run publishing to main, or one run per sweep entry, each on its own branch."""
    base = check_settings(dict(base or {}))
    if not sweep:
        return [Run(run_id, base)]
    return [
        Run(f"{run_id}-{n}", check_settings({**base, **entry}), revision=f"sweep-{run_id}-{n}")
        for n, entry in enumerate(sweep, start=1)
    ]


def default_options(**overrides) -> argparse.Namespace:
    """The command line's defaults, for calling these functions from a notebook."""
    return argparse.Namespace(**{**vars(build_parser().parse_args([])), **overrides})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-repo", default=os.environ.get("HF_REPO"), help="Hugging Face repository to publish to, org/name.")
    parser.add_argument("--public", action="store_true", help="Create the repository as public (default private).")
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--worker-tag", default=WORKER_TAG, help="Workers the task may run on.")
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda", help="cuda fails the task unless it can use a GPU."
    )
    parser.add_argument("--timeout-minutes", type=int, default=30, help="Aborts a task if it runs longer than this.")
    parser.add_argument("--epochs", type=int, help="Passed to task/train_model.py.")
    parser.add_argument("--seed", type=int, help="Passed to task/train_model.py.")
    parser.add_argument(
        "--sweep", type=json.loads, help="A JSON list of training settings, one task each, e.g. "
        '\'[{"learning-rate": 0.003}, {"hidden-sizes": [128, 128]}]\'.'
    )
    parser.add_argument("--no-wait", action="store_true", help="Submit and exit without following the task.")
    parser.add_argument("--dry-run", action="store_true", help="Print the request and send nothing.")
    return parser


def bundle_task_code() -> str:
    """The task's files as a base64 tarball, with Unix line endings whatever the client's."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in TASK_FILES:
            content = (TASK_DIR / name).read_bytes().replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(content))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_work_requirement(
    args: argparse.Namespace, name: str, runs: list[Run], hf_token: str
) -> tuple[WorkRequirement, list[Task]]:
    bundle = bundle_task_code()
    tasks = [
        Task(
            name=run.run_id,
            taskType="bash",
            arguments=["-c", BOOTSTRAP],
            environment={
                "TASK_BUNDLE": bundle,
                "HF_REPO": args.hf_repo,
                "HF_REVISION": run.revision,
                "HF_TOKEN": hf_token,
                "HF_PUBLIC": "true" if args.public else "false",
                "TORCH_DEVICE": args.device,
                "RUN_ID": run.run_id,
                "TRAIN_ARGS": run.train_args,
            },
        )
        for run in runs
    ]
    group = TaskGroup(
        name="training",
        runSpecification=RunSpecification(
            taskTypes=["bash"],
            workerTags=[args.worker_tag],
            namespaces=[args.namespace],
            taskTimeout=timedelta(minutes=args.timeout_minutes),
            maximumTaskRetries=0,
        ),
        finishIfAllTasksFinished=True,
        # In a sweep, one failed run shouldn't stop the others.
        finishIfAnyTaskFailed=len(runs) == 1,
    )
    return WorkRequirement(namespace=args.namespace, name=name, taskGroups=[group]), tasks


def ensure_dataset_on_hub(repo: str, public: bool) -> None:
    """Copy dataset/cal_housing.tgz to the repository's "data" branch, once, for the nodes to load."""
    from huggingface_hub import HfApi

    api = HfApi()  # uses HF_TOKEN
    api.create_repo(repo, repo_type="model", private=not public, exist_ok=True)
    if api.file_exists(repo, DATASET.name, revision="data"):
        return
    api.create_branch(repo, branch="data", repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=DATASET, path_in_repo=DATASET.name, repo_id=repo, revision="data",
        commit_message="California Housing dataset, which training nodes load from here",
    )
    print(f"copied {DATASET.name} to the {repo} data branch", flush=True)


def submit(client: PlatformClient, work_requirement: WorkRequirement, tasks: list[Task]) -> str:
    """Submit the tasks, after making sure the dataset is where the nodes will look for it."""
    environment = tasks[0].environment
    ensure_dataset_on_hub(environment["HF_REPO"], environment["HF_PUBLIC"] == "true")
    submitted = client.work_client.add_work_requirement(work_requirement)
    client.work_client.add_tasks_to_task_group_by_id(submitted.taskGroups[0].id, tasks)
    return submitted.id


def task_states(client: PlatformClient, work_requirement_id: str) -> list[dict]:
    """Each task's status, and the region and compute source of the node running it."""
    states = []
    for task in client.work_client.find_tasks(TaskSearch(workRequirementId=work_requirement_id)):
        details = None
        if task.nodeId:
            details = client.worker_pool_client.get_node_by_id(task.nodeId).details
        states.append({
            "run": task.name,
            "status": task.status.name if task.status else "",
            "region": getattr(details, "region", None) or "",
            "source": getattr(details, "sourceName", None) or "",
            "errors": [f"{e.errorType} exit {e.processExitCode}: {e.error}" for e in task.errors or []],
        })
    return sorted(states, key=lambda s: s["run"])


def follow(
    client: PlatformClient, work_requirement_id: str, on_poll: Callable[[WorkRequirement], None] | None = None
) -> int:
    """Report status changes until the work requirement finishes; 0 means every task completed."""
    last = None
    warned_starved = False
    while True:
        work_requirement = client.work_client.get_work_requirement_by_id(work_requirement_id)
        group = work_requirement.taskGroups[0]
        if on_poll:
            on_poll(work_requirement)
        elif work_requirement.status != last:
            print(f"{datetime.now():%H:%M:%S}  {work_requirement.status.name}", flush=True)
        last = work_requirement.status
        if group.starved and not warned_starved and not on_poll:
            print(
                "no worker yet. A scale-to-zero pool takes some minutes to start a node; "
                "if this persists, check a pool exists with this worker tag.",
                flush=True,
            )
            warned_starved = True
        if work_requirement.status in FINISHED:
            break
        time.sleep(POLL_SECONDS)

    failed = client.work_client.find_tasks(
        TaskSearch(workRequirementId=work_requirement_id, statuses=[TaskStatus.FAILED])
    )
    for task in failed:
        for error in task.errors or []:
            print(f"{task.name}: {error.errorType} exit {error.processExitCode}: {error.error}", file=sys.stderr)
    return 0 if work_requirement.status == WorkRequirementStatus.COMPLETED and not failed else 1


def print_dry_run(work_requirement: WorkRequirement, tasks: list[Task]) -> None:
    """Print the request with the token and the bundle left out."""
    shown = []
    for task in tasks:
        environment = dict(task.environment)
        environment["HF_TOKEN"] = "<set>" if environment["HF_TOKEN"] else "<NOT SET>"
        environment["TASK_BUNDLE"] = f"<{len(environment['TASK_BUNDLE'])} characters of base64>"
        shown.append({**asdict(task), "environment": environment})
    print(json.dumps({"workRequirement": asdict(work_requirement), "tasks": shown}, indent=2, default=str))


def new_run_id() -> str:
    return f"yd-training-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.hf_repo:
        parser.error("set HF_REPO in .env or pass --hf-repo")
    run_id = new_run_id()
    base = {k: v for k, v in (("epochs", args.epochs), ("seed", args.seed)) if v is not None}
    try:
        runs = plan_runs(run_id, base, args.sweep)
    except (ValueError, TypeError, AttributeError) as error:
        parser.error(f"--sweep: {error}")
    work_requirement, tasks = build_work_requirement(args, run_id, runs, os.environ.get("HF_TOKEN", ""))

    if args.dry_run:
        print_dry_run(work_requirement, tasks)
        return 0

    if missing_env("YD_KEY", "YD_SECRET", "HF_TOKEN"):
        return 2

    client = make_client()
    try:
        work_requirement_id = submit(client, work_requirement, tasks)
        print(f"submitted work requirement {work_requirement_id} ({run_id}, {len(runs)} task{'s' * (len(runs) > 1)})", flush=True)
        if args.no_wait:
            return 0
        status = follow(client, work_requirement_id)
        # run_task.sh pushes each task's output there, success or failure.
        for run in runs:
            where = "" if run.revision == "main" else f"  model: https://huggingface.co/{args.hf_repo}/tree/{run.revision}"
            print(f"log: {run.log_url(args.hf_repo)}{where}")
        return status
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
