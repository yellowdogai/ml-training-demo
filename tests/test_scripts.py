"""Exercises train.py against a stub, with the real SDK models.

  python -m unittest discover tests

No account needed. This proves the order of the calls, what is reported, and the
request the SDK really sends. It cannot prove the platform accepts that request;
only a real run does.
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train as submit_task  # the root train.py, which submits the task
from yellowdog_client.model import TaskStatus, WorkRequirementStatus
from yellowdog_client.common.json import Json

ENV = {"YD_KEY": "k", "YD_SECRET": "s", "HF_TOKEN": "secret-token"}


def work_requirement(status, starved=False):
    return SimpleNamespace(id="ydid:workreq:1", status=status, taskGroups=[SimpleNamespace(id="ydid:taskgrp:1", starved=starved)])


def stub_client(statuses, failed_tasks=()):
    client = mock.MagicMock()
    client.work_client.add_work_requirement.return_value = work_requirement(WorkRequirementStatus.RUNNING)
    client.work_client.get_work_requirement_by_id.side_effect = statuses
    client.work_client.find_tasks.return_value = list(failed_tasks)
    return client


class SubmitTaskTest(unittest.TestCase):
    def run_main(self, client, *extra):
        argv = ["train.py", "--hf-repo", "org/name", "--worker-tag", "tag", *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict("os.environ", ENV), mock.patch.object(
            submit_task, "make_client", return_value=client
        ), mock.patch.object(submit_task.time, "sleep"), mock.patch.object(
            submit_task, "ensure_dataset_on_hub"
        ) as self.dataset_copy:
            return submit_task.main()

    def test_submits_and_reports_success(self):
        client = stub_client([work_requirement(WorkRequirementStatus.RUNNING), work_requirement(WorkRequirementStatus.COMPLETED)])
        self.assertEqual(self.run_main(client), 0)
        group_id, tasks = client.work_client.add_tasks_to_task_group_by_id.call_args.args
        self.assertEqual(group_id, "ydid:taskgrp:1")
        self.assertEqual(tasks[0].environment["HF_TOKEN"], "secret-token")
        self.assertEqual(tasks[0].arguments, ["-c", submit_task.BOOTSTRAP])
        self.assertEqual(tasks[0].environment["TORCH_DEVICE"], "cuda")
        client.close.assert_called_once()

    def test_the_dataset_is_on_the_hub_before_the_tasks_are_submitted(self):
        client = stub_client([work_requirement(WorkRequirementStatus.COMPLETED)])
        self.run_main(client)
        self.dataset_copy.assert_called_once_with("org/name", False)

    def test_the_repository_copy_of_the_dataset_is_the_one_scikit_learn_checks(self):
        sys.path.insert(0, str(submit_task.TASK_DIR))
        import dataset

        self.assertEqual(dataset.check_archive(submit_task.DATASET), submit_task.DATASET)
        self.assertEqual(dataset.REPO_COPY, submit_task.DATASET)

    def test_task_only_targets_the_worker_tag(self):
        client = stub_client([work_requirement(WorkRequirementStatus.COMPLETED)])
        self.run_main(client)
        requirement = client.work_client.add_work_requirement.call_args.args[0]
        self.assertEqual(requirement.taskGroups[0].runSpecification.workerTags, ["tag"])

    def test_failure_exits_non_zero_and_reads_the_failed_tasks(self):
        error = SimpleNamespace(error="boom", errorType="stub", processExitCode=1)
        failed = SimpleNamespace(name="train-and-publish", errors=[error])
        client = stub_client([work_requirement(WorkRequirementStatus.FAILED)], [failed])
        self.assertEqual(self.run_main(client), 1)
        search = client.work_client.find_tasks.call_args.args[0]
        self.assertEqual(search.statuses, [TaskStatus.FAILED])

    def test_task_serialises_as_the_sdk_sends_it(self):
        # The stubs above never serialise; this is the JSON add_tasks_to_task_group_by_id would send.
        args = submit_task.default_options(hf_repo="org/name", worker_tag="tag", namespace="ns")
        runs = submit_task.plan_runs("run-1", {"epochs": 3})
        _, (task,) = submit_task.build_work_requirement(args, "run-1", runs, "token")
        body = json.loads(json.dumps(Json.dump(task)))
        self.assertEqual(body["taskType"], "bash")
        self.assertEqual(body["environment"]["RUN_ID"], "run-1")
        self.assertEqual(body["environment"]["HF_REVISION"], "main")
        self.assertEqual(body["environment"]["TRAIN_ARGS"], "--epochs 3")
        # No inputs or outputs: the task needs no storage, only Hugging Face.
        self.assertNotIn("data", body)
        self.assertNotIn("taskData", body)

    def test_bundle_has_every_task_module_the_scripts_import(self):
        import ast
        import base64
        import io
        import tarfile

        task_dir = submit_task.TASK_DIR
        local = {p.stem for p in task_dir.glob("*.py")}
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(submit_task.bundle_task_code()))) as tar:
            bundled = set(tar.getnames())
        for script in ("train_model.py", "predict.py"):
            tree = ast.parse((task_dir / script).read_text(encoding="utf-8"))
            imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            for module in imported & local:
                self.assertIn(f"{module}.py", bundled, f"{script} imports {module}, which the task bundle lacks")

    def test_a_sweep_is_one_task_per_entry_each_on_its_own_branch(self):
        client = stub_client([work_requirement(WorkRequirementStatus.COMPLETED)])
        sweep = '[{"learning-rate": 0.003}, {"hidden-sizes": [128, 128]}]'
        with mock.patch("builtins.print"):
            self.assertEqual(self.run_main(client, "--epochs", "40", "--sweep", sweep), 0)
        tasks = client.work_client.add_tasks_to_task_group_by_id.call_args.args[1]
        environments = [t.environment for t in tasks]
        self.assertEqual([e["TRAIN_ARGS"] for e in environments],
                         ["--epochs 40 --learning-rate 0.003", "--epochs 40 --hidden-sizes 128 128"])
        run_ids = [e["RUN_ID"] for e in environments]
        self.assertEqual([e["HF_REVISION"] for e in environments], [f"sweep-{r}" for r in run_ids])
        self.assertNotIn("main", [e["HF_REVISION"] for e in environments])
        requirement = client.work_client.add_work_requirement.call_args.args[0]
        self.assertFalse(requirement.taskGroups[0].finishIfAnyTaskFailed, "one failed run must not stop the rest")

    def test_settings_that_could_inject_shell_are_refused(self):
        for bad in ({"epochs": "3; reboot"}, {"hidden-sizes": ["64"]}, {"unknown": 1}, {"seed": True}):
            with self.assertRaises(ValueError):
                submit_task.plan_runs("r", sweep=[bad])
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self.run_main(mock.MagicMock(), "--sweep", '[{"epochs": "x"}]')

    def test_dry_run_does_not_print_the_token(self):
        with mock.patch("builtins.print") as printed:
            self.assertEqual(self.run_main(mock.MagicMock(), "--dry-run"), 0)
        self.assertNotIn("secret-token", str(printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
