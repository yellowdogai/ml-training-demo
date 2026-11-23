"""The buttons and views behind yd-training.ipynb. Each notebook step calls one function here.

Nothing that costs money or changes an account happens without a button click. For a
headless run (jupyter nbconvert --execute), set DEMO_AUTO=1 to click Apply and Start
training automatically; DEMO_SWEEP=1 also starts the sweep. Clean up is never automatic.
"""

import os

# Keep Hugging Face's download progress bars and Windows symlink warning out of the views.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import json  # noqa: E402
import shutil
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as CallTimeout
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import HTML, Markdown, display

import configure
import train
from yd_common import NAMESPACE, WORKER_TAG, make_client  # also loads .env

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "task"))
from dataset import REPO_COPY, load_california_housing  # noqa: E402

AUTO = os.environ.get("DEMO_AUTO") == "1"
AUTO_SWEEP = os.environ.get("DEMO_SWEEP") == "1"
# On-demand price range of a g5.xlarge across the example's regions, for the cost notes.
PRICE_PER_HOUR = "about $1.00 to $1.40"
POLL_SECONDS = 10
CALL_TIMEOUT = 30  # the SDK's requests have no timeout; a stalled one must not freeze the view

STATE_COLOURS = {"node running": "#2ca02c", "no capacity": "#d62728", "some zones full": "#ff7f0e",
                 "not tried": "#bbbbbb", "usable": "#4c78a8", "skipped": "#d62728"}
REGION_PLACES = {
    "us-east-1": ("N. Virginia", 38.9, -77.4), "us-east-2": ("Ohio", 40.0, -83.0),
    "us-west-1": ("N. California", 37.8, -122.4), "us-west-2": ("Oregon", 45.6, -121.2),
    "ca-central-1": ("Montreal", 45.5, -73.6), "sa-east-1": ("Sao Paulo", -23.6, -46.6),
    "eu-west-1": ("Ireland", 53.3, -6.3), "eu-west-2": ("London", 51.5, -0.1),
    "eu-west-3": ("Paris", 48.9, 2.4), "eu-central-1": ("Frankfurt", 50.1, 8.7),
    "eu-north-1": ("Stockholm", 59.3, 18.1), "eu-south-2": ("Spain", 41.6, -0.9),
    "ap-south-1": ("Mumbai", 19.1, 72.9), "ap-northeast-1": ("Tokyo", 35.7, 139.7),
    "ap-northeast-2": ("Seoul", 37.6, 127.0), "ap-northeast-3": ("Osaka", 34.7, 135.5),
    "ap-southeast-1": ("Singapore", 1.4, 103.8), "ap-southeast-2": ("Sydney", -33.9, 151.2),
    "ap-east-1": ("Hong Kong", 22.3, 114.2),
}

_state = {"run": None, "runs": [], "plan": None}
_calls = ThreadPoolExecutor(max_workers=4)


# --- Small helpers --------------------------------------------------------------------

def _config() -> dict:
    return tomllib.loads(configure.CONFIG.read_text(encoding="utf-8"))


def _box(kind: str, title: str, body: str = "") -> HTML:
    colours = {"ok": ("#e8f5e9", "#2e7d32"), "warn": ("#fff8e1", "#b26a00"), "bad": ("#ffebee", "#c62828"),
               "info": ("#e3f2fd", "#1565c0")}
    background, border = colours[kind]
    return HTML(f'<div style="background:{background};border-left:4px solid {border};padding:8px 12px;'
                f'margin:6px 0;color:#222"><b>{title}</b>{"<br>" + body if body else ""}</div>')


def _call(fn, *args):
    """Run a YellowDog call with a time limit; None if it didn't answer in time."""
    try:
        return _calls.submit(fn, *args).result(timeout=CALL_TIMEOUT)
    except CallTimeout:
        return None


def _button(label: str, style: str = "success", width: str = "220px") -> widgets.Button:
    return widgets.Button(description=label, button_style=style, layout=widgets.Layout(width=width, height="36px"))


def _region_map(frame: pd.DataFrame, colour: str, height: int = 360):
    fig = px.scatter_geo(frame, lat="lat", lon="lon", color=colour, hover_name="place", text="order",
                         color_discrete_map=STATE_COLOURS, projection="natural earth")
    fig.update_traces(marker={"size": 18}, textfont={"color": "white", "size": 10})
    fig.update_layout(height=height, margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend_title_text="")
    return fig


def _regions_frame(names) -> pd.DataFrame:
    rows = []
    for order, name in enumerate(names, start=1):
        place, lat, lon = REGION_PLACES.get(name, (name, 0.0, 0.0))
        rows.append({"order": order, "region": name, "place": place, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def _hub_json(name: str, revision: str = "main") -> dict:
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(os.environ["HF_REPO"], name, revision=revision, force_download=True)).read_text())


def _log_tail(run_id: str, last: int = 30) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(os.environ["HF_REPO"], f"logs/{run_id}.txt", revision="logs", force_download=True)
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if not l.startswith("  ")]
    return "\n".join(lines[-last:])


def _guarded(handler, button: widgets.Button, retry_label: str, area: widgets.Output):
    """Run a button handler inside an output area, with any error shown there.

    What a click handler displays only reaches the notebook through an Output widget;
    otherwise JupyterLab puts it, and any error, in its log where no one sees it.
    """
    def run(clicked):
        with area:
            _run_shown(handler, clicked, button, retry_label)
    return run


def _run_shown(handler, clicked, button, retry_label):
    try:
        handler(clicked)
    except Exception as error:
        display(_box("bad", "That didn't work:", f"{type(error).__name__}: {error}"))
        button.disabled, button.description = False, retry_label
        if AUTO:
            raise


def _click_if_auto(handler, button: widgets.Button) -> None:
    """In a headless run, act as if the button was clicked. Called directly, so errors fail the run."""
    if AUTO:
        handler(button)


# --- Step 1: check ----------------------------------------------------------------------

def _checks() -> list[tuple[bool, str, str]]:
    results = []
    for name, why in (("YD_KEY", "the YellowDog application key"), ("YD_SECRET", "the YellowDog application key"),
                      ("HF_REPO", "the Hugging Face repository to publish to, as org/name"),
                      ("HF_TOKEN", "a Hugging Face token with write access")):
        results.append((bool(os.environ.get(name)), f"{name} is set in .env", f"Add {name}: {why}."))

    profile = os.environ.get("AWS_PROFILE", "")
    if not shutil.which("aws"):
        results.append((False, "AWS CLI installed", "Install the AWS CLI, then restart JupyterLab."))
    else:
        try:
            account = configure.account_id()
            results.append((True, f"Signed in to AWS (account {account}{', profile ' + profile if profile else ''})", ""))
        except RuntimeError as error:
            hint = f"aws sso login --profile {profile}" if profile else "aws sso login (or aws configure)"
            results.append((False, "Signed in to AWS",
                            f"Run <code>{hint}</code> in a terminal, then run this cell again. "
                            f"To use a named profile, set AWS_PROFILE in .env. ({error})"))

    results.append((bool(configure.yd_path()), "YellowDog CLI installed",
                    "Run <code>python -m pip install -r requirements.txt</code> in a terminal, then restart JupyterLab."))

    if os.environ.get("YD_KEY") and os.environ.get("YD_SECRET"):
        client = make_client()
        try:
            client.keyring_client.find_all_keyrings()
            results.append((True, f"YellowDog key works (namespace {NAMESPACE})", ""))
        except Exception as error:  # the SDK raises its own types
            results.append((False, "YellowDog key works", f"YellowDog refused the key: {str(error)[:200]}"))
        finally:
            client.close()

    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi

            who = HfApi().whoami()
            role = who.get("auth", {}).get("accessToken", {}).get("role", "")
            results.append((role in ("write", "fineGrained"), f"Hugging Face token works ({who['name']}, {role or 'unknown'} access)",
                            "The token needs write access to publish the model."))
        except Exception as error:
            results.append((False, "Hugging Face token works", f"Hugging Face refused the token: {str(error)[:200]}"))
    return results


def check() -> bool:
    """Step 1: everything the demo needs, with how to fix anything missing."""
    results = _checks()
    rows = "".join(
        f'<tr><td style="font-size:18px">{"✅" if ok else "❌"}</td><td>{label}'
        f'{"" if ok else "<br><span style=color:#c62828>" + fix + "</span>"}</td></tr>'
        for ok, label, fix in results
    )
    display(HTML(f"<table>{rows}</table>"))
    if all(ok for ok, _, _ in results):
        display(_box("ok", "All set.", "Go on to step 2."))
        return True
    display(_box("bad", "Fix the items marked ❌, then run this cell again.",
                 "Settings go in the <code>.env</code> file in the project folder. After changing it, "
                 "restart the kernel (<b>Kernel > Restart Kernel</b>) and run the cells from the top."))
    return False


# --- Step 2: configure ------------------------------------------------------------------

def set_up() -> None:
    """Step 2: show what will be set up and any warnings, then apply it with a button."""
    config = _config()
    display(Markdown(f"Looking up {len(config['regions'])} regions in AWS. This takes about 20 seconds..."))
    out = widgets.Output()
    display(out)
    with out:  # each region's result, including why any was skipped, stays above the map
        try:
            regions = configure.discover_all(config)
        except SystemExit:
            regions = []
    usable = {r.name: r for r in regions}
    frame = _regions_frame(config["regions"])
    frame["state"] = ["usable" if name in usable else "skipped" for name in frame["region"]]
    frame["zones"] = [", ".join(s.zone for s in usable[n].subnets) if n in usable else "" for n in frame["region"]]
    _region_map(frame, "state").show()
    display(frame[["order", "region", "place", "state", "zones"]].set_index("order"))

    # What will happen, and what to watch out for.
    keyring, _, credential = config["credential"].partition("/")
    n_sources = sum(len(r.subnets) for r in regions)
    plan = [
        f"AWS: the IAM user <b>{config['iam_user']}</b>, which may only launch, stop and terminate instances "
        f"tagged Project=yd-training (aws-policy.json).",
        f"YellowDog: its access key in the keyring <b>{keyring}</b>, an image family, "
        f"<b>{n_sources}</b> compute sources in <b>{len(regions)}</b> regions, and the template "
        f"<b>{configure.template_name(config)}</b>, in namespace <b>{NAMESPACE}</b>.",
        f"A worker pool with up to <b>{config['pool']['max_nodes']}</b> {config['instance_type']} node(s). "
        f"It starts empty and costs nothing until you train.",
    ]
    display(_box("info", "Clicking Apply will set up:", "<br>".join("• " + p for p in plan)))

    warnings = []
    skipped = [n for n in config["regions"] if n not in usable]
    if skipped:
        warnings.append(f"{len(skipped)} region(s) skipped and won't be used: {', '.join(skipped)}. "
                        "See the reasons printed above the map; edit yd-training.toml to change.")
    client = make_client()
    try:
        names = {k.name for k in client.keyring_client.find_all_keyrings()}
        if keyring not in names:
            warnings.append(f"The keyring <b>{keyring}</b> is new. Its password is shown <b>once</b> after Apply: "
                            "save it somewhere safe, because YellowDog can't show it again.")
        pools = list(configure.our_pools(client, NAMESPACE, WORKER_TAG))
        if any(p.claimedWorkerCount or p.workingWorkerCount for p in pools):
            warnings.append("The current worker pool is running a task, so it won't be replaced until it's idle.")
        elif pools:
            warnings.append(f"The current worker pool ({pools[0].name}) will be replaced by a new one.")
    finally:
        client.close()
    if not regions:
        display(_box("bad", "No usable region.", "See the reasons printed above the map. If you don't use each "
                     "region's default VPC, set <code>AWS_VPC</code> in .env to your VPC's name or ID."))
        return
    for warning in warnings:
        display(_box("warn", "Check:", warning))

    button = _button("Apply configuration")
    result = widgets.Output()
    display(widgets.HTML("<b>If that looks right, click Apply.</b> It takes about a minute, and is safe to repeat."),
            button, result)

    def apply(_):
        button.disabled, button.description = True, "Applying..."
        with result:
            client = make_client()
            try:
                print("1. AWS IAM user")
                configure.ensure_iam_user(config["iam_user"])
                print("2. YellowDog keyring")
                password = configure.ensure_credential(client, config["iam_user"], keyring, credential, False)
                if password:
                    display(_box("warn", "Save this keyring password now. It won't be shown again:",
                                 f"<code style='font-size:15px'>{password}</code>"))
                print("3. YellowDog templates")
                configure.RESOURCES.write_text(json.dumps(configure.build_resources(regions, config, NAMESPACE), indent=2) + "\n",
                                               encoding="utf-8")
                configure.yd_cli("resource", "create", "--namespace", NAMESPACE, str(configure.RESOURCES))
                print("4. Worker pool")
                configure.replace_pool(client, config, NAMESPACE, WORKER_TAG)
                display(_box("ok", "Configured.", "Go on to step 3."))
                button.description = "Applied"
            except (Exception, SystemExit) as error:
                display(_box("bad", "Configuration failed:", str(error)))
                button.disabled, button.description = False, "Try again"
                if AUTO:
                    raise
            finally:
                client.close()

    button.on_click(apply)
    _click_if_auto(apply, button)


# --- Step 3: train ----------------------------------------------------------------------

def _region_states(regions: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    states = []
    for region, group in sources.groupby("region", sort=False):
        if group["instances"].sum() > 0:
            state = "node running"
        elif (group["status"] == "ERRORED").all():
            state = "no capacity"
        elif (group["status"] == "ERRORED").any():
            state = "some zones full"
        else:
            state = "not tried"
        states.append({"region": region, "state": state})
    merged = regions.merge(pd.DataFrame(states), on="region", how="left") if states else regions.assign(state=None)
    return merged.fillna({"state": "not tried"})


def _watch(work_requirement_id: str, view: widgets.Output, header: widgets.HTML) -> str:
    """Refresh the live view until the work finishes; returns its final status."""
    client = make_client()
    regions = _regions_frame(_config()["regions"])
    started = time.time()
    try:
        while True:
            requirement = _call(client.work_client.get_work_requirement_by_id, work_requirement_id)
            if requirement is None:
                header.value = f"<i>YellowDog didn't answer in {CALL_TIMEOUT} s; trying again...</i>"
                time.sleep(POLL_SECONDS)
                continue
            status = requirement.status.name
            sources = pd.DataFrame(_call(configure.pool_sources, client, NAMESPACE, WORKER_TAG) or [])
            tasks = pd.DataFrame(_call(train.task_states, client, work_requirement_id) or [])
            minutes = (time.time() - started) / 60
            where = ""
            if not tasks.empty and (tasks["region"] != "").any():
                running = tasks[tasks["region"] != ""].iloc[0]
                where = f" on a node in <b>{running['source'].removeprefix('g5-xlarge-')}</b>"
            header.value = (f"<b>{requirement.name}</b>: <b>{status}</b>{where} "
                            f"<span style='color:#666'>({minutes:.1f} min, updated {time.strftime('%H:%M:%S')})</span>")
            with view:
                view.clear_output(wait=True)
                if not sources.empty:
                    _region_map(_region_states(regions, sources), "state", height=320).show()
                    tried = sources[sources["status"] != "NEW"]
                    if not tried.empty:
                        display(tried[["source", "status", "instances", "message"]].reset_index(drop=True))
                if not tasks.empty:
                    display(tasks[["run", "status", "region", "source"]])
            if requirement.status in train.FINISHED:
                return status
            time.sleep(POLL_SECONDS)
    finally:
        client.close()


def _start(runs: list, button: widgets.Button, done_label: str) -> bool:
    """Submit runs, show them live, then report each one's outcome."""
    header, view, outcome = widgets.HTML(), widgets.Output(), widgets.Output()
    display(header, view, outcome)
    name = runs[0].run_id.rsplit("-", 1)[0] if len(runs) > 1 else runs[0].run_id
    options = train.default_options()
    requirement, tasks = train.build_work_requirement(options, name, runs, os.environ["HF_TOKEN"])
    client = make_client()
    try:
        header.value = "Submitting (and making sure the dataset is on Hugging Face)..."
        requirement_id = train.submit(client, requirement, tasks)
    finally:
        client.close()
    header.value = "Submitted. YellowDog is looking for a GPU..."
    status = _watch(requirement_id, view, header)
    client = make_client()
    try:
        states = {s["run"]: s for s in train.task_states(client, requirement_id)}
    finally:
        client.close()
    with outcome:
        for run in runs:
            state = states.get(run.run_id, {})
            link = run.log_url(options.hf_repo)
            if state.get("status") == "COMPLETED":
                display(_box("ok", f"{run.run_id} finished, trained in {state.get('region')}.",
                             f'<a href="{link}" target="_blank">Full log</a>'))
            else:
                display(_box("bad", f"{run.run_id} ended {state.get('status', status)}.",
                             "; ".join(state.get("errors", [])) + f'<br><a href="{link}" target="_blank">Full log</a>'))
                try:
                    print(_log_tail(run.run_id))
                except Exception:
                    print("(no log: the run stopped before it could write one)")
    button.description = done_label
    button.disabled = False
    return all(states.get(r.run_id, {}).get("status") == "COMPLETED" for r in runs)


def start_training() -> None:
    """Step 3: choose settings, then start a training run with a button and watch it live."""
    config = _config()
    epochs = widgets.IntSlider(value=100, min=10, max=300, step=10, description="Epochs",
                               style={"description_width": "80px"}, layout=widgets.Layout(width="420px"))
    seed = widgets.IntText(value=42, description="Seed", style={"description_width": "80px"},
                           layout=widgets.Layout(width="200px"))
    button = _button("Start training")
    display(widgets.HTML("<b>1.</b> Choose the settings (the defaults are fine):"), epochs, seed)
    display(_box("info", "Clicking Start training will:",
                 f"• start <b>one {config['instance_type']}</b> (1 × NVIDIA A10G) in the first region with capacity, "
                 f"{PRICE_PER_HOUR} an hour on demand;<br>"
                 "• take about <b>10 minutes</b>: ~4 to boot, ~2 to train, then 5 idle before it's terminated;<br>"
                 f"• publish the model to <b>{os.environ.get('HF_REPO')}</b> on Hugging Face, replacing the one on main."))
    area = widgets.Output()
    display(widgets.HTML("<b>2.</b> Click to start. The view below updates every 10 seconds until the run ends."),
            button, area)

    def start(_):
        button.disabled, button.description = True, "Training..."
        run = train.plan_runs(train.new_run_id(), {"epochs": epochs.value, "seed": seed.value})[0]
        _state["run"] = run
        if not _start([run], button, "Train again") and AUTO:
            raise RuntimeError(f"run {run.run_id} failed: see its log above")

    start = _guarded(start, button, "Try again", area)
    button.on_click(start)
    _click_if_auto(start, button)


# --- Step 4: results --------------------------------------------------------------------

def results() -> None:
    """Step 4: how good the model is, where it trained, and the model in use."""
    import onnxruntime
    from huggingface_hub import hf_hub_download
    from sklearn.model_selection import train_test_split

    repo = os.environ["HF_REPO"]
    run = _state["run"]
    revision = run.revision if run else "main"
    metrics = _hub_json("metrics.json", revision)
    if run and metrics.get("run_id") != run.run_id:
        display(_box("warn", "This notebook's run didn't publish a model,",
                     f"so this shows the latest one on main ({metrics.get('run_id')})."))
    node = metrics.get("node") or {}
    display(HTML(
        "<div style='display:flex;gap:12px;flex-wrap:wrap'>" + "".join(
            f"<div style='border:1px solid #ddd;border-radius:6px;padding:10px 16px;min-width:140px'>"
            f"<div style='color:#666;font-size:12px'>{label}</div><div style='font-size:20px'><b>{value}</b></div></div>"
            for label, value in (
                ("Test RMSE", f"{metrics['test_rmse']:.3f}"),
                ("Baseline (always the mean)", f"{metrics['baseline_test_rmse']:.3f}"),
                ("R²", f"{metrics['test_r2']:.3f}"),
                ("Trained in", node.get("availability_zone", "?")),
                ("GPU", metrics.get("gpu_name") or "CPU"),
                ("Training time", f"{metrics['train_seconds']} s"),
            )) + "</div>"))
    display(Markdown(f"Run `{metrics['run_id']}`, published at "
                     f"[huggingface.co/{repo}](https://huggingface.co/{repo}/tree/{revision}). "
                     "RMSE is in $100,000s: lower is better."))

    # Score the held-out test rows with the published ONNX file alone, as a server would.
    data = load_california_housing(HERE / "data", REPO_COPY)
    seed = (_state["run"].settings.get("seed") if _state["run"] else None) or 42
    _, x_test, _, y_test = train_test_split(data.data, data.target, test_size=0.1, random_state=seed)
    session = onnxruntime.InferenceSession(hf_hub_download(repo, "model.onnx", revision=revision, force_download=True))
    predicted = session.run(["output"], {"input": x_test.astype(np.float32)})[0][:, 0]
    rmse = float(np.sqrt(np.mean((predicted - y_test) ** 2)))
    match = abs(rmse - metrics["test_rmse"]) < 1e-3
    display(_box("ok" if match else "warn", f"model.onnx scored here: RMSE {rmse:.4f}",
                 "Matches the run's own result: the published file is the trained model." if match
                 else "Differs from the run's result. Was it trained with a different seed?"))

    test = pd.DataFrame(x_test, columns=data.feature_names).assign(actual=y_test, predicted=predicted)
    test["error"] = test["predicted"] - test["actual"]
    fig = px.scatter(test, x="actual", y="predicted", opacity=0.4,
                     title="Predicted against actual median house value ($100,000s)")
    fig.add_shape(type="line", x0=0, y0=0, x1=5.2, y1=5.2, line={"dash": "dash", "color": "grey"})
    fig.update_layout(height=420)
    fig.show()
    fig = px.scatter_geo(test, lat="Latitude", lon="Longitude", color="error", color_continuous_scale="RdBu_r",
                         range_color=(-1.5, 1.5), scope="usa", title="Where it over- (red) and under-predicts (blue)")
    fig.update_geos(fitbounds="locations")
    fig.update_traces(marker={"size": 4})
    fig.update_layout(height=460)
    fig.show()

    log = widgets.Output()
    with log:
        try:
            print(_log_tail(metrics["run_id"], last=40))
        except Exception:
            print("(no log for this run)")
    accordion = widgets.Accordion(children=[log])
    accordion.set_title(0, "The run's log (click to open)")
    display(accordion)
    display(_box("info", "Serve it:", "model.onnx is self-contained, so any KServe v2 server (Triton, MLServer) can serve it. "
                 "See the model card on Hugging Face for a request body."))


# --- Optional: sweep --------------------------------------------------------------------

SWEEP_CHOICES = {
    "Learning rate 0.0003": {"learning-rate": 0.0003},
    "Learning rate 0.003": {"learning-rate": 0.003},
    "Wider: 128 × 128": {"hidden-sizes": [128, 128]},
    "Smaller: 32": {"hidden-sizes": [32]},
    "Deeper: 64 × 64 × 64": {"hidden-sizes": [64, 64, 64]},
}


def sweep() -> None:
    """Optional: train several variants at once, each on its own node, and rank them."""
    config = _config()
    boxes = [widgets.Checkbox(value=i < 4, description=label, indent=False) for i, label in enumerate(SWEEP_CHOICES)]
    button = _button("Start sweep", style="warning")
    display(widgets.HTML("<b>1.</b> Tick the variants to train (each adds to step 3's settings):"), widgets.VBox(boxes))
    max_nodes = config["pool"]["max_nodes"]
    if max_nodes < 2:
        display(_box("warn", f"The pool may only start {max_nodes} node, so the variants will run one after another.",
                     "To run them side by side, set <code>max_nodes</code> under <code>[pool]</code> in "
                     "<code>yd-training.toml</code> to the number of variants, then click Apply in step 2 again."))
    display(_box("info", "Clicking Start sweep will:",
                 f"• start one {config['instance_type']} per variant ({PRICE_PER_HOUR} an hour each), "
                 f"up to {max_nodes} at once, each wherever there's capacity;<br>"
                 "• publish each variant to its own branch, <b>not</b> main, then rank them by test RMSE."))
    area = widgets.Output()
    display(widgets.HTML("<b>2.</b> Click to start."), button, area)

    def start(_):
        chosen = [SWEEP_CHOICES[b.description] for b in boxes if b.value]
        if not chosen:
            display(_box("warn", "Tick at least one variant."))
            return
        button.disabled, button.description = True, "Sweeping..."
        base = _state["run"].settings if _state["run"] else {}
        runs = train.plan_runs(train.new_run_id(), base, chosen)
        _state["runs"] = runs
        ok = _start(runs, button, "Sweep again")
        _rank(runs)
        if not ok and AUTO:
            raise RuntimeError("a sweep run failed: see its log above")

    start = _guarded(start, button, "Try again", area)
    button.on_click(start)
    if AUTO_SWEEP:
        start(button)


def _rank(runs) -> None:
    rows = []
    for run in runs:
        row = {"variant": run.train_args, "run": run.run_id}
        try:
            m = _hub_json("metrics.json", run.revision)
            row.update({"test RMSE": round(m["test_rmse"], 4), "R²": round(m["test_r2"], 3),
                        "trained in": (m.get("node") or {}).get("availability_zone", ""), "branch": run.revision})
        except Exception:
            row["test RMSE"] = None
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("test RMSE", na_position="last").reset_index(drop=True)
    display(Markdown("**Ranked by test RMSE (lower is better):**"), table)
    best = table.iloc[0]
    if best["test RMSE"] is not None:
        display(_box("ok", f"Best: {best['variant']}",
                     f'<a href="https://huggingface.co/{os.environ["HF_REPO"]}/tree/{best["branch"]}" target="_blank">'
                     f"Its model, on branch {best['branch']}</a>"))


# --- Optional: clean up -----------------------------------------------------------------

def clean_up() -> None:
    """Optional: remove everything step 2 made. Never automatic."""
    config = _config()
    confirm = widgets.Checkbox(value=False, indent=False,
                               description=f"I want to remove the pool, templates, keyring and IAM user {config['iam_user']}")
    button = _button("Remove everything", style="danger")
    button.disabled = True
    result = widgets.Output()
    display(_box("warn", "This removes everything step 2 set up.",
                 "Your VPCs and the Hugging Face repository are not touched. Running step 2 again sets it all up afresh."),
            confirm, button, result)
    confirm.observe(lambda change: setattr(button, "disabled", not change["new"]), names="value")

    def remove(_):
        button.disabled, button.description = True, "Removing..."
        with result:
            client = make_client()
            try:
                configure.teardown(client, config, NAMESPACE, WORKER_TAG)
                display(_box("ok", "Removed."))
                button.description = "Removed"
            except (Exception, SystemExit) as error:
                display(_box("bad", "Clean up failed:", str(error)))
                button.disabled, button.description = False, "Try again"
            finally:
                client.close()

    button.on_click(remove)

