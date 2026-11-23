"""Set up everything training needs in AWS and YellowDog, from yd-training.toml.

  python configure.py --dry-run    # look up each region and show the plan; changes nothing
  python configure.py              # set up, or bring up to date; safe to re-run
  python configure.py --teardown   # remove everything this made

What it does, in order:

1. AWS: an IAM user that YellowDog launches instances as, limited by
   aws-policy.json to instances tagged Project=yd-training. IAM is global, so
   this is the same whatever regions you use.
2. YellowDog keyring: an access key for that user, stored in a keyring. The key
   is never printed or written to disk. A new keyring's password is printed once.
3. Regions: for each region in yd-training.toml, the AMI, the subnets that reach the
   internet in zones offering the instance type, and the VPC's default security
   group. Nothing is created in AWS here. From these it makes a YellowDog image
   family (an image per region), a compute source per subnet, and a compute
   requirement template that tries the sources in order.
4. Worker pool: replaces any pool with this worker tag with a new one that starts
   empty, scales up when a task is waiting and back to zero when idle. A pool
   copies its template and boot script when it is made, so this is how changes
   to either take effect. A pool with a task running is left alone.

Needs the AWS CLI, logged in as someone who can manage IAM (AWS_PROFILE in .env
works). The YellowDog CLI, for step 3, comes with requirements.txt.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from yellowdog_client.model import (
    AutoShutdown,
    AwsCredential,
    ComputeRequirementTemplateUsage,
    NodeWorkerTarget,
    ProvisionedWorkerPoolProperties,
    WorkerPoolSearch,
)

from yd_common import NAMESPACE, WORKER_TAG, make_client, missing_env

HERE = Path(__file__).resolve().parent
# Not config.toml: the YellowDog CLI reads a file of that name in the current directory as its own.
CONFIG = HERE / "yd-training.toml"
POLICY = HERE / "aws-policy.json"
USER_DATA = HERE / "worker-userdata.sh"
# What step 3 generates for the YellowDog CLI. Specific to your account, so not committed.
RESOURCES = HERE / "yellowdog-resources.json"
IMAGE_FAMILY = "ubuntu-24-nvidia"
# Every instance gets this tag, and the IAM policy only allows instances that have it.
PROJECT_TAG = {"Project": "yd-training"}
ENDED = {"SHUTDOWN", "TERMINATED"}


def log(message: str) -> None:
    print(message, flush=True)


# --- AWS --------------------------------------------------------------------------

def aws(*args: str, region: str | None = None):
    command = ["aws", "--output", "json", *(["--region", region] if region else []), *args]
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "AWS_PAGER": ""})
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"aws {' '.join(args[:2])} failed")
    return json.loads(result.stdout) if result.stdout.strip() else None


def account_id() -> str:
    return aws("sts", "get-caller-identity")["Account"]


def ensure_iam_user(user: str) -> None:
    try:
        aws("iam", "get-user", "--user-name", user)
        log(f"  IAM user {user} exists")
    except RuntimeError:
        aws("iam", "create-user", "--user-name", user, "--tags", "Key=Project,Value=yd-training")
        log(f"  created IAM user {user}")
    # Compact: IAM limits a user's inline policy to 2,048 characters.
    policy = json.dumps(json.loads(POLICY.read_text(encoding="utf-8").replace("ACCOUNT_ID", account_id())), separators=(",", ":"))
    aws("iam", "put-user-policy", "--user-name", user, "--policy-name", "yellowdog-provisioning",
        "--policy-document", policy)
    log(f"  applied {POLICY.name}")


# --- YellowDog keyring --------------------------------------------------------------

def ensure_credential(client, user: str, keyring: str, name: str, rotate: bool) -> str | None:
    """Store the IAM user's access key in the keyring. Returns a new keyring's password, shown only once."""
    password = None
    keyrings = {k.name: k for k in client.keyring_client.find_all_keyrings()}
    if keyring not in keyrings:
        created = client.keyring_client.add_keyring(keyring, "AWS credential YellowDog launches training nodes with")
        password = created.keyringPassword
        log(f"  created keyring {keyring}. Store its password now; it is not shown again: {password}")
        existing = set()
    else:
        existing = {c.name for c in client.keyring_client.get_keyring(keyrings[keyring].id).credentials or []}
    if name in existing and not rotate:
        log(f"  credential {keyring}/{name} exists (--rotate-key replaces it)")
        return password

    key = aws("iam", "create-access-key", "--user-name", user)["AccessKey"]
    credential = AwsCredential(
        name=name, accessKeyId=key["AccessKeyId"], secretAccessKey=key["SecretAccessKey"], description=f"IAM user {user}"
    )
    # A new key takes a few seconds to work everywhere in AWS, and YellowDog checks it.
    for attempt in range(6):
        try:
            client.keyring_client.put_credential_by_name(keyring, credential)
            break
        except Exception as error:  # the SDK raises its own types for a rejected credential
            if attempt == 5:
                aws("iam", "delete-access-key", "--user-name", user, "--access-key-id", key["AccessKeyId"])
                raise RuntimeError(f"YellowDog did not accept the new key, so it was deleted: {error}") from error
            time.sleep(10)
    log(f"  stored access key {key['AccessKeyId']} as {keyring}/{name}")
    # Keep only the key just stored: an IAM user can have two, and old ones are a risk.
    for old in aws("iam", "list-access-keys", "--user-name", user)["AccessKeyMetadata"]:
        if old["AccessKeyId"] != key["AccessKeyId"]:
            aws("iam", "delete-access-key", "--user-name", user, "--access-key-id", old["AccessKeyId"])
            log(f"  deleted the old access key {old['AccessKeyId']}")
    return password


# --- Regions ----------------------------------------------------------------------

@dataclass
class Subnet:
    id: str
    zone: str
    public: bool  # routes through an internet gateway, so the instance needs a public IP


@dataclass
class Region:
    name: str
    ami: str = ""
    ami_name: str = ""
    vpc: str = ""
    security_group: str = ""
    subnets: list[Subnet] = field(default_factory=list)
    problem: str = ""


def find_vpc(region: str, wanted: str | None) -> str:
    if not wanted or wanted == "default":
        filters = ["Name=is-default,Values=true"]
    elif wanted.startswith("vpc-"):
        filters = [f"Name=vpc-id,Values={wanted}"]
    else:
        filters = [f"Name=tag:Name,Values={wanted}"]
    vpcs = aws("ec2", "describe-vpcs", "--filters", *filters, region=region)["Vpcs"]
    if not vpcs and filters[0].startswith("Name=is-default"):
        raise LookupError("no default VPC; set AWS_VPC in .env to your VPC's Name tag or ID")
    if len(vpcs) != 1:
        raise LookupError(f"{'no' if not vpcs else len(vpcs)} VPC matching '{wanted or 'default'}'")
    return vpcs[0]["VpcId"]


def internet_routes(region: str, vpc: str) -> tuple[dict[str, str], str]:
    """Each explicitly associated subnet's route to 0.0.0.0/0, and the main route table's."""
    tables = aws("ec2", "describe-route-tables", "--filters", f"Name=vpc-id,Values={vpc}", region=region)["RouteTables"]
    by_subnet, main = {}, ""
    for table in tables:
        target = ""
        for route in table.get("Routes", []):
            if route.get("DestinationCidrBlock") == "0.0.0.0/0" and route.get("State") == "active":
                target = route.get("GatewayId") or route.get("NatGatewayId") or route.get("TransitGatewayId") or ""
        for association in table.get("Associations", []):
            if association.get("Main"):
                main = target
            elif association.get("SubnetId"):
                by_subnet[association["SubnetId"]] = target
    return by_subnet, main


def vpc_setting(config: dict) -> str | None:
    """AWS_VPC in .env wins over yd-training.toml; neither means each region's default VPC."""
    return os.environ.get("AWS_VPC") or config.get("vpc")


def discover(name: str, config: dict) -> Region:
    region = Region(name)
    instance_type = config["instance_type"]
    try:
        region.ami = aws("ssm", "get-parameter", "--name", config["ami_parameter"], region=region.name)["Parameter"]["Value"]
        region.ami_name = aws("ec2", "describe-images", "--image-ids", region.ami, region=region.name)["Images"][0]["Name"]
        region.vpc = find_vpc(region.name, vpc_setting(config))
        offered = {
            o["Location"]
            for o in aws("ec2", "describe-instance-type-offerings", "--location-type", "availability-zone",
                         "--filters", f"Name=instance-type,Values={instance_type}", region=region.name)["InstanceTypeOfferings"]
        }
        if not offered:
            raise LookupError(f"{instance_type} is not offered here")

        routes, main_route = internet_routes(region.name, region.vpc)
        for subnet in aws("ec2", "describe-subnets", "--filters", f"Name=vpc-id,Values={region.vpc}", region=region.name)["Subnets"]:
            route = routes.get(subnet["SubnetId"], main_route)
            if subnet["AvailabilityZone"] in offered and route:
                region.subnets.append(Subnet(subnet["SubnetId"], subnet["AvailabilityZone"], route.startswith("igw-")))
        region.subnets.sort(key=lambda s: s.zone)
        if not region.subnets:
            raise LookupError(f"no subnet with an internet route in a zone offering {instance_type}")

        region.security_group = aws(
            "ec2", "describe-security-groups",
            "--filters", f"Name=vpc-id,Values={region.vpc}", "Name=group-name,Values=default", region=region.name,
        )["SecurityGroups"][0]["GroupId"]
    except (LookupError, RuntimeError, KeyError, IndexError) as error:
        region.problem = str(error)
    return region


def discover_all(config: dict) -> list[Region]:
    names = config.get("regions", [])
    if not names:
        raise SystemExit(f"{CONFIG.name} lists no regions")
    log(f"looking up {len(names)} regions...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        regions = list(pool.map(lambda name: discover(name, config), names))
    for r in regions:
        if r.problem:
            log(f"  {r.name:15} SKIPPED: {r.problem}")
        else:
            log(f"  {r.name:15} {r.ami}  zones {', '.join(s.zone.removeprefix(r.name) for s in r.subnets)}")
    usable = [r for r in regions if not r.problem]
    if not usable:
        raise SystemExit("no usable region: see the reasons above")
    return usable


def template_name(config: dict) -> str:
    return f"{config['instance_type'].replace('.', '-')}-multi-region"


def build_resources(regions: list[Region], config: dict, namespace: str) -> list[dict]:
    """The YellowDog CLI specification: image family, a source per subnet, and the template."""
    instance_type = config["instance_type"]
    # The AMI's build date names the image group, so a new AMI is a new group.
    dates = {m.group(0) for r in regions if (m := re.search(r"20\d{6}", r.ami_name))}
    family = {
        "resource": "MachineImageFamily",
        "namespace": namespace,
        "name": IMAGE_FAMILY,
        "osType": "LINUX",
        "access": "PRIVATE",
        "imageGroups": [{
            "name": f"dlami-base-oss-nvidia-{max(dates) if dates else 'latest'}",
            "osType": "LINUX",
            "images": [
                {"name": r.name, "osType": "LINUX", "provider": "AWS", "providerImageId": r.ami, "regions": [r.name]}
                for r in regions
            ],
        }],
    }
    sources = []
    for r in regions:
        for subnet in r.subnets:
            sources.append({
                "resource": "ComputeSourceTemplate",
                "namespace": namespace,
                "description": f"{instance_type} on demand in {subnet.zone}",
                "source": {
                    "type": "co.yellowdog.platform.model.AwsInstancesComputeSource",
                    "name": f"{instance_type.replace('.', '-')}-{subnet.zone}",
                    "credential": config["credential"],
                    "region": r.name,
                    "availabilityZone": subnet.zone,
                    "subnetId": subnet.id,
                    "securityGroupId": r.security_group,
                    "assignPublicIp": subnet.public,
                    "instanceType": instance_type,
                    # "*" lets the worker pool supply the image family, which picks this region's AMI.
                    "imageId": "*",
                    "bootVolumeSizeGb": config.get("boot_volume_gb", 100),
                    "instanceTags": PROJECT_TAG,
                },
            })
    template = {
        "resource": "ComputeRequirementTemplate",
        "type": "co.yellowdog.platform.model.ComputeRequirementStaticTemplate",
        "namespace": namespace,
        "name": template_name(config),
        "description": f"{instance_type} in {len(regions)} regions, tried in order: {', '.join(r.name for r in regions)}",
        "strategyType": "co.yellowdog.platform.model.WaterfallProvisionStrategy",
        "imagesId": "*",
        "sources": [{"sourceTemplateId": f"{namespace}/{s['source']['name']}"} for s in sources],
    }
    return [family, *sources, template]


def yd_path() -> str | None:
    """The YellowDog CLI: the one installed next to this Python, else any on the PATH."""
    local = Path(sys.executable).parent / ("yd.exe" if os.name == "nt" else "yd")
    return str(local) if local.exists() else shutil.which("yd")


def yd_cli(*args: str) -> None:
    yd = yd_path()
    if not yd:
        raise SystemExit("needs the YellowDog CLI: python -m pip install -r requirements.txt")
    # The CLI reads YD_KEY, YD_SECRET and YD_URL from the environment, which yd_common filled from .env.
    if subprocess.run([yd, *args, "--yes", "--quiet"]).returncode != 0:
        raise SystemExit(f"yd {' '.join(args)} failed")


# --- Worker pool ------------------------------------------------------------------

def our_pools(client, namespace: str, worker_tag: str):
    # Searched by namespace: listing every pool needs a global permission.
    for summary in client.worker_pool_client.get_worker_pools(WorkerPoolSearch(namespace=namespace)).list_all():
        if summary.status is None or summary.status.name in ENDED:
            continue
        pool = client.worker_pool_client.get_worker_pool_by_id(summary.id)
        if getattr(pool, "properties", None) and pool.properties.workerTag == worker_tag:
            yield summary


def pool_sources(client, namespace: str, worker_tag: str) -> list[dict]:
    """The newest pool's compute sources in waterfall order: where it is trying, and why each failed."""
    pools = sorted(our_pools(client, namespace, worker_tag), key=lambda p: p.createdTime or datetime.min)
    if not pools:
        return []
    pool = client.worker_pool_client.get_worker_pool_by_id(pools[-1].id)
    requirement = client.compute_client.get_compute_requirement_by_id(pool.computeRequirementId)
    rows = []
    for source in requirement.provisionStrategy.sources:
        message = getattr(source, "statusMessage", None) or ""
        if "sufficient" in message and "capacity" in message:
            message = "no capacity"
        rows.append({
            "source": source.name,
            "region": getattr(source, "region", ""),
            "zone": getattr(source, "availabilityZone", ""),
            "status": source.status.name if source.status else "",
            "instances": source.instanceSummary.aliveCount if source.instanceSummary else 0,
            "message": message,
        })
    return rows


def build_pool_request(config: dict, namespace: str, worker_tag: str, template_id: str, family_id: str, name: str):
    pool = config.get("pool", {})
    usage = ComputeRequirementTemplateUsage(
        templateId=template_id,
        requirementName=name,
        requirementNamespace=namespace,
        requirementTag=worker_tag,
        targetInstanceCount=0,
        # The family has an image per region; the platform picks the one for the region a node starts in.
        imagesId=family_id,
        userData=USER_DATA.read_text(encoding="utf-8").replace("\r\n", "\n"),
    )
    properties = ProvisionedWorkerPoolProperties(
        createNodeWorkers=NodeWorkerTarget.per_node(1),
        minNodes=0,
        maxNodes=pool.get("max_nodes", 1),
        nodeBootTimeout=timedelta(minutes=pool.get("node_boot_minutes", 15)),
        idleNodeShutdown=AutoShutdown(enabled=True, timeout=timedelta(minutes=pool.get("idle_node_minutes", 5))),
        # Must be explicit: left unset, the platform shuts an idle pool down after
        # 30 minutes, and an empty pool counts as idle, so it would never scale up.
        idlePoolShutdown=AutoShutdown(enabled=False),
        workerTag=worker_tag,
    )
    return usage, properties


def replace_pool(client, config: dict, namespace: str, worker_tag: str) -> None:
    existing = list(our_pools(client, namespace, worker_tag))
    busy = [p for p in existing if p.claimedWorkerCount or p.workingWorkerCount]
    if busy:
        log(f"  left worker pool {busy[0].name} alone: it is running a task. Re-run when it is idle.")
        return
    template = client.compute_client.get_compute_requirement_template_by_name(namespace, template_name(config))
    family = client.images_client.get_image_family_by_name(namespace, IMAGE_FAMILY)
    # A shut-down pool keeps its name, so each pool gets a new one.
    name = f"{worker_tag}-{datetime.now():%Y%m%d-%H%M%S}"
    usage, properties = build_pool_request(config, namespace, worker_tag, template.id, family.id, name)
    created = client.worker_pool_client.provision_worker_pool(usage, properties)
    log(f"  created worker pool {name}: 0 to {properties.maxNodes} nodes, tag {worker_tag}")
    for old in existing:
        client.worker_pool_client.shutdown_worker_pool_by_id(old.id)
        log(f"  shut down the previous pool {old.name}")
    if not created.id:
        raise RuntimeError("the platform did not return the new pool")


# --- Teardown ---------------------------------------------------------------------

def teardown(client, config: dict, namespace: str, worker_tag: str) -> None:
    keyring, _, _ = config["credential"].partition("/")
    user = config["iam_user"]
    log("worker pools")
    for pool in our_pools(client, namespace, worker_tag):
        client.worker_pool_client.shutdown_worker_pool_by_id(pool.id)
        log(f"  shut down {pool.name}")
    log("YellowDog templates and image family")
    if not RESOURCES.exists():
        RESOURCES.write_text(json.dumps(build_resources(discover_all(config), config, namespace), indent=2), encoding="utf-8")
    yd_cli("resource", "remove", "--namespace", namespace, str(RESOURCES))
    RESOURCES.unlink()
    log(f"keyring {keyring}")
    try:
        client.keyring_client.delete_keyring_by_name(keyring)
        log("  deleted")
    except Exception as error:  # already gone
        log(f"  not deleted: {error}")
    log(f"IAM user {user}")
    try:
        for key in aws("iam", "list-access-keys", "--user-name", user)["AccessKeyMetadata"]:
            aws("iam", "delete-access-key", "--user-name", user, "--access-key-id", key["AccessKeyId"])
        aws("iam", "delete-user-policy", "--user-name", user, "--policy-name", "yellowdog-provisioning")
        aws("iam", "delete-user", "--user-name", user)
        log("  deleted, with its access keys and policy")
    except RuntimeError as error:
        log(f"  not deleted: {error}")


# --- Main -------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--worker-tag", default=WORKER_TAG, help="Tasks reach the pool through this tag.")
    parser.add_argument("--dry-run", action="store_true", help="Look up the regions and show the plan; change nothing.")
    parser.add_argument("--rotate-key", action="store_true", help="Replace the AWS access key in the keyring.")
    parser.add_argument("--keep-pool", action="store_true", help="Don't replace the worker pool.")
    parser.add_argument("--teardown", action="store_true", help="Remove the pool, YellowDog resources, keyring and IAM user.")
    parser.add_argument("--yes", action="store_true", help="Don't ask before --teardown.")
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    keyring, _, credential = config["credential"].partition("/")
    user = config["iam_user"]

    if args.dry_run:
        regions = discover_all(config)
        resources = build_resources(regions, config, args.namespace)
        log(f"\nwould set up, in namespace {args.namespace}:")
        log(f"  IAM user {user} with {POLICY.name}, and its access key in keyring {keyring} as {credential}")
        log(f"  image family {IMAGE_FAMILY}, {len(resources) - 2} compute sources in {len(regions)} regions,"
            f" template {template_name(config)}")
        log(f"  a worker pool with tag {args.worker_tag}, replacing any idle one")
        return 0

    if missing_env("YD_KEY", "YD_SECRET"):
        return 2
    client = make_client()
    try:
        if args.teardown:
            if not args.yes and input(f"remove everything in namespace {args.namespace} and IAM user {user}? [y/N] ") != "y":
                return 1
            teardown(client, config, args.namespace, args.worker_tag)
            return 0

        log("1. AWS")
        ensure_iam_user(user)
        log("2. YellowDog keyring")
        ensure_credential(client, user, keyring, credential, args.rotate_key)
        log("3. regions")
        regions = discover_all(config)
        if not regions:
            log("no usable region")
            return 1
        RESOURCES.write_text(json.dumps(build_resources(regions, config, args.namespace), indent=2) + "\n", encoding="utf-8")
        yd_cli("resource", "create", "--namespace", args.namespace, str(RESOURCES))
        log(f"  template {args.namespace}/{template_name(config)}: {sum(len(r.subnets) for r in regions)} sources")
        log("4. worker pool")
        if args.keep_pool:
            log("  kept (--keep-pool)")
        else:
            replace_pool(client, config, args.namespace, args.worker_tag)
    finally:
        client.close()
    log("\ndone. Next: python train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
