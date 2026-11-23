"""configure.py against a stubbed AWS CLI and a stubbed platform, with the real SDK models.

  python -m unittest discover tests
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import requests.adapters

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configure  # noqa: E402

import os  # noqa: E402

# A developer's .env may name their own VPC; the tests give the VPC in each config.
os.environ.pop("AWS_VPC", None)
from yellowdog_client import PlatformClient  # noqa: E402
from yellowdog_client.model import ApiKey, ServicesSchema  # noqa: E402

CONFIG = {
    "credential": "kr/cred",
    "instance_type": "g5.xlarge",
    "ami_parameter": "/aws/service/x",
    "vpc": "demo",
}


def fake_aws(*args, region=None):
    """One VPC: subnet a public, b private behind NAT, c with no route, d in a zone without g5."""
    command = args[:2]
    if command == ("ssm", "get-parameter"):
        return {"Parameter": {"Value": f"ami-{region}"}}
    if command == ("ec2", "describe-images"):
        return {"Images": [{"Name": "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261120"}]}
    if command == ("ec2", "describe-vpcs"):
        return {"Vpcs": [{"VpcId": "vpc-1"}]} if "Name=tag:Name,Values=demo" in args else {"Vpcs": []}
    if command == ("ec2", "describe-instance-type-offerings"):
        return {"InstanceTypeOfferings": [{"Location": f"{region}{z}"} for z in "abc"]}
    if command == ("ec2", "describe-route-tables"):
        return {
            "RouteTables": [
                {"Associations": [{"Main": True}], "Routes": []},
                {"Associations": [{"SubnetId": "subnet-a"}],
                 "Routes": [{"DestinationCidrBlock": "0.0.0.0/0", "GatewayId": "igw-1", "State": "active"}]},
                {"Associations": [{"SubnetId": "subnet-b"}],
                 "Routes": [{"DestinationCidrBlock": "0.0.0.0/0", "NatGatewayId": "nat-1", "State": "active"}]},
            ]
        }
    if command == ("ec2", "describe-subnets"):
        return {"Subnets": [{"SubnetId": f"subnet-{z}", "AvailabilityZone": f"{region}{z}"} for z in "dcba"]}
    if command == ("ec2", "describe-security-groups"):
        return {"SecurityGroups": [{"GroupId": "sg-default"}]}
    raise AssertionError(f"unexpected aws call {args}")


class SetupRegionsTest(unittest.TestCase):
    def discover(self, name, **config):
        with mock.patch.object(configure, "aws", side_effect=fake_aws):
            return configure.discover(name, {**CONFIG, **config})

    def test_keeps_only_subnets_that_reach_the_internet_where_the_type_is_offered(self):
        region = self.discover("eu-west-2")
        self.assertEqual(region.problem, "")
        self.assertEqual(
            [(s.id, s.public) for s in region.subnets], [("subnet-a", True), ("subnet-b", False)]
        )
        self.assertEqual((region.ami, region.vpc, region.security_group), ("ami-eu-west-2", "vpc-1", "sg-default"))

    def test_a_region_without_the_vpc_is_skipped_not_fatal(self):
        region = self.discover("us-east-1", vpc="missing")
        self.assertIn("no VPC matching 'missing'", region.problem)

    def test_resources_try_regions_in_the_order_given(self):
        regions = [self.discover("eu-west-2"), self.discover("us-east-1")]
        resources = configure.build_resources(regions, CONFIG, "ns")

        family, *sources, template = resources
        images = family["imageGroups"][0]["images"]
        self.assertEqual(family["imageGroups"][0]["name"], "dlami-base-oss-nvidia-20261120")
        self.assertEqual([(i["providerImageId"], i["regions"]) for i in images],
                         [("ami-eu-west-2", ["eu-west-2"]), ("ami-us-east-1", ["us-east-1"])])

        self.assertEqual(
            [s["sourceTemplateId"] for s in template["sources"]],
            ["ns/g5-xlarge-eu-west-2a", "ns/g5-xlarge-eu-west-2b", "ns/g5-xlarge-us-east-1a", "ns/g5-xlarge-us-east-1b"],
        )
        self.assertEqual(template["name"], "g5-xlarge-multi-region")
        self.assertEqual(template["strategyType"], "co.yellowdog.platform.model.WaterfallProvisionStrategy")
        first = sources[0]["source"]
        self.assertEqual(first["imageId"], "*")
        self.assertTrue(first["assignPublicIp"])
        self.assertFalse(sources[1]["source"]["assignPublicIp"])
        self.assertNotIn("iamInstanceProfileArn", first, "nodes need no IAM role")


class WorkerPoolTest(unittest.TestCase):
    def request(self, config=None):
        return configure.build_pool_request(config or {}, "ns", "tag", "ydid:crt:1", "ydid:imgfam:1", "pool-1")

    def test_sends_a_pool_that_can_sit_at_zero(self):
        # Caught at the HTTP adapter: the JSON the real SDK sends. 0 and False are falsy;
        # if the SDK dropped them, the platform would apply its 30 minute idle pool
        # shutdown and the pool would never scale up.
        sent = []

        def capture(_adapter, request, **_kwargs):
            sent.append(request)
            raise requests.exceptions.InvalidURL("stop")

        with mock.patch.object(requests.adapters.HTTPAdapter, "send", capture):
            client = PlatformClient.create(ServicesSchema(defaultUrl="https://x.invalid/api"), ApiKey("a", "b"))
            with self.assertRaises(Exception):
                client.worker_pool_client.provision_worker_pool(*self.request())
        self.assertTrue(sent[0].url.endswith("/workerPools/provisioned/template"))
        body = json.loads(sent[0].body)
        self.assertEqual(body["requirementTemplateUsage"]["targetInstanceCount"], 0)
        self.assertNotIn("\r", body["requirementTemplateUsage"]["userData"])
        properties = body["provisionedProperties"]
        self.assertEqual(properties["minNodes"], 0)
        self.assertEqual(properties["maxNodes"], 1)
        self.assertEqual(properties["idlePoolShutdown"], {"enabled": False})
        self.assertEqual(properties["idleNodeShutdown"], {"enabled": True, "timeout": "PT5M"})
        self.assertEqual(properties["workerTag"], "tag")

    def test_pool_settings_come_from_config(self):
        _, properties = self.request({"pool": {"max_nodes": 3, "idle_node_minutes": 2}})
        self.assertEqual(properties.maxNodes, 3)
        self.assertEqual(properties.idleNodeShutdown.timeout.total_seconds(), 120)

    def test_a_pool_running_a_task_is_not_replaced(self):
        client = mock.MagicMock()
        busy = SimpleNamespace(id="ydid:wrkrpool:1", name="old", namespace="ns", claimedWorkerCount=1,
                               workingWorkerCount=1, status=SimpleNamespace(name="RUNNING"))
        client.worker_pool_client.get_worker_pools.return_value.list_all.return_value = [busy]
        client.worker_pool_client.get_worker_pool_by_id.return_value = SimpleNamespace(
            properties=SimpleNamespace(workerTag="tag"))
        with mock.patch("builtins.print"):
            configure.replace_pool(client, {"instance_type": "g5.xlarge"}, "ns", "tag")
        client.worker_pool_client.provision_worker_pool.assert_not_called()
        client.worker_pool_client.shutdown_worker_pool_by_id.assert_not_called()

    def test_an_idle_pool_is_replaced_after_the_new_one_exists(self):
        client = mock.MagicMock()
        idle = SimpleNamespace(id="ydid:wrkrpool:1", name="old", namespace="ns", claimedWorkerCount=0,
                               workingWorkerCount=0, status=SimpleNamespace(name="EMPTY"))
        client.worker_pool_client.get_worker_pools.return_value.list_all.return_value = [idle]
        client.worker_pool_client.get_worker_pool_by_id.return_value = SimpleNamespace(
            properties=SimpleNamespace(workerTag="tag"))
        with mock.patch("builtins.print"):
            configure.replace_pool(client, {"instance_type": "g5.xlarge"}, "ns", "tag")
        client.compute_client.get_compute_requirement_template_by_name.assert_called_once_with("ns", "g5-xlarge-multi-region")
        client.worker_pool_client.provision_worker_pool.assert_called_once()
        client.worker_pool_client.shutdown_worker_pool_by_id.assert_called_once_with("ydid:wrkrpool:1")

    def test_user_data_needs_no_storage_and_no_installer_patch(self):
        usage, _ = self.request()
        self.assertTrue(usage.userData.startswith("#!/bin/bash\n"))
        self.assertNotIn("rclone", usage.userData)
        self.assertNotIn("sed -i -e", usage.userData)


class PolicyTest(unittest.TestCase):
    def test_instances_are_only_launched_and_managed_with_the_project_tag(self):
        policy = json.loads(configure.POLICY.read_text(encoding="utf-8"))
        statements = {s["Sid"]: s for s in policy["Statement"]}
        launch = statements["LaunchOnlyTaggedInstances"]
        self.assertEqual(launch["Condition"], {"StringEquals": {"aws:RequestTag/Project": "yd-training"}})
        manage = statements["ManageOnlyTaggedInstances"]
        self.assertIn("ec2:TerminateInstances", manage["Action"])
        self.assertEqual(manage["Condition"], {"StringEquals": {"aws:ResourceTag/Project": "yd-training"}})
        self.assertEqual(configure.PROJECT_TAG, {"Project": "yd-training"})
        unconditioned = [a for s in policy["Statement"] if "Condition" not in s
                         for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])]
        for action in ("ec2:RunInstances", "ec2:TerminateInstances", "ec2:CreateTags", "iam:PassRole"):
            if action == "ec2:RunInstances":
                continue  # allowed unconditioned only on images, subnets and the like, checked below
            self.assertNotIn(action, unconditioned)
        # Everything a launch uses (images, subnets, volumes...) except the instance itself.
        self.assertTrue(statements["WhatALaunchUses"]["NotResource"].endswith(":instance/*"))


if __name__ == "__main__":
    unittest.main()
