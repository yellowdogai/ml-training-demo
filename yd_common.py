"""What the two scripts share: settings from .env, and the platform client."""

import os
import sys

from dotenv import load_dotenv
from yellowdog_client import PlatformClient
from yellowdog_client.model import ApiKey, ServicesSchema

# Real environment variables win over .env.
load_dotenv()

DEFAULT_URL = "https://portal.yellowdog.ai/api"
DEFAULT_NAMESPACE = "development"
# The link between the two scripts: tasks only run on workers with this tag.
DEFAULT_WORKER_TAG = "yd-training"

NAMESPACE = os.environ.get("YD_NAMESPACE", DEFAULT_NAMESPACE)
WORKER_TAG = os.environ.get("YD_WORKER_TAG", DEFAULT_WORKER_TAG)


def missing_env(*names: str) -> bool:
    """Report any unset variables; True means the caller should stop."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        print(f"set {', '.join(missing)} in .env or the environment", file=sys.stderr)
    return bool(missing)


def make_client() -> PlatformClient:
    return PlatformClient.create(
        ServicesSchema(defaultUrl=os.environ.get("YD_URL", DEFAULT_URL)),
        ApiKey(os.environ["YD_KEY"], os.environ["YD_SECRET"]),
    )
