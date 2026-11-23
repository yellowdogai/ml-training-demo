#!/bin/bash
# Run as root on each node at boot, in whichever region it starts. Installs the
# YellowDog agent and makes sure it has the 'bash' task type. The node needs
# outbound internet access and nothing else: no IAM role, storage or open ports.

set -euo pipefail

# YellowDog's agent installer, always the latest version.
RESOURCES="https://raw.githubusercontent.com/yellowdog/resources/main/agent-install/linux"
YD_AGENT_HOME="/opt/yellowdog/agent"

SCRIPT="$(mktemp)"
trap 'rm -f "$SCRIPT"' EXIT
curl -fLsS --retry 5 --retry-connrefused -o "$SCRIPT" "$RESOURCES/yd-agent-installer.sh"
bash "$SCRIPT"

# The installer's default configuration includes 'bash'; add it if that changes.
if ! grep -q '"bash"' "$YD_AGENT_HOME/application.yaml"; then
  sed -i '/^yda.taskTypes:/a\  - name: "bash"\n    run: "/bin/bash"' "$YD_AGENT_HOME/application.yaml"
fi

echo "*** yd-training: node ready"

# No agent restart here: the installer has already started it, and a node can
# claim a task within seconds, which a restart would kill.
