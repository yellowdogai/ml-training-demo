#!/usr/bin/env bash
# Prepares this project on Linux or macOS. Run it from the project folder:
#
#   ./setup.sh
#
# It installs what's missing, without sudo on Linux:
#   - uv, which installs Python 3.12 (into ~/.local/bin and uv's own folder)
#   - the AWS CLI v2 (into ~/.local/aws-cli, linked from ~/.local/bin)
#   - .venv, with requirements.txt (including the YellowDog CLI)
#   - .env, copied from .env.example if you don't have one
#
# Safe to re-run: anything already installed is left alone.

set -euo pipefail

UV_VERSION=0.13.7
PYTHON_VERSION=3.12
BIN="$HOME/.local/bin"
cd "$(dirname "$0")"
export PATH="$BIN:$PATH"

step() { printf '\n=== %s\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
have curl || { echo "needs curl; install it with your package manager, then run this again" >&2; exit 1; }

step "uv ${UV_VERSION} (installs Python)"
if have uv; then
  echo "already installed: $(uv --version)"
else
  curl --proto '=https' --tlsv1.2 -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_NO_MODIFY_PATH=1 sh
fi

step "Python ${PYTHON_VERSION}"
uv python install --no-bin "$PYTHON_VERSION"   # just for .venv: no python3.12 shortcut on the PATH

step "AWS CLI v2"
if have aws; then
  echo "already installed: $(aws --version)"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  if have brew; then
    brew install awscli
  else
    curl -fsSL -o /tmp/AWSCLIV2.pkg "https://awscli.amazonaws.com/AWSCLIV2.pkg"
    echo "installing the AWS CLI for all users, which asks for your password"
    sudo installer -pkg /tmp/AWSCLIV2.pkg -target /
    rm -f /tmp/AWSCLIV2.pkg
  fi
else
  arch="$(uname -m)"   # x86_64 or aarch64
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  curl -fsSL -o "$work/awscli.zip" "https://awscli.amazonaws.com/awscli-exe-linux-${arch}.zip"
  if have unzip; then
    unzip -q "$work/awscli.zip" -d "$work"
  else
    # No unzip: Python can extract it, but drops the files' execute bits, so restore them.
    uv run --no-project --python "$PYTHON_VERSION" python -m zipfile -e "$work/awscli.zip" "$work"
    chmod -R +x "$work/aws"
  fi
  "$work/aws/install" --install-dir "$HOME/.local/aws-cli" --bin-dir "$BIN" --update
  echo "installed: $(aws --version)"
fi

step "Project environment, .venv"
if [[ -x .venv/bin/python ]]; then
  echo "already exists: $(.venv/bin/python --version)"
else
  uv venv --seed --python "$PYTHON_VERSION" .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt
# Strip notebook outputs, which can hold account details, from commits.
[[ -d .git ]] && .venv/bin/python -m nbstripout --install

step ".env"
if [[ -f .env ]]; then
  echo "already exists; left as it is"
else
  cp .env.example .env
  echo "created from .env.example: fill it in"
fi

cat <<EOF

=== Done. Next:
  1. Fill in .env: your YellowDog key, Hugging Face repository and token.
  2. Sign in to AWS, if you haven't:  aws configure sso   (or aws configure)
  3. Activate the environment in each new terminal:  source .venv/bin/activate
  4. Then:  jupyter lab yd-training.ipynb    or    python configure.py --dry-run
EOF
# A new login shell shows whether ~/.local/bin is on the PATH in new terminals.
if ! grep -q "$BIN" <<<"$(bash -lc 'echo $PATH' 2>/dev/null)"; then
  echo
  echo "Note: add $BIN to your PATH (for example in ~/.bashrc) to use aws and uv in new terminals."
fi
