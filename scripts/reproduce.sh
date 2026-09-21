#!/bin/sh
set -eu

protocol_path=${1:-protocol/freeze_manifest.json}
if [ "$#" -gt 0 ]; then
  shift
fi

make setup
make verify
uv run shadowskillbench reproduce --protocol "$protocol_path" "$@"
