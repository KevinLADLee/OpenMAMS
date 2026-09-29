#!/usr/bin/env bash
set -euo pipefail
NTN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$NTN_ROOT/runs/backhaul_400}"
geometry="${2:-$NTN_ROOT/examples/400sats}"
if [[ -e "$output" ]]; then
  echo "Output already exists: $output" >&2; exit 1
fi
mkdir -p -- "$output"
openmams-ntn endpoint --topology "$geometry/hong_kong.json" --link-direction uplink \
  --buildings "$NTN_ROOT/examples/400sats/buildings.csv" --output "$output/uplink"
# Use the same terminal-density scene at both endpoints.
openmams-ntn endpoint --topology "$geometry/istanbul.json" --link-direction downlink \
  --buildings "$NTN_ROOT/examples/400sats/buildings.csv" --open-site-user1 \
  --downlink-ground-rx-gain-dbi 40 --output "$output/downlink"
openmams-ntn combine --uplink-dir "$output/uplink" --downlink-dir "$output/downlink" \
  --output "$output/effective"
