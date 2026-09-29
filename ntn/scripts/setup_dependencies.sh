#!/usr/bin/env bash
# Fetch pinned upstream repositories. Source stays in ignored third_party/.
set -euo pipefail
NTN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---fetch}"
case "$mode" in
  --fetch|--leopath) ;;
  *) echo "Usage: $0 [--fetch|--leopath] [destination]" >&2; exit 2 ;;
esac
destination="${2:-$NTN_ROOT/third_party}"
mkdir -p -- "$destination"
fetch_pinned() {
  local name="$1" url="$2" ref="$3" target="$destination/$1"
  if [[ -e "$target" ]]; then
    if [[ ! -d "$target/.git" ]] || [[ "$(git -C "$target" rev-parse HEAD)" != "$ref" ]] ||
       [[ -n "$(git -C "$target" status --porcelain)" ]]; then
      echo "Refusing to overwrite existing/modified checkout: $target" >&2
      exit 1
    fi
  else
    git init -q "$target"
    git -C "$target" remote add origin "$url"
    git -C "$target" fetch --depth 1 origin "$ref"
    git -C "$target" checkout -q --detach FETCH_HEAD
  fi
  printf '%s %s\n' "$name" "$(git -C "$target" rev-parse HEAD)"
}
fetch_pinned LEOPath https://github.com/Fundacio-i2CAT/LEOPath.git af3a7d9bb060fab9eb02ee6ad8a13d7b269e2db1
fetch_pinned OpenNTN https://github.com/ant-uni-bremen/OpenNTN.git f294df0480f31a4148e8fe4af9b17dd4abcdb869
if [[ "$mode" == --leopath ]]; then
  # Only the TLE generator is used; no routing/plotting/dev dependency bundle.
  "${PYTHON:-python}" -m pip install -e "$NTN_ROOT[topology]"
  "${PYTHON:-python}" -m pip install --no-deps -e "$destination/LEOPath"
fi
echo "OpenNTN was fetched only. Its full PHY is optional; see its pinned README."
