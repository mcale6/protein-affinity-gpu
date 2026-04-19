#!/usr/bin/env bash

set -euo pipefail

VERSION_FILE="./src/protein_affinity_gpu/version.py"
BUMP_PART="${1:-patch}"
CURRENT_VERSION=$(awk -F'"' '/^__version__/{print $2}' "$VERSION_FILE")

IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"
major=$((10#$major))
minor=$((10#$minor))
patch=$((10#$patch))

case "$BUMP_PART" in
  major)
    major=$((major + 1))
    minor=0
    patch=0
    ;;
  minor)
    minor=$((minor + 1))
    patch=0
    ;;
  patch)
    patch=$((patch + 1))
    ;;
  none)
    ;;
  *)
    echo "Usage: ./update_pkg.sh [major|minor|patch|none]" >&2
    exit 1
    ;;
esac

NEW_VERSION="${major}.${minor}.${patch}"

if [ "$BUMP_PART" != "none" ]; then
  perl -0pi -e "s/__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" "$VERSION_FILE"
  echo "Updated version from ${CURRENT_VERSION} to ${NEW_VERSION}"
else
  echo "Keeping version ${CURRENT_VERSION}"
fi

rm -rf build dist
python3 -m build
