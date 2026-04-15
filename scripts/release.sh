#!/usr/bin/env bash

set -euo pipefail

TMP_ARCHIVE=

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel 2>/dev/null || true)

if [[ -z "$REPO_ROOT" ]]; then
  printf 'error: scripts/release.sh must run from inside the parakeet-cli repo\n' >&2
  exit 1
fi

cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  scripts/release.sh prepare <version>
  scripts/release.sh tag <version>
  scripts/release.sh update-tap <version> --tap-dir <path>

Commands:
  prepare
      Update Cargo.toml and Cargo.lock to <version> and scaffold
      docs/releases/v<version>.md if it does not already exist.

  tag
      Create an annotated git tag named v<version> on the current HEAD.

  update-tap
      Download the GitHub source tarball for v<version>, compute its sha256,
      and update Formula/parakeet-cli.rb in the given Homebrew tap repo.

Examples:
  scripts/release.sh prepare 0.1.4
  scripts/release.sh tag 0.1.4
  scripts/release.sh update-tap 0.1.4 --tap-dir ../homebrew-tap
EOF
}

die() {
  printf 'error: %s\n' "$1" >&2
  exit 1
}

cleanup_archive() {
  if [[ -n ${TMP_ARCHIVE:-} ]]; then
    rm -f "$TMP_ARCHIVE"
    TMP_ARCHIVE=
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

normalize_version() {
  local input=$1
  input=${input#v}

  [[ $input =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must look like 0.1.4 or v0.1.4"
  printf '%s\n' "$input"
}

release_tag() {
  printf 'v%s\n' "$(normalize_version "$1")"
}

release_notes_path() {
  printf '%s/docs/releases/%s.md\n' "$REPO_ROOT" "$(release_tag "$1")"
}

ensure_file() {
  [[ -f "$1" ]] || die "expected file not found: $1"
}

write_release_notes_template() {
  local version=$1
  local tag
  local notes_file

  tag=$(release_tag "$version")
  notes_file=$(release_notes_path "$version")

  if [[ -f "$notes_file" ]]; then
    printf 'release notes already exist: %s\n' "$notes_file"
    return 0
  fi

  cat >"$notes_file" <<EOF
## Summary


## What's Changed

- TODO

## Install / Upgrade

\`\`\`bash
brew update
brew upgrade lucataco/tap/parakeet-cli
\`\`\`

Or build from source:

\`\`\`bash
git clone https://github.com/lucataco/parakeet-cli.git
cd parakeet-cli
cargo build --release --bin parakeet
\`\`\`

## Quick Start

\`\`\`bash
parakeet download
parakeet devices
parakeet listen
\`\`\`

## Notes

- macOS on Apple Silicon is the supported target
- replace this placeholder summary before publishing ${tag}
EOF

  printf 'created release notes template: %s\n' "$notes_file"
}

set_package_version() {
  local version=$1

  ensure_file Cargo.toml
  ensure_file Cargo.lock

  NEW_VERSION="$version" perl -0pi -e 's/(\[package\]\nname = "parakeet-cli"\nversion = ")[^"]+(")/$1 . $ENV{NEW_VERSION} . $2/se' Cargo.toml
  NEW_VERSION="$version" perl -0pi -e 's/(name = "parakeet-cli"\nversion = ")[^"]+(")/$1 . $ENV{NEW_VERSION} . $2/ge' Cargo.lock

  printf 'updated crate version to %s in Cargo.toml and Cargo.lock\n' "$version"
}

prepare_release() {
  local version=$1

  require_cmd git
  require_cmd perl

  set_package_version "$version"
  write_release_notes_template "$version"

  printf '\nnext:\n'
  printf '  1. Review docs/releases/%s.md\n' "$(release_tag "$version")"
  printf '  2. Commit the version bump and notes\n'
  printf '  3. Push main, then run scripts/release.sh tag %s\n' "$version"
}

create_tag() {
  local version=$1
  local tag

  require_cmd git

  tag=$(release_tag "$version")

  if git rev-parse --verify "refs/tags/$tag" >/dev/null 2>&1; then
    die "tag already exists: $tag"
  fi

  git tag -a "$tag" -m "$tag"
  printf 'created tag %s on %s\n' "$tag" "$(git rev-parse --short HEAD)"
}

update_tap() {
  local version=$1
  local tap_dir=$2
  local tag
  local formula
  local tarball_url
  local sha

  require_cmd curl
  require_cmd git
  require_cmd mktemp
  require_cmd perl
  require_cmd shasum

  tag=$(release_tag "$version")
  formula="$tap_dir/Formula/parakeet-cli.rb"
  tarball_url="https://github.com/lucataco/parakeet-cli/archive/refs/tags/${tag}.tar.gz"

  [[ -d "$tap_dir" ]] || die "tap directory not found: $tap_dir"
  ensure_file "$formula"

  TMP_ARCHIVE=$(mktemp -t "parakeet-${tag}.XXXXXX.tar.gz")
  trap cleanup_archive EXIT

  curl -fsSL "$tarball_url" -o "$TMP_ARCHIVE"
  sha=$(shasum -a 256 "$TMP_ARCHIVE" | cut -d ' ' -f 1)

  RELEASE_TAG="$tag" RELEASE_SHA="$sha" perl -0pi -e 's{^  url ".*"\n  sha256 ".*"$}{  url "https://github.com/lucataco/parakeet-cli/archive/refs/tags/$ENV{RELEASE_TAG}.tar.gz"\n  sha256 "$ENV{RELEASE_SHA}"}m' "$formula"

  printf 'updated %s to %s\n' "$formula" "$tag"
  printf 'sha256: %s\n' "$sha"

  cleanup_archive
  trap - EXIT
}

main() {
  local command=${1:-}

  case "$command" in
    prepare)
      [[ $# -eq 2 ]] || die 'usage: scripts/release.sh prepare <version>'
      prepare_release "$(normalize_version "$2")"
      ;;
    tag)
      [[ $# -eq 2 ]] || die 'usage: scripts/release.sh tag <version>'
      create_tag "$(normalize_version "$2")"
      ;;
    update-tap)
      [[ $# -eq 4 && $3 == "--tap-dir" ]] || die 'usage: scripts/release.sh update-tap <version> --tap-dir <path>'
      update_tap "$(normalize_version "$2")" "$4"
      ;;
    help|-h|--help|'')
      usage
      ;;
    *)
      die "unknown command: $command"
      ;;
  esac
}

main "$@"
