#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$SCRIPT_DIR/plugin"
MODEL_DIR="$PLUGIN_DIR/models"
INSTALL_DIR="${CLEARVOICE_PLUGIN_DIR:-$HOME/.local/lib/clearvoice/ladspa}"
PLUGIN_SO="$PLUGIN_DIR/target/release/libclearvoice_ladspa.so"
DEST="$INSTALL_DIR/libclearvoice_ladspa.so"
REVISION_FILE="$DEST.revision"
RELEASE_API="https://api.github.com/repos/aask1357/fastenhancer/releases/tags/onnx-48khz-v1"
FASTENHANCER_B_SHA256=70e23bba3d41e80d30ebc5eba39d9df64f0e0315f31c772022bb17576c4d96bf
FASTENHANCER_S_SHA256=f04ece2beed330da367264c54cedded62f65a117fbde5c005d3a88fc796d0ba3
FASTENHANCER_M_SHA256=c7da800810b583f4734d757c6e14d235f3eec81476121b595743e5866b66efa2
RELEASE_JSON=""
TEMP_FILES=()

cleanup() {
    for file in "${TEMP_FILES[@]}"; do
        rm -f "$file"
    done
}
trap cleanup EXIT

verify_model() {
    local file="$1" expected="$2"
    if [[ ! -f "$file" ]]; then
        echo "Missing model file: $file" >&2
        return 1
    fi
    printf '%s  %s\n' "$expected" "$file" | sha256sum --check
}

if [[ "${1:-}" == "--verify-model" ]]; then
    if [[ "$#" != 3 ]]; then
        echo "Usage: $0 --verify-model FILE SHA256" >&2
        exit 2
    fi
    verify_model "$2" "$3"
    exit $?
fi

for cmd in cargo curl git install python3 sha256sum; do
    command -v "$cmd" &>/dev/null || {
        echo "Missing build command: $cmd" >&2
        exit 1
    }
done

release_asset_url() {
    local asset_name="$1"
    python3 -c '
import json, sys
release = json.load(sys.stdin)
asset = next((item for item in release.get("assets", []) if item.get("name") == sys.argv[1]), None)
if not asset or not asset.get("browser_download_url"):
    raise SystemExit(f"Release asset not found: {sys.argv[1]}")
print(asset["browser_download_url"])
' "$asset_name" <<<"$RELEASE_JSON"
}

ensure_model() {
    local name="$1" expected="$2" destination="$MODEL_DIR/$1" url temporary
    if [[ -f "$destination" ]] && \
        printf '%s  %s\n' "$expected" "$destination" | sha256sum --check --status; then
        echo "Model verified: $destination"
        return
    fi

    # Cached in this shell; release_asset_url runs in a $(...) subshell.
    [[ -n "$RELEASE_JSON" ]] || RELEASE_JSON="$(curl -fsSL "$RELEASE_API")"
    url="$(release_asset_url "$name")"
    temporary="$(mktemp "$MODEL_DIR/$name.tmp.XXXXXX")"
    TEMP_FILES+=("$temporary")
    curl -fsSL "$url" -o "$temporary"
    verify_model "$temporary" "$expected"
    mv -f "$temporary" "$destination"
    echo "Fetched verified model: $destination"
}

mkdir -p "$MODEL_DIR"
ensure_model fastenhancer_b.onnx "$FASTENHANCER_B_SHA256"
ensure_model fastenhancer_s.onnx "$FASTENHANCER_S_SHA256"
ensure_model fastenhancer_m.onnx "$FASTENHANCER_M_SHA256"

(cd "$PLUGIN_DIR" && cargo build --release --locked --target-dir "$PLUGIN_DIR/target")

mkdir -p "$INSTALL_DIR"
if [[ -f "$DEST" ]] && cmp -s "$PLUGIN_SO" "$DEST"; then
    echo "Plugin bytes unchanged: $DEST"
else
    candidate="$(mktemp "$DEST.tmp.XXXXXX")"
    TEMP_FILES+=("$candidate")
    install -m755 "$PLUGIN_SO" "$candidate"
    mv -f "$candidate" "$DEST"
    echo "Installed ClearVoice LADSPA plugin: $DEST"
fi

revision="$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$SCRIPT_DIR" status --porcelain -- plugin)" ]]; then
    dirty=true
else
    dirty=false
fi
plugin_sha256="$(sha256sum "$DEST" | cut -d ' ' -f1)"
revision_text="git=$revision\ndirty=$dirty\nsha256=$plugin_sha256\n"
if [[ ! -f "$REVISION_FILE" ]] || [[ "$(<"$REVISION_FILE")" != "$(printf '%b' "$revision_text")" ]]; then
    revision_tmp="$(mktemp "$REVISION_FILE.tmp.XXXXXX")"
    TEMP_FILES+=("$revision_tmp")
    printf '%b' "$revision_text" >"$revision_tmp"
    mv -f "$revision_tmp" "$REVISION_FILE"
fi
echo "Revision: $REVISION_FILE"
