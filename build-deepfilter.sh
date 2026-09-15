#!/usr/bin/env bash
set -euo pipefail

REVISION=a20ca9f6c201b3661842901d085dee7b0a81a45c
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/clearvoice/deepfilter-pr617"
SOURCE_DIR="$CACHE_DIR/source"
DEST_DIR="$HOME/.local/lib/clearvoice/ladspa"
DEST="$DEST_DIR/libdeep_filter_ladspa.so"

for cmd in git cargo rustc python3 ldd objdump; do
    command -v "$cmd" &>/dev/null || {
        echo "Missing build command: $cmd" >&2
        exit 1
    }
done

mkdir -p "$CACHE_DIR"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
        https://github.com/danielhuang/DeepFilterNet.git "$SOURCE_DIR"
    git -C "$SOURCE_DIR" checkout --detach "$REVISION"
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$REVISION" ]]; then
    echo "Unexpected DeepFilter source revision in $SOURCE_DIR" >&2
    exit 1
fi

python3 - "$SOURCE_DIR/ladspa/src/lib.rs" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
for debug_line in (
    '        println!("thread exiting")\n',
    '        dbg!(&channels);\n',
    '                dbg!(&e);\n',
    '                dbg!(self.hop_size);\n',
):
    text = text.replace(debug_line, "")
text = text.replace(
    "            if let Err(e) = self.raw_audio_sender.try_send(frame) {",
    "            if self.raw_audio_sender.try_send(frame).is_err() {",
)
path.write_text(text)
PY

CARGO_PROFILE_RELEASE_LTO=fat \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
RUSTFLAGS='-C target-cpu=native' \
    cargo build --locked --profile=release-lto -p deep-filter-ladspa \
    --manifest-path "$SOURCE_DIR/Cargo.toml"

plugin="$SOURCE_DIR/target/release-lto/libdeep_filter_ladspa.so"
ldd "$plugin" >/dev/null
objdump -d -M intel "$plugin" | grep -E '\bzmm[0-9]+' >/dev/null

mkdir -p "$DEST_DIR"
candidate="$DEST.new.$$"
revision_candidate="$DEST_DIR/libdeep_filter_ladspa.revision.new.$$"
trap 'rm -f "$candidate" "$revision_candidate"' EXIT
install -m755 "$plugin" "$candidate"
printf '%s\n' "$REVISION" >"$revision_candidate"
mv -f "$candidate" "$DEST"
mv -f "$revision_candidate" "$DEST_DIR/libdeep_filter_ladspa.revision"
trap - EXIT

echo "Installed private DeepFilter: $DEST"
sha256sum "$DEST"
