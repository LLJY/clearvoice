#!/usr/bin/env bash
set -euo pipefail

PIPEWIRE_VERSION=1.6.8
PIPEWIRE_COMMIT=b741e0c74f5436f0c925f7741140db0efd32cf4e
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/clearvoice/pipewire-$PIPEWIRE_VERSION-legacy"
SOURCE_DIR="$CACHE_DIR/source"
BUILD_DIR="$CACHE_DIR/build"
DEST_DIR="$HOME/.local/lib/clearvoice/spa-0.2/aec"
DEST="$DEST_DIR/libspa-aec-webrtc.so"

for cmd in git meson ninja pkg-config python3 ldd readelf pipewire; do
    command -v "$cmd" &>/dev/null || {
        echo "Missing build command: $cmd" >&2
        exit 1
    }
done

if [[ "$(pkg-config --modversion webrtc-audio-processing 2>/dev/null || true)" != 0.3.* ]]; then
    echo "Install the legacy library first:" >&2
    echo "  sudo pacman -S --needed webrtc-audio-processing-0.3" >&2
    exit 1
fi

if ! pipewire --version | grep -q " $PIPEWIRE_VERSION$"; then
    echo "This build is pinned to PipeWire $PIPEWIRE_VERSION." >&2
    exit 1
fi

mkdir -p "$CACHE_DIR"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
        https://github.com/PipeWire/pipewire.git "$SOURCE_DIR"
    git -C "$SOURCE_DIR" checkout --detach "$PIPEWIRE_COMMIT"
fi

if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$PIPEWIRE_COMMIT" ]]; then
    echo "Unexpected PipeWire source revision in $SOURCE_DIR" >&2
    exit 1
fi

python3 - "$SOURCE_DIR/meson.build" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
replacement = """else
  # ClearVoice private build: require the legacy backend with beamforming.
  webrtc_dep = dependency('webrtc-audio-processing',
    version : ['>= 0.2', '< 1.0'],
    required : get_option('echo-cancel-webrtc'))
  cdata.set('HAVE_WEBRTC', webrtc_dep.found())
  summary({'WebRTC Echo Canceling < 1.0': webrtc_dep.found()}, bool_yn: true, section: 'Misc dependencies')
endif

# On FreeBSD"""
pattern = re.compile(
    r"else\n  webrtc_dep = dependency\('webrtc-audio-processing-2',.*?\nendif\n\n# On FreeBSD",
    re.DOTALL,
)
if "ClearVoice private build" not in text:
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit("Could not patch PipeWire WebRTC dependency selection")
    path.write_text(text)
PY

meson_args=(
    --buildtype=release
    -Ddocs=disabled
    -Dman=disabled
    -Dtests=disabled
    -Dexamples=disabled
    -Dsession-managers=[]
    -Decho-cancel-webrtc=enabled
)
if [[ -f "$BUILD_DIR/meson-private/coredata.dat" ]]; then
    meson setup --reconfigure "$BUILD_DIR" "$SOURCE_DIR" "${meson_args[@]}"
else
    meson setup "$BUILD_DIR" "$SOURCE_DIR" "${meson_args[@]}"
fi
ninja -C "$BUILD_DIR" spa/plugins/aec/libspa-aec-webrtc.so

plugin="$BUILD_DIR/spa/plugins/aec/libspa-aec-webrtc.so"
grep -q '^#define HAVE_WEBRTC$' "$BUILD_DIR/config.h"
! grep -q '^#define HAVE_WEBRTC[12]$' "$BUILD_DIR/config.h"
readelf -d "$plugin" | grep -q 'libwebrtc_audio_processing.so.1'
! readelf -d "$plugin" | grep -q 'libwebrtc-audio-processing-[12]'
ldd "$plugin" | grep -q 'libwebrtc_audio_processing.so.1 =>'

mkdir -p "$DEST_DIR"
candidate="$DEST.new.$$"
trap 'rm -f "$candidate"' EXIT
install -m755 "$plugin" "$candidate"
mv -f "$candidate" "$DEST"
trap - EXIT

echo "Installed private beamformer: $DEST"
sha256sum "$DEST"
