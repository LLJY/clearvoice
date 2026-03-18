#!/usr/bin/env bash
set -euo pipefail

APP_ID="clearvoice"
APP_NAME="ClearVoice"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.local/share/$APP_ID"
DESKTOP_DIR="$HOME/.config/autostart"

# ── Dependency check ──────────────────────────────────────────────────────────
echo "Checking dependencies..."
ok=true

for cmd in pipewire pw-cli pw-dump wpctl pactl; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "  MISSING: $cmd"
        ok=false
    else
        echo "  OK: $cmd"
    fi
done

# LADSPA plugin
ladspa_found=false
for dir in /usr/lib/ladspa /usr/lib64/ladspa /usr/local/lib/ladspa "$HOME/.ladspa"; do
    if [ -f "$dir/libdeep_filter_ladspa.so" ]; then
        echo "  OK: DeepFilterNet LADSPA ($dir/libdeep_filter_ladspa.so)"
        ladspa_found=true
        break
    fi
done
if [ "$ladspa_found" = false ]; then
    echo "  MISSING: libdeep_filter_ladspa.so"
    echo "    Install: paru -S libdeep_filter_ladspa-bin"
    ok=false
fi

# Python + PyGObject
if python3 -c "import gi; gi.require_version('Gtk','3.0'); from gi.repository import Gtk" 2>/dev/null; then
    echo "  OK: PyGObject (GTK3)"
else
    echo "  MISSING: PyGObject / python3-gi"
    ok=false
fi

# AppIndicator
if python3 -c "
import gi
try:
    gi.require_version('AyatanaAppIndicator3','0.1')
    from gi.repository import AyatanaAppIndicator3
except:
    gi.require_version('AppIndicator3','0.1')
    from gi.repository import AppIndicator3
" 2>/dev/null; then
    echo "  OK: AppIndicator"
else
    echo "  WARNING: No AppIndicator — will fall back to Gtk.StatusIcon"
fi

if [ "$ok" = false ]; then
    echo ""
    echo "Some dependencies are missing. Fix them and re-run."
    exit 1
fi

# ── Install ───────────────────────────────────────────────────────────────────
echo ""
echo "Installing $APP_NAME..."

mkdir -p "$INSTALL_DIR"
cp "$SCRIPT_DIR/clearvoice.py" "$INSTALL_DIR/clearvoice.py"
chmod +x "$INSTALL_DIR/clearvoice.py"

echo "  Installed to $INSTALL_DIR/clearvoice.py"

# ── Autostart .desktop file ──────────────────────────────────────────────────
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_DIR/$APP_ID.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=PipeWire noise cancellation, beamforming & AEC tray tool
Exec=python3 $INSTALL_DIR/clearvoice.py
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Audio;
X-GNOME-Autostart-enabled=true
X-KDE-autostart-after=panel
DESKTOP

echo "  Autostart entry: $DESKTOP_DIR/$APP_ID.desktop"
echo ""
echo "Done. $APP_NAME will start on next login."
echo "To start now:  python3 $INSTALL_DIR/clearvoice.py &"
