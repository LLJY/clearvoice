# clearvoice-git (Arch / AUR)

Builds ClearVoice from GitHub `master` and installs it system-wide with an XDG
autostart entry, so it starts at every graphical login.

## Install

The stock DeepFilterNet LADSPA plugin comes from the AUR, so install it first,
then build this package from the repo:

```bash
paru -S libdeep_filter_ladspa-bin    # or yay / makepkg
cd packaging/aur
makepkg -si
systemctl --user restart wireplumber pipewire-pulse   # load the ClearVoice policy once
```

To publish to the AUR, copy `PKGBUILD`, `clearvoice.install` and `.SRCINFO` into
the AUR git repo. After editing the PKGBUILD, regenerate `.SRCINFO` with
`makepkg --printsrcinfo > .SRCINFO`.

## What it builds

- the tray app (`/usr/bin/clearvoice`, `/usr/share/clearvoice/`)
- the private legacy-WebRTC beamformer for the installed PipeWire 1.6.x series
  (`/usr/lib/clearvoice/spa-0.2/`)
- the ClearVoice constant-latency LADSPA plugin with its DFN3-LL and FastEnhancer
  models (`/usr/lib/clearvoice/ladspa/`), compiled with `-C target-cpu=native`
- the WirePlumber base-mic lock policy (`/usr/share/wireplumber/`)
- application and autostart entries (`/usr/share/applications/`, `/etc/xdg/autostart/`)

The build needs network access: it clones PipeWire and DeepFilterNet at pinned
commits, downloads the FastEnhancer models (SHA-256 verified) and the Python
wheels used to export the DFN3-LL models. Expect several minutes.

Rebuild the package after PipeWire moves to a new major.minor series; the
beamformer is pinned per series.

## Coexisting with a development install

Plugins built by `./build-beamformer.sh` and `./build-plugin.sh` under
`~/.local/lib/clearvoice/` take precedence over the packaged ones. A previous
`./deploy.sh` install also autostarts from `~/.config/autostart/clearvoice.desktop`;
remove that file (and `~/.local/share/applications/clearvoice.desktop`) to use
the package's entries.
