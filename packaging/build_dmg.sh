#!/usr/bin/env bash
# Build MLX Creator.app + MLX-Creator.dmg from the sources in this folder.
# Reproducible from a fresh clone (no ad-hoc steps). Output: packaging/MLX-Creator.dmg
set -e
cd "$(dirname "$0")"
APP="build/MLX Creator.app"

echo "→ assembling app bundle"
rm -rf build && mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp Info.plist "$APP/Contents/Info.plist"
clang -framework Cocoa -arch arm64 -O2 -o "$APP/Contents/MacOS/MLX Creator" launcher.m
cp launcher.sh "$APP/Contents/MacOS/launch.sh"; chmod +x "$APP/Contents/MacOS/launch.sh"
cp firstrun.sh "$APP/Contents/Resources/firstrun.sh"; chmod +x "$APP/Contents/Resources/firstrun.sh"
cp icon.icns "$APP/Contents/Resources/icon.icns"

echo "→ bundling app code (no weights / venv / outputs)"
# NOTE: leading-slash excludes are anchored to the transfer root, so they only
# drop the repo-root data dirs — NOT vendored code dirs named the same
# (e.g. vendor/mflux/src/mflux/models/).
rsync -a --exclude '/models/' --exclude '/.venv/' --exclude '/outputs/' --exclude '/.git/' \
  --exclude '/packaging/' --exclude '__pycache__/' --exclude '*.pyc' \
  ../ "$APP/Contents/Resources/app/"

echo "→ ad-hoc signing (unsigned distribution; see FIRST-LAUNCH.txt)"
codesign --force --deep -s - "$APP" >/dev/null 2>&1 || true

echo "→ building DMG (app + Applications alias + first-launch guide)"
STAGE=dmg_stage; rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
cp FIRST-LAUNCH.txt "$STAGE/FIRST-LAUNCH.txt"
ln -s /Applications "$STAGE/Applications"
rm -f MLX-Creator.dmg
hdiutil create -volname "MLX Creator" -srcfolder "$STAGE" -ov -format UDZO MLX-Creator.dmg >/dev/null
rm -rf "$STAGE"
echo "✓ Built packaging/MLX-Creator.dmg ($(du -h MLX-Creator.dmg | cut -f1))"
