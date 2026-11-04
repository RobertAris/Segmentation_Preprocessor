#!/bin/zsh
# Double-click in Finder, or pass a prepared run directory from Terminal.
set -euo pipefail

workspace_dir="${0:A:h}"
slicer_executable=""
for app_path in /Applications/*Slicer*.app(Nom) "$HOME"/Applications/*Slicer*.app(Nom); do
  if [[ -x "$app_path/Contents/MacOS/Slicer" ]]; then
    slicer_executable="$app_path/Contents/MacOS/Slicer"
    break
  fi
done
if [[ -z "$slicer_executable" ]]; then
  print -u2 '3D Slicer was not found in /Applications or ~/Applications.'
  exit 1
fi

cd "$workspace_dir"
exec env DSA_WORKFLOW_ROOT="$workspace_dir" DSA_PREPARED_DIR="${1:-}" \
  "$slicer_executable" --no-splash --python-script "$workspace_dir/slicer/LaunchDSA.py"
