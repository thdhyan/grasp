#!/usr/bin/env bash
# Run from repository root. Makes a backup and injects the compat import header.
set -euo pipefail


TARGET="third_party/gqcnn/gqcnn/model/tf/network_tf.py"
BACKUP="$TARGET.bak.$(date +%s)"


if [ ! -f "$TARGET" ]; then
echo "ERROR: $TARGET not found. Run from repo root or adjust path."
exit 2
fi


cp "$TARGET" "$BACKUP"


echo "Backup saved to $BACKUP"


# We replace the first occurrence of the line importing tensorflow.contrib.framework
python - <<'PY'
from pathlib import Path
p=Path('$TARGET')
s=p.read_text()
old='import tensorflow.contrib.framework as tcf'
if old not in s:
print('Warning: expected import line not found; no changes made.')
raise SystemExit(0)
new='''import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()\n\ntry:\n import tensorflow.contrib.framework as tcf\n import tensorflow.contrib.layers as contrib_layers\nexcept Exception:\n from .compat import tcf, layers as contrib_layers # noqa: F401\n'''
ns=s.replace(old, new, 1)
p.write_text(ns)
print('Patched', '$TARGET')
PY


echo "Done. Next: install dependencies:\n pip install tf_slim"