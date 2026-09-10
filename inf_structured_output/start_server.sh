#!/bin/sh
set -eu
module_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# Check the fixed endpoint used by the notebook before starting another process.
if "$module_dir/.venv/bin/python" - <<'PY'
import json
import socket
import sys
from urllib.request import urlopen

try:
    with urlopen('http://127.0.0.1:8080/health', timeout=2) as response:
        healthy = json.load(response).get('status') == 'ok'
    with urlopen('http://127.0.0.1:8080/v1/models', timeout=2) as response:
        models = json.load(response).get('data', [])
    if healthy and any(model.get('id') == 'local-lfm' for model in models):
        print('local-lfm is already running at http://127.0.0.1:8080.')
        print('Open main.ipynb and Run All; no second server is needed.')
        sys.exit(0)
except (OSError, ValueError):
    pass

try:
    with socket.create_connection(('127.0.0.1', 8080), timeout=2):
        pass
except ConnectionRefusedError:
    sys.exit(10)  # No listener: start our server.
except OSError as error:
    print(f'Cannot check port 8080: {error}', file=sys.stderr)
    sys.exit(1)
print('Port 8080 is occupied, but local-lfm is not ready or is not the serving model. '
      'Wait if it is loading; otherwise inspect the process using port 8080.', file=sys.stderr)
sys.exit(1)
PY
then
    exit 0
else
    result=$?
    if [ "$result" -ne 10 ]; then
        exit "$result"
    fi
fi

exec "$module_dir/.runtime/llama-b10883/llama-server" \
  -m "$module_dir/.runtime/LFM2.5-1.2B-Instruct-Q4_K_M.gguf" \
  --alias local-lfm -c 4096 --host 127.0.0.1 --port 8080 -t 4
