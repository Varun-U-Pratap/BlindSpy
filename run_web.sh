#!/usr/bin/env bash
# Start the BlindSpy interactive web demo (Streamlit) and open it in the browser.
# Usage: ./run_web.sh [PORT]        (default port 8501; stop with Ctrl+C)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8501}"
URL="http://localhost:${PORT}"

# Run from the repository folder so .streamlit/config.toml applies.
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is not installed (https://docs.astral.sh/uv/)." >&2
  exit 1
fi

echo "Syncing dependencies..."
uv sync --quiet

echo "Starting BlindSpy demo at ${URL}  (Ctrl+C to stop)"

# Open the browser once the server is up (config sets headless = true, so Streamlit won't).
(
  for _ in $(seq 1 30); do
    if curl -s -o /dev/null "${URL}/_stcore/health" 2>/dev/null; then
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL" >/dev/null 2>&1 || true
      elif command -v open >/dev/null 2>&1; then
        open "$URL" >/dev/null 2>&1 || true
      fi
      break
    fi
    sleep 1
  done
) &

exec uv run streamlit run app/demo.py --server.port "$PORT"
