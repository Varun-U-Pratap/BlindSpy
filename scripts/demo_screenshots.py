"""Screenshots for the report: reports/demo/demo_table.png (the A1 sample's predictions),
reports/demo/demo_explain.png (row 10 clicked, its SHAP chart) and, with --mlflow-url, two of the
MLflow UI on the official store: reports/demo/mlflow_runs.png (the run list) and
reports/demo/mlflow_run_E12-final-B.png (one final run: metrics and tags).

Headless Chrome is driven over its DevTools protocol (light theme, 1400 x 900 CSS pixels at 2x),
because Streamlit draws the page only after its websocket connects, which a plain
`chrome --screenshot` does not wait for.

Usage: start the demo (uv run streamlit run app/demo.py --server.port 8501) and, for the MLflow
shots, the UI (uv run mlflow ui --backend-store-uri sqlite:///mlruns/utsav-s-r/mlflow.db), then
uv run python scripts/demo_screenshots.py [--url http://localhost:8501]
    [--mlflow-url http://localhost:5000]
"""

import argparse
import base64
import json
import sqlite3
import subprocess
import tempfile
import time
import urllib.request

from websockets.sync.client import connect

from blindspy.data import ROOT

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / "reports" / "demo"
W, H, PORT = 1400, 900, 9333
ROW_10 = (397, 700)  # checkbox of row 10 in the predictions table, CSS pixels
CLOSE_SIDE_PANEL = (1563, 35)  # MLflow 3.17 opens a side panel on each new session; closed


def shot(url: str, out, clicks=(), scroll: int = 0, w: int = W, h: int = H, profile=None) -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        profile = profile or tmp
        chrome = subprocess.Popen(
            [CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
             f"--user-data-dir={profile}", "--hide-scrollbars", f"--window-size={w},{h}",
             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        try:
            for _ in range(50):
                try:
                    tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                    break
                except OSError:
                    time.sleep(0.2)
            page = next(t for t in tabs if t["type"] == "page")
            with connect(page["webSocketDebuggerUrl"], max_size=2**27) as ws:
                n = 0

                def call(method, **params):
                    nonlocal n
                    n += 1
                    ws.send(json.dumps({"id": n, "method": method, "params": params}))
                    while True:
                        msg = json.loads(ws.recv())
                        if msg.get("id") == n:
                            if "error" in msg:
                                raise RuntimeError(f"{method}: {msg['error']}")
                            return msg.get("result", {})

                call("Emulation.setEmulatedMedia",
                     features=[{"name": "prefers-color-scheme", "value": "light"}])  # fmt: skip
                call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                     deviceScaleFactor=2, mobile=False)  # fmt: skip
                call("Page.enable")
                call("Page.navigate", url=url)
                time.sleep(14)  # models load and the table renders
                for x, y in clicks:
                    for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
                        call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left",
                             clickCount=1)  # fmt: skip
                    time.sleep(8)
                if scroll:
                    call("Runtime.evaluate", expression="document.querySelector("
                         f"'[data-testid=stMain]')?.scrollBy(0, {scroll})")  # fmt: skip
                    time.sleep(2)
                png = call("Page.captureScreenshot", format="png")["data"]
                out.write_bytes(base64.b64decode(png))
                print(f"Wrote {out.relative_to(ROOT)}")
        finally:
            chrome.terminate()
            chrome.wait(timeout=20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--mlflow-url")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    shot(a.url, OUT / "demo_table.png")
    shot(a.url, OUT / "demo_explain.png", clicks=[ROW_10], scroll=520)
    if a.mlflow_url:
        db = ROOT / "mlruns" / "utsav-s-r" / "mlflow.db"
        with sqlite3.connect(db) as con:
            exp, run = con.execute("select experiment_id, run_uuid from runs "
                                   "where name = 'E12-final-B'").fetchone()  # fmt: skip
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
            base = f"{a.mlflow_url}/#/experiments/{exp}/runs"
            shot(base, OUT / "mlflow_runs.png", clicks=[CLOSE_SIDE_PANEL], w=1600, h=1000,
                 profile=profile)  # fmt: skip
            shot(f"{base}/{run}", OUT / "mlflow_run_E12-final-B.png", clicks=[CLOSE_SIDE_PANEL],
                 w=1600, h=1000,
                 profile=profile)  # fmt: skip


if __name__ == "__main__":
    main()
