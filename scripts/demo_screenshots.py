"""Screenshots of the running demo for the report: reports/demo/demo_table.png (the A1 sample's
predictions) and reports/demo/demo_explain.png (row 10 clicked, its SHAP chart).

Headless Chrome is driven over its DevTools protocol (light theme, 1400 x 900 CSS pixels at 2x),
because Streamlit draws the page only after its websocket connects, which a plain
`chrome --screenshot` does not wait for.

Usage: start the demo (uv run streamlit run app/demo.py --server.port 8501), then
uv run python scripts/demo_screenshots.py [--url http://localhost:8501]
"""

import argparse
import base64
import json
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


def shot(url: str, out, clicks=(), scroll: int = 0) -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        chrome = subprocess.Popen(
            [CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
             f"--user-data-dir={profile}", "--hide-scrollbars", f"--window-size={W},{H}",
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
                call("Emulation.setDeviceMetricsOverride", width=W, height=H,
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
    url = ap.parse_args().url
    OUT.mkdir(exist_ok=True)
    shot(url, OUT / "demo_table.png")
    shot(url, OUT / "demo_explain.png", clicks=[ROW_10], scroll=520)


if __name__ == "__main__":
    main()
