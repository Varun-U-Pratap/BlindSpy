"""Record the 1-minute demo video: reports/demo/demo_video.mp4.

Follows reports/demo/video_script.md on the real app: starts `streamlit run app/demo.py` fresh
(each model loads on its first use), drives a headless Chrome over its DevTools protocol (1280 x 720), captures the
page's frames with their real timestamps, draws a cursor, click marks and a caption bar on them,
and encodes them at 10 frames per second with scripts/encode_video.swift (AVFoundation). The
screen of this Mac is not recorded; a title card names the command that starts the demo.

Usage: uv run python scripts/record_demo.py --encoder <compiled encode_video> [--keep <dir>]
(compile once: swiftc -O scripts/encode_video.swift -o <dir>/encode_video)
"""

import argparse
import base64
import io
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from websockets.sync.client import connect

from blindspy.data import ROOT

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / "reports" / "demo" / "demo_video.mp4"
W, H, FPS, PORT, DEMO_PORT = 1280, 720, 10, 9334, 8504
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ROW_H, HEAD_H = 35, 35  # Streamlit's data-grid row and header height, CSS pixels


class Page:
    """A headless Chrome tab; every frame is kept with its time and the caption shown then."""

    def __init__(self, ws):
        self.ws, self.n, self.frames = ws, 0, []
        self.cursor, self.clicks, self.caption = (W * 0.6, H * 0.5), [], ""

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr):
        try:
            r = self.call("Runtime.evaluate", expression=expr, returnByValue=True)
        except RuntimeError:  # the page is between documents while it navigates
            return None
        return r.get("result", {}).get("value")

    def snap(self):
        # PNG bytes, decoded only when rendering (decoded frames would need about 1.6 GB)
        png = base64.b64decode(self.call("Page.captureScreenshot", format="png")["data"])
        self.frames.append((time.monotonic(), png, self.cursor, list(self.clicks), self.caption))

    def hold(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.snap()

    def wait_for(self, cond, timeout=60):
        end = time.monotonic() + timeout
        while not self.js(cond):
            if time.monotonic() > end:
                raise TimeoutError(cond)
            self.snap()
        self.hold(0.6)  # let the rerun settle on screen

    def rect(self, expr):
        r = self.js(f"(() => {{ const e = {expr}; if (!e) return null; "
                    "const b = e.getBoundingClientRect(); "
                    "return [b.left, b.top, b.width, b.height]; })()")  # fmt: skip
        if r is None:
            raise LookupError(expr)
        return r

    def click(self, x, y):
        x0, y0 = self.cursor
        start = time.monotonic()
        while (t := (time.monotonic() - start) / 0.6) < 1:  # glide the cursor there
            e = t * t * (3 - 2 * t)
            self.cursor = (x0 + (x - x0) * e, y0 + (y - y0) * e)
            self.snap()
        self.cursor = (x, y)
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)
        self.clicks.append((time.monotonic(), x, y))

    def click_el(self, expr):
        x, y, w, h = self.rect(expr)
        self.click(x + w / 2, y + h / 2)

    def choose(self, box: int, option: str):
        """Open the sidebar's box-th select box and pick the option with this text."""
        # Streamlit 1.66 select boxes: a react-aria ComboBox; clicking its group opens the list
        self.click_el("document.querySelectorAll('[data-testid=stSidebar] "
                      "[data-testid=stSelectbox] .react-aria-ComboBox [role=group]')"
                      f"[{box}]")  # fmt: skip
        self.hold(0.8)
        self.click_el(f"[...document.querySelectorAll('[role=option]')].find(e => "
                      f"e.textContent.trim() === {json.dumps(option)})")  # fmt: skip

    def click_row(self, row: int):
        self.scroll_to("document.querySelector('[data-testid=stDataFrame]')", "center")
        x, y, _, _ = self.rect("document.querySelector('[data-testid=stDataFrame]')")
        self.click(x + 18, y + HEAD_H + ROW_H * (row - 1) + ROW_H / 2)

    def show_chart(self, heading: str):
        """Wait until the SHAP chart under the heading is loaded (an image below it), then
        scroll the heading to the top; scrolling earlier leaves the chart below the view."""
        h = ("[...document.querySelectorAll('h3')]"
             f".find(e => e.innerText.startsWith({json.dumps(heading)}))")  # fmt: skip
        self.wait_for(f"(() => {{ const h = {h}; if (!h) return false; const top = "
                      "h.getBoundingClientRect().top; const im = [...document.querySelectorAll("
                      "'[data-testid=stMain] img')].find(i => i.getBoundingClientRect().top > top);"
                      " return !!im && im.complete && im.naturalHeight > 0; })()")  # fmt: skip
        self.scroll_to(h)

    def scroll_to(self, expr, block="start"):
        self.js(f"{expr}.scrollIntoView({{behavior: 'smooth', block: '{block}'}})")
        self.hold(1.2)


def caption_bar(draw, text, font):
    if text:
        draw.rectangle([0, H - 44, W, H], fill=(20, 24, 32, 215))
        draw.text((24, H - 34), text, font=font, fill=(255, 255, 255, 255))


def card(lines, fonts):
    img = Image.new("RGB", (W, H), (248, 249, 251))
    d = ImageDraw.Draw(img)
    y = H / 2 - 30 * len(lines)
    for text, f in zip(lines, fonts, strict=True):
        tw = d.textlength(text, font=f)
        d.text(((W - tw) / 2, y), text, font=f, fill=(30, 34, 44))
        y += f.size + 22
    return img


def render(page: Page, frames_dir, intro, outro):
    """Write the frames to disk one by one: intro card, one frame every 1/FPS s of the
    recording (the newest capture at that time, with cursor, click rings and caption drawn on),
    outro card. Returns the number of frames."""
    font = ImageFont.truetype(FONT, 21)
    arrow = [(0, 0), (0, 22), (6, 17), (10, 26), (14, 24), (10, 15), (17, 15)]
    n = 0

    def save(img):
        nonlocal n
        img.save(frames_dir / f"f{n:05d}.png")
        n += 1

    for _ in range(int(3.5 * FPS)):
        save(intro)
    t0, t_end = page.frames[0][0], page.frames[-1][0]
    i, t, decoded = 0, t0, (None, None)
    while t <= t_end:
        while i + 1 < len(page.frames) and page.frames[i + 1][0] <= t:
            i += 1
        _, png, (cx, cy), clicks, cap = page.frames[i]
        if decoded[0] != i:
            decoded = (i, Image.open(io.BytesIO(png)).convert("RGB"))
        img = decoded[1].copy()
        d = ImageDraw.Draw(img, "RGBA")
        for tc, x, y in clicks:
            age = t - tc
            if 0 <= age < 0.7:  # a ring that grows and fades after each click
                r = 10 + 30 * age / 0.7
                alpha = int(255 * (1 - age / 0.7))
                d.ellipse([x - r, y - r, x + r, y + r], outline=(230, 60, 60, alpha), width=4)
        d.polygon([(cx + px, cy + py) for px, py in arrow], fill=(20, 20, 20, 255),
                  outline=(255, 255, 255, 255))  # fmt: skip
        caption_bar(d, cap, font)
        save(img)
        t += 1 / FPS
    for _ in range(int(3 * FPS)):
        save(outro)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="compiled scripts/encode_video.swift")
    ap.add_argument("--keep", help="also keep the rendered frames in this folder")
    a = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    demo = subprocess.Popen(["uv", "run", "streamlit", "run", "app/demo.py", "--server.port",
                             str(DEMO_PORT)], cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)  # fmt: skip
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                               f"--user-data-dir={tmp.name}/profile", "--hide-scrollbars",
                               f"--window-size={W},{H}", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    try:
        for _ in range(100):  # the demo server is up (a model loads when its task is first shown)
            try:
                urllib.request.urlopen(f"http://localhost:{DEMO_PORT}/_stcore/health", timeout=1)
                break
            except OSError:
                time.sleep(0.3)
        else:
            raise SystemExit(f"the demo did not start on port {DEMO_PORT}")
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                tab = next(t for t in tabs if t["type"] == "page")
                break
            except (OSError, StopIteration):
                time.sleep(0.2)
        else:
            raise SystemExit("headless Chrome did not open a page")
        with connect(tab["webSocketDebuggerUrl"], max_size=2**27) as ws:
            p = Page(ws)
            p.call("Emulation.setEmulatedMedia",
                   features=[{"name": "prefers-color-scheme", "value": "light"}])  # fmt: skip
            p.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1,
                   mobile=False)  # fmt: skip
            p.call("Page.enable")
            p.caption = "A fresh start: the demo loads the A1 model (each model loads on first use)"
            p.call("Page.navigate", url=f"http://localhost:{DEMO_PORT}")
            p.wait_for("!!document.querySelector('[data-testid=stDataFrame]')")
            p.caption = "Task A1, VPN or not: 20 test flows from app/samples/, 17 of 20 correct"
            p.hold(4)
            p.caption = "Click a row: row 10, a VPN flow predicted as Non-VPN (confidence 0.63)"
            p.click_row(10)
            p.show_chart("Row 10: predicted")
            p.caption = "Its explanation: a two-packet flow, most gaps -1 (not measurable)"
            p.hold(5)
            p.scroll_to("document.querySelector('h1')")
            p.caption = "Task B: VPN or not and the traffic type together, 14 classes"
            p.choose(0, "B: VPN or not and traffic type together (14 classes)")
            p.wait_for("document.body.innerText.includes('B_test_sample.csv:')")
            p.hold(2.5)
            p.caption = "Click row 7: a VPN-CHAT flow predicted as VPN-FT (confidence 0.73)"
            p.click_row(7)
            p.show_chart("Row 7: predicted VPN-FT")
            p.hold(4.5)
            p.scroll_to("document.querySelector('h1')")
            p.caption = "A malformed file (one column renamed) is rejected with a clear message"
            p.choose(1, "malformed_renamed_column.csv")
            p.wait_for("document.body.innerText.includes('was rejected')")
            p.hold(4)
            p.caption = "Task A2-VPN: the 7-class model for VPN traffic, on its own sample file"
            p.choose(0, "A2-VPN: traffic type of VPN flows (7 classes)")
            p.wait_for("document.body.innerText.includes('A2-VPN_test_sample.csv:')")
            p.hold(4)
        big, mid, small = (
            ImageFont.truetype(f, s) for f, s in ((BOLD, 56), (FONT, 26), (FONT, 20))
        )
        intro = card(["BlindSpy demo",
                       "Classifying encrypted traffic from 23 flow timing features",
                       "started with:  uv run streamlit run app/demo.py",
                       f"recorded {date.today().isoformat()} from commit {commit}"],
                     [big, mid, mid, small])  # fmt: skip
        outro = card(["BlindSpy", "github.com/Varun-U-Pratap/BlindSpy"], [big, mid])
        frames_dir = Path(tmp.name) / "frames"
        frames_dir.mkdir()
        n = render(p, frames_dir, intro, outro)
        print(f"{n} frames ({n / FPS:.1f} s) from {len(p.frames)} captures")
        subprocess.run([a.encoder, "encode", str(frames_dir), str(FPS), str(OUT)], check=True)
        if a.keep:
            shutil.copytree(frames_dir, a.keep, dirs_exist_ok=True)
    finally:
        chrome.terminate()
        chrome.wait(timeout=20)
        demo.terminate()
        demo.wait(timeout=20)
        tmp.cleanup()


if __name__ == "__main__":
    main()
