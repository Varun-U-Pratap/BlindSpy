# Demo video: 1-minute recording script

**The committed video, `reports/demo/demo_video.mp4`** (1 min 12 s, 1280 x 720, H.264, 4.2 MB),
was made by `scripts/record_demo.py` on 31 Oct 2026 from commit 45a1c59. It follows the steps
below on the real app: it starts the demo fresh, drives a headless Chrome through them, captures
the page's own frames and adds a cursor, click marks, captions and title cards; it does not record
this Mac's screen, so the Terminal command is shown on the title card. To make it again (demo not
running, port 8504 free):

```bash
swiftc -O scripts/encode_video.swift -o /tmp/encode_video
```

```bash
uv run python scripts/record_demo.py --encoder /tmp/encode_video
```

The steps below are also the script for recording the real screen by hand.

The backup video of the demo (`app/demo.py`), recorded on the presenting Mac with the built-in
screen recorder. It shows the demo starting from a fresh start, a correct and a wrong prediction
with their explanations, a 14-class task, and the rejection of a malformed file.

## Before recording

1. Close other windows and notifications (Focus: Do Not Disturb).
2. Open Terminal in the repository folder: `cd ~/Developer/BlindSpy`. Make sure no demo is already
   running. Open a browser window with an empty tab next to Terminal (the demo does not open a tab
   by itself).
3. Press **Cmd+Shift+5**. Choose **Record Entire Screen** (or **Record Selected Portion** around
   Terminal and the browser). Under **Options**: **Save to → Other Location…** and pick
   `~/Developer/BlindSpy/backups/` (git-ignored, so the raw file stays out of the repository);
   Microphone **None** unless you narrate; **Show Mouse Clicks** on.

## Recording (about 60 seconds)

| Time | Do this | The viewer sees |
| --- | --- | --- |
| 0:00 | Click **Record**. In Terminal type `uv run streamlit run app/demo.py` and press Return; in the browser tab go to `localhost:8501`. | The demo starts from a fresh start (allow about 5 seconds for the models to load). |
| 0:08 | Wait for the table. Point at "17 of 20 predicted correctly" and the confidence bars. | Task A1 (VPN or not) on the A1 sample file: label, true label, calibrated confidence, top 3 features. |
| 0:15 | Click the checkbox of **row 10**; scroll down to the chart. | Row 10 is a VPN flow predicted as Non-VPN (confidence 0.6288). The chart shows a two-packet flow: most gaps are "-1 (not measurable)". |
| 0:27 | Scroll up. In the sidebar set **Task** to **B: VPN or not and traffic type together (14 classes)**. | The sample switches to `B_test_sample.csv`; 14-class predictions appear. |
| 0:35 | Click the checkbox of **row 7**; scroll down. | A VPN-CHAT flow predicted as VPN-FT (confidence 0.7308), with the features behind it. |
| 0:45 | Scroll up. Set **Sample file** to `malformed_renamed_column.csv`. | A red message: the file is rejected because `flowBytesPerSecond` is missing and `flowBytesPerSec` is unexpected. |
| 0:52 | Set **Task** to **A2-VPN** (its sample loads). | The VPN-only 7-class model on its own sample file. |
| 0:58 | Stop the recording: the stop button in the menu bar (or Cmd+Ctrl+Esc). | |

Then stop the demo in Terminal with Ctrl+C.

## After recording

For a hand-made recording: shrink it to 720p with HandBrakeCLI (well under GitHub's 50 MB warning)
before committing; the raw recording stays only in `backups/`.
