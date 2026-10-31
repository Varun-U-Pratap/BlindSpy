"""Assemble the report drafts into one report: reports/report/BlindSpy_report_<tag>.pdf and .docx.

The markdown drafts in reports/report/drafts/ are the master copy. This script reads them in
order, drops each draft's "Working draft" note, and writes
- a PDF: markdown -> HTML (this script) -> PDF with headless Google Chrome;
- a DOCX: markdown -> a list of blocks (JSON) -> scripts/report_docx.js (the `docx` npm package).

Only the markdown the drafts use is supported: headings (#, ##, ###), paragraphs, bullet and
numbered lists, tables, PNG images and inline **bold**, *italic* and `code`.

After the numbered drafts comes the reference list (reports/references.md), then the appendix
(drafts/appendix.md), whose lines `{{include <file>}}` insert a whole repository file (its title
line dropped, its headings one level lower) and `{{git tags}}` a table of the git tags.

Usage: uv run python scripts/build_report.py --tag v0.2-tuned --node-modules <dir with docx>
(--tag final writes the hand-in report, BlindSpy_report.pdf and .docx.)
(`npm install --prefix <dir> docx` once; the folder is not part of the repository.)
"""

import argparse
import html
import json
import os
import re
import struct
import subprocess
import tempfile
from datetime import date
from pathlib import Path

from blindspy.data import ROOT

DRAFTS = ROOT / "reports" / "report" / "drafts"
OUT = ROOT / "reports" / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TITLE = "BlindSpy"
SUBTITLE = "Classifying encrypted traffic (ISCXVPN2016) from 23 flow timing features"


def inline(text: str) -> list[dict]:
    """Runs of plain, **bold**, *italic* and `code` text (code first, so its contents stay
    literal)."""
    runs = []
    for part in re.split(r"(`[^`]+`)", text):
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            runs.append({"text": part[1:-1], "code": True})
            continue
        for bpart in re.split(r"(\*\*[^*]+\*\*)", part):
            if bpart.startswith("**") and bpart.endswith("**") and len(bpart) > 4:
                runs.append({"text": bpart[2:-2], "b": True})
                continue
            for ipart in re.split(r"(?<![\w*])(\*[^*\s][^*]*\*)(?![\w*])", bpart):
                if ipart.startswith("*") and ipart.endswith("*") and len(ipart) > 2:
                    runs.append({"text": ipart[1:-1], "i": True})
                elif ipart:
                    runs.append({"text": ipart})
    return runs


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def parse(text: str, base: Path) -> list[dict]:
    """Markdown -> blocks."""
    blocks, para, lines = [], [], text.splitlines()

    def flush():
        text = " ".join(para)
        if para and not text.startswith("*Working draft"):  # each draft's own note
            blocks.append({"t": "p", "runs": inline(text)})
        para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
        elif m := re.match(r"^(#{1,3}) (.+)$", line):
            flush()
            blocks.append({"t": "h", "level": len(m.group(1)), "runs": inline(m.group(2))})
        elif m := re.match(r"^!\[(.*)\]\((.+\.png)\)$", line.strip()):
            flush()
            path = (base / m.group(2)).resolve()
            w, h = png_size(path)
            blocks.append({"t": "img", "alt": m.group(1), "path": str(path), "w": w, "h": h})
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append({"t": "table", "rows": [[inline(c) for c in r] for r in rows],
                           "plain": rows})  # fmt: skip
            continue
        elif m := re.match(r"^(- |\d+\. )(.*)$", line):
            flush()
            kind = "ul" if m.group(1) == "- " else "ol"
            items = []
            while i < len(lines) and (m := re.match(r"^(- |\d+\. )(.*)$", lines[i])):
                item = [m.group(2)]
                i += 1
                while i < len(lines) and re.match(r"^ {2,}\S", lines[i]):
                    item.append(lines[i].strip())
                    i += 1
                items.append(inline(" ".join(item)))
            blocks.append({"t": kind, "items": items})
            continue
        else:
            para.append(line.strip())
        i += 1
    flush()
    return blocks


def include(path: Path) -> str:
    """A repository file for the appendix: title line dropped, headings one level lower (### at
    most, the deepest level the report has)."""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]

    def shift(m: re.Match) -> str:
        return "#" * min(3, len(m.group(1)) + 1) + " "

    return "\n".join(re.sub(r"^(#{1,6}) ", shift, x) for x in lines)


def git_tags() -> str:
    fmt = "%(refname:short)|%(*objectname:short)|%(taggerdate:short)|%(contents:subject)"
    out = subprocess.run(["git", "for-each-ref", "refs/tags", "--sort=taggerdate",
                          f"--format={fmt}"], cwd=ROOT, check=True, capture_output=True,
                         text=True).stdout.splitlines()  # fmt: skip
    rows = ["| tag | commit | date | message |", "|---|---|---|---|"]
    for line in out:
        tag, commit, day, subject = line.split("|", 3)
        rows.append(f"| {tag} | {commit} | {day} | {subject.split(': ', 1)[-1]} |")
    return "\n".join(rows)


def appendix(path: Path) -> str:
    text = []
    for line in path.read_text().splitlines():
        if m := re.fullmatch(r"\{\{include (.+)\}\}", line.strip()):
            text.append(include(ROOT / m.group(1)))
        elif line.strip() == "{{git tags}}":
            text.append(git_tags())
        else:
            text.append(line)
    return "\n".join(text)


def load_drafts() -> list[dict]:
    blocks = []
    for path in sorted(DRAFTS.glob("[0-9][0-9]_*.md")):
        part = parse(path.read_text(), path.parent)
        if blocks:
            blocks.append({"t": "pagebreak"})
        blocks += part
    # the reference list, kept in one file from the start and copied into the report
    refs = (ROOT / "reports" / "references.md").read_text().split("\n", 1)[1]
    blocks += [{"t": "pagebreak"}, *parse("# References\n" + refs, ROOT / "reports")]
    if (DRAFTS / "appendix.md").exists():
        blocks += [{"t": "pagebreak"}, *parse(appendix(DRAFTS / "appendix.md"), DRAFTS)]
    for b in blocks:
        if b["t"] == "table":
            b["font_pt"], b["anywhere"] = table_font(b["plain"])
    for n, b in enumerate((b for b in blocks if b["t"] == "img"), start=1):
        b["caption"] = f"Figure {n}. {b['alt']}"
    return blocks


def html_runs(runs: list[dict]) -> str:
    out = []
    for r in runs:
        t = html.escape(r["text"])
        if r.get("code"):
            t = f"<code>{t}</code>"
        if r.get("b"):
            t = f"<strong>{t}</strong>"
        if r.get("i"):
            t = f"<em>{t}</em>"
        out.append(t)
    return "".join(out)


CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10pt; line-height: 1.4;
       color: #111; }
h1 { font-size: 18pt; margin: 0 0 8pt; } h2 { font-size: 13pt; margin: 14pt 0 4pt; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; }
h2, h3 { break-after: avoid; } p { margin: 0 0 6pt; } li { margin-bottom: 2pt; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.5pt; background: #f2f2f2;
       padding: 0 2px; }
table { border-collapse: collapse; margin: 4pt 0 10pt; width: 100%; break-inside: auto; }
th, td { border: 0.5pt solid #999; padding: 2pt 3pt; text-align: left; vertical-align: top;
         overflow-wrap: break-word; }
th { background: #e8eef5; } tr { break-inside: avoid; }
table.anywhere td, table.anywhere th { overflow-wrap: anywhere; }
figure { margin: 6pt 0 10pt; break-inside: avoid; }
img { max-width: 100%; display: block; margin: 0 auto; }
figcaption { text-align: center; font-size: 8.5pt; font-style: italic; margin-top: 3pt; }
.title { text-align: center; padding-top: 80mm; }
.title h1 { font-size: 30pt; } .title .sub { font-size: 14pt; margin-bottom: 30pt; }
.pb { break-after: page; }
"""


TEXT_WIDTH_PT = 505  # A4 minus the side margins
MIN_FONT_PT = 5.5


def table_font(plain: list[list[str]]) -> tuple[float, bool]:
    """Font size (pt) by the table's widest row, made smaller if the longest word of every
    column would not fit side by side; (MIN_FONT_PT, True) = break inside words as well."""
    width = max(sum(len(c) for c in r) for r in plain)
    size = 6.5 if width > 140 else 8 if width > 90 else 9
    words = [max((len(w) for r in plain for w in (r[j] if j < len(r) else "").split()), default=1)
             for j in range(len(plain[0]))]  # fmt: skip
    need = (sum(words) + 1.2 * len(words)) * 0.55  # pt per point of font size, incl. padding
    fit = TEXT_WIDTH_PT / need
    return (size, False) if fit >= size else (max(fit, MIN_FONT_PT), fit < MIN_FONT_PT)


def to_html(blocks: list[dict], front: list[str]) -> str:
    sub = f'<p class="sub">{SUBTITLE}</p>'
    lines = [f"<p>{html.escape(x)}</p>" for x in front]
    body = ['<div class="title">', f"<h1>{TITLE}</h1>", sub, *lines, '</div><div class="pb"></div>']
    for b in blocks:
        t = b["t"]
        if t == "h":
            body.append(f"<h{b['level']}>{html_runs(b['runs'])}</h{b['level']}>")
        elif t == "p":
            body.append(f"<p>{html_runs(b['runs'])}</p>")
        elif t in ("ul", "ol"):
            items = "".join(f"<li>{html_runs(r)}</li>" for r in b["items"])
            body.append(f"<{t}>{items}</{t}>")
        elif t == "img":
            alt, cap = html.escape(b["alt"]), html.escape(b["caption"])
            body.append(f'<figure><img src="file://{b["path"]}" alt="{alt}">'
                        f"<figcaption>{cap}</figcaption></figure>")  # fmt: skip
        elif t == "table":
            head, *rows = b["rows"]
            th = "".join(f"<th>{html_runs(c)}</th>" for c in head)
            trs = "".join("<tr>" + "".join(f"<td>{html_runs(c)}</td>" for c in r) + "</tr>"
                          for r in rows)  # fmt: skip
            cls = "anywhere" if b["anywhere"] else ""
            body.append(f'<table class="{cls}" style="font-size:{b["font_pt"]:.2f}pt">'
                        f"<thead><tr>{th}</tr></thead>"
                        f"<tbody>{trs}</tbody></table>")  # fmt: skip
        elif t == "pagebreak":
            body.append('<div class="pb"></div>')
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{TITLE}</title>'
            f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--node-modules", required=True, help="folder that holds node_modules/docx")
    a = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    milestone = "Final report (hand-in)" if a.tag == "final" else f"Report at milestone {a.tag}"
    front = [milestone, f"Built {date.today().isoformat()} from the "
             f"report drafts at commit {commit}"]  # fmt: skip
    blocks = load_drafts()
    name = "BlindSpy_report" if a.tag == "final" else f"BlindSpy_report_{a.tag}"
    pdf, docx = OUT / f"{name}.pdf", OUT / f"{name}.docx"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(to_html(blocks, front))
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={pdf}", str(page)],
                       check=True, capture_output=True)  # fmt: skip
        spec = Path(tmp) / "blocks.json"
        spec.write_text(json.dumps({"title": TITLE, "subtitle": SUBTITLE, "front": front,
                                    "blocks": blocks}))  # fmt: skip
        env = {**os.environ, "NODE_PATH": str(Path(a.node_modules) / "node_modules")}
        subprocess.run(["node", str(ROOT / "scripts" / "report_docx.js"), str(spec), str(docx)],
                       check=True, env=env)  # fmt: skip
    print("wrote", pdf.relative_to(ROOT), "and", docx.relative_to(ROOT))


if __name__ == "__main__":
    main()
