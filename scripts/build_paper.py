"""Build the research-paper version of the report: reports/paper/BlindSpy_paper.docx.

The source is reports/paper/BlindSpy_paper.md (written from the report drafts and the generated
tables; tests/test_paper.py checks its numbers against them). Its format is the report drafts'
markdown (scripts/build_report.py parses it) plus:
- `% ` lines at the top: title, authors, affiliation;
- `## Abstract`, then the abstract and a `**Index Terms**—...` paragraph;
- `#` section headings, numbered I, II, ...; `##` subsections, lettered A, B, ...;
- `Table: <caption>` right before a table (numbered TABLE I, II, ...); an image's alt text is its
  caption (Fig. 1, 2, ...);
- a `[[wide]]` line before a table or image makes it span both columns.
Citations [n] use the numbers of reports/references.md; the paper renumbers them by first use and
lists only the cited references. The build stops if a "Section X-Y", "Table N" or "Fig. n"
reference has no target.

Usage: uv run python scripts/build_paper.py --node-modules <dir with node_modules/docx>
(`npm install --prefix <dir> docx@9.15.0` once; the folder is not part of the repository.)
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from blindspy.data import ROOT

_spec = importlib.util.spec_from_file_location("build_report", ROOT / "scripts" / "build_report.py")
_build_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build_report)
inline, parse = _build_report.inline, _build_report.parse

SOURCE = ROOT / "reports" / "paper" / "BlindSpy_paper.md"
REFERENCES = ROOT / "reports" / "references.md"
OUT = ROOT / "reports" / "paper" / "BlindSpy_paper.docx"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
CITE = re.compile(r"\[(\d+(?:, \d+)*)\]")


def text_of(runs: list[dict]) -> str:
    return "".join(r["text"] for r in runs)


def references() -> dict[int, str]:
    """reports/references.md -> {number: text}."""
    refs, cur = {}, None
    for line in REFERENCES.read_text().splitlines():
        if m := re.match(r"^(\d+)\. (.*)$", line):
            cur = int(m.group(1))
            refs[cur] = m.group(2).strip()
        elif cur is not None and line.startswith("   ") and line.strip():
            refs[cur] += " " + line.strip()
    return refs


def map_runs(runs: list[dict], fn) -> list[dict]:
    return [{**r, "text": fn(r["text"])} for r in runs]


def build_spec(source: str) -> dict:
    head = [line[2:].strip() for line in source.splitlines() if line.startswith("% ")]
    title, authors, affiliation = head
    body = "\n".join(line for line in source.splitlines() if not line.startswith("% "))
    blocks = parse(body, SOURCE.parent)

    # Abstract and index terms: everything before the first level-1 heading
    first = next(i for i, b in enumerate(blocks) if b["t"] == "h" and b["level"] == 1)
    front = [b for b in blocks[:first] if b["t"] == "p"]
    is_terms = [text_of(b["runs"]).startswith("Index Terms") for b in front]
    abstract = [b["runs"] for b, terms in zip(front, is_terms, strict=True) if not terms]
    index_terms = [text_of(b["runs"]) for b, terms in zip(front, is_terms, strict=True) if terms]
    assert len(abstract) == 1 and len(index_terms) == 1, "one abstract and one Index Terms line"
    blocks = blocks[first:]

    # Number sections, tables and figures; attach captions and wide markers
    out, sec, sub = [], 0, 0
    tables = figures = 0
    caption, wide = None, False
    targets = set()
    for b in blocks:
        t = text_of(b["runs"]) if b["t"] in ("p", "h") else ""
        # the wide marker and a caption can end up in one paragraph
        if b["t"] == "p" and t.startswith("[[wide]] Table: "):
            wide, t = True, t[len("[[wide]] ") :]
        if b["t"] == "p" and t == "[[wide]]":
            wide = True
            continue
        if b["t"] == "p" and t.startswith("Table: "):
            caption = t[len("Table: ") :]
            continue
        if b["t"] == "h" and b["level"] == 1:
            sec, sub = sec + 1, 0
            b = {**b, "label": ROMAN[sec - 1]}
        elif b["t"] == "h" and b["level"] == 2:
            sub += 1
            letter = chr(ord("A") + sub - 1)
            b = {**b, "label": letter}
            targets.add(f"Section {ROMAN[sec - 1]}-{letter}")
        if b["t"] == "table":
            assert caption, "every table needs a `Table:` caption line"
            tables += 1
            b = {**b, "label": f"TABLE {ROMAN[tables - 1]}", "caption": caption, "wide": wide}
            caption, wide = None, False
        elif b["t"] == "img":
            figures += 1
            b = {**b, "label": f"Fig. {figures}.", "caption": b["alt"], "wide": wide}
            wide = False
        out.append(b)
    targets |= {f"Section {ROMAN[i]}" for i in range(sec)}
    targets |= {f"Table {ROMAN[i]}" for i in range(tables)}
    targets |= {f"Fig. {i + 1}" for i in range(figures)}

    # Check every cross-reference in the text
    def all_text(bs):
        for b in bs:
            if b["t"] in ("p", "h"):
                yield text_of(b["runs"])
            elif b["t"] in ("ul", "ol"):
                yield from (text_of(item) for item in b["items"])
            elif b["t"] == "table":
                yield from (text_of(c) for row in b["rows"] for c in row)
            elif b["t"] == "img":
                yield b["caption"]

    texts = [text_of(r) for r in abstract] + list(all_text(out))
    pattern = r"Sections? [IVX]+(?:-[A-Z])?|Table [IVX]+|Fig\. \d+"
    for ref in {m for t in texts for m in re.findall(pattern, t)}:
        ref = ref.replace("Sections", "Section")
        assert ref in targets, f"cross-reference without a target: {ref}"
    assert not [t for t in texts if "[[" in t or "Table:" in t], "unused marker left in the text"

    # Renumber citations by first use
    refs, order = references(), {}
    for t in texts:
        for group in CITE.findall(t):
            for n in group.split(", "):
                order.setdefault(int(n), len(order) + 1)
    missing = [n for n in order if n not in refs]
    assert not missing, f"cited but not in references.md: {missing}"

    def renumber(s: str) -> str:
        def new(m):
            return "[" + ", ".join(str(order[int(n)]) for n in m.group(1).split(", ")) + "]"

        # a non-breaking hyphen keeps "Section V-G" on one line
        return re.sub(r"(Section [IVX]+)-([A-Z])", "\\1\u2011\\2", CITE.sub(new, s))

    def fix(b):
        if b["t"] in ("p", "h"):
            return {**b, "runs": map_runs(b["runs"], renumber)}
        if b["t"] in ("ul", "ol"):
            return {**b, "items": [map_runs(item, renumber) for item in b["items"]]}
        if b["t"] == "img":
            return {**b, "caption": renumber(b["caption"])}
        return b

    ref_list = []
    for old, new in sorted(order.items(), key=lambda kv: kv[1]):
        runs = [{**r, "code": False} for r in inline(refs[old])]
        ref_list.append({"n": new, "runs": runs})
    return {
        "title": title,
        "authors": authors,
        "affiliation": affiliation,
        "abstract": map_runs(abstract[0], renumber),
        "index_terms": index_terms[0].split("—", 1)[1].strip(),
        "blocks": [fix(b) for b in out],
        "references": ref_list,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node-modules", required=True, help="folder that holds node_modules/docx")
    a = ap.parse_args()
    spec = build_spec(SOURCE.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "paper.json"
        path.write_text(json.dumps(spec))
        env = {**os.environ, "NODE_PATH": str(Path(a.node_modules) / "node_modules")}
        subprocess.run(["node", str(ROOT / "scripts" / "paper_docx.js"), str(path), str(OUT)],
                       check=True, env=env)  # fmt: skip
    print(f"wrote {OUT.relative_to(ROOT)}: {len(spec['references'])} references")


if __name__ == "__main__":
    main()
