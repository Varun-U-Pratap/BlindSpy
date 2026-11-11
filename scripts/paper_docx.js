// Write the research-paper DOCX from the spec made by scripts/build_paper.py.
// Layout in the style of an IEEE conference paper: US Letter, title block across the page, then
// two columns; tables and figures marked wide span both columns (continuous section breaks).
// Usage (called by build_paper.py): NODE_PATH=<dir>/node_modules node paper_docx.js spec.json out.docx
const fs = require("fs");
const {
  AlignmentType, BorderStyle, Document, Footer, ImageRun, LevelFormat, Packer, PageNumber,
  Paragraph, SectionType, Table, TableCell, TableLayoutType, TableRow, TabStopType, TextRun,
  WidthType,
} = require("docx");

const [specPath, outPath] = process.argv.slice(2);
const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));

const FONT = "Times New Roman";
const PAGE = { width: 12240, height: 15840 }; // US Letter, DXA
const MARGIN = { top: 1080, bottom: 1440, left: 1080, right: 1080, footer: 720 };
const GAP = 360; // space between the two columns
const FULL = PAGE.width - MARGIN.left - MARGIN.right; // 10080
const COLUMN = (FULL - GAP) / 2; // 4860
const DXA_PER_PX = 15; // 1440 DXA per inch, 96 px per inch

function runs(list, size, extra = {}) {
  return list.map((r) => new TextRun({
    text: r.text, bold: !!r.b || !!extra.bold, italics: !!r.i || !!extra.italics, size, font: FONT,
    ...(r.code ? { font: "Courier New" } : {}),
  }));
}

function body(children, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: opts.noIndent ? undefined : { firstLine: 202 },
    spacing: { after: 0, line: 240 },
    children,
  });
}

function tableBlock(b, width) {
  const plain = b.rows.map((row) => row.map((c) => c.map((r) => r.text).join("")));
  const ncol = plain[0].length;
  // width per column: its longest data cell, or the longest word of its header (headers wrap)
  const longestWord = (t) => Math.max(...t.split(/\s+/).map((w) => w.length));
  const chars = [...Array(ncol).keys()].map((j) => 3 + Math.max(5, longestWord(plain[0][j] || ""),
    ...plain.slice(1).map((r) => (r[j] || "").length)));
  const total = chars.reduce((s, c) => s + c, 0);
  const widths = chars.map((c) => Math.floor((width * c) / total));
  widths[ncol - 1] += width - widths.reduce((s, w) => s + w, 0);
  const rule = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const last = b.rows.length - 1;
  const rows = b.rows.map((row, i) => new TableRow({
    tableHeader: i === 0,
    cantSplit: true,
    children: widths.map((w, j) => new TableCell({
      width: { size: w, type: WidthType.DXA },
      borders: {
        top: i <= 1 ? rule : none,
        bottom: i === 0 || i === last ? rule : none,
        left: none, right: none,
      },
      margins: { top: 20, bottom: 20, left: 50, right: 50 },
      children: [new Paragraph({
        alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: runs(row[j] || [], 15, { bold: i === 0 }),
      })],
    })),
  }));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 0 },
      children: [new TextRun({ text: b.label, size: 16, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 80 },
      children: [new TextRun({ text: b.caption, size: 16, font: FONT, smallCaps: true })] }),
    new Table({ width: { size: width, type: WidthType.DXA }, columnWidths: widths,
      layout: TableLayoutType.FIXED, alignment: AlignmentType.CENTER, rows }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

function imageBlock(b, width) {
  const maxPx = width / DXA_PER_PX;
  const scale = Math.min(1, maxPx / b.w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(b.path),
        transformation: { width: Math.round(b.w * scale), height: Math.round(b.h * scale) },
        altText: { title: b.caption, description: b.caption, name: b.label } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 }, children: [
      new TextRun({ text: `${b.label} `, size: 16, font: FONT }),
      new TextRun({ text: b.caption, size: 16, font: FONT }),
    ] }),
  ];
}

// Title block (one column)
const titleBlock = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: spec.title, size: 48, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: spec.authors, size: 22, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 280 },
    children: [new TextRun({ text: spec.affiliation, size: 18, italics: true, font: FONT })] }),
];

// Two-column body; wide tables/figures get a one-column section of their own
const sections = [{ columns: 1, children: titleBlock }];
let current = [];
const flush = () => { if (current.length) sections.push({ columns: 2, children: current }); current = []; };

current.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: [
  new TextRun({ text: "Abstract", bold: true, italics: true, size: 18, font: FONT }),
  new TextRun({ text: "—", bold: true, size: 18, font: FONT }),
  ...runs(spec.abstract, 18, { bold: true }),
] }));
current.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: [
  new TextRun({ text: "Index Terms", bold: true, italics: true, size: 18, font: FONT }),
  new TextRun({ text: `—${spec.index_terms}`, bold: true, size: 18, font: FONT }),
] }));

let listCount = 0;
for (const b of spec.blocks) {
  if (b.t === "h" && b.level === 1) {
    current.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
      spacing: { before: 200, after: 100 },
      style: "SectionHeading",
      children: [new TextRun({ text: `${b.label}. `, size: 20, font: FONT, smallCaps: true }),
        ...b.runs.map((r) => new TextRun({ text: r.text, size: 20, font: FONT, smallCaps: true }))] }));
  } else if (b.t === "h") {
    current.push(new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 }, children: [
      new TextRun({ text: `${b.label}. `, italics: true, size: 20, font: FONT }),
      ...runs(b.runs, 20, { italics: true }),
    ] }));
  } else if (b.t === "p") {
    current.push(body(runs(b.runs, 20)));
  } else if (b.t === "ul" || b.t === "ol") {
    listCount += 1;
    for (const item of b.items) {
      current.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED,
        numbering: { reference: b.t === "ul" ? "bullets" : "numbers", level: 0, instance: listCount },
        spacing: { after: 20 }, children: runs(item, 20) }));
    }
  } else if (b.t === "table" || b.t === "img") {
    const make = b.t === "table" ? tableBlock : imageBlock;
    if (b.wide) {
      flush();
      sections.push({ columns: 1, children: make(b, FULL) });
    } else {
      current.push(...make(b, COLUMN));
    }
  }
}

current.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text: "References", size: 20, font: FONT, smallCaps: true })] }));
for (const r of spec.references) {
  current.push(new Paragraph({
    alignment: AlignmentType.LEFT,
    tabStops: [{ type: TabStopType.LEFT, position: 360 }],
    indent: { left: 360, hanging: 360 },
    spacing: { after: 30 },
    children: [new TextRun({ text: `[${r.n}]\t`, size: 16, font: FONT }), ...runs(r.runs, 16)],
  }));
}
flush();

const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 16, font: FONT })] })] });

const doc = new Document({
  title: spec.title,
  creator: "BlindSpy team",
  lastModifiedBy: "BlindSpy team",
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "SectionHeading", name: "Section Heading", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 20, smallCaps: true },
        paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 0 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 200 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] },
  ] },
  sections: sections.map((s, i) => ({
    properties: {
      ...(i === 0 ? {} : { type: SectionType.CONTINUOUS }),
      page: { size: PAGE, margin: MARGIN },
      column: { count: s.columns, space: GAP },
    },
    footers: { default: footer },
    children: s.children,
  })),
});
Packer.toBuffer(doc).then((buf) => fs.writeFileSync(outPath, buf));
