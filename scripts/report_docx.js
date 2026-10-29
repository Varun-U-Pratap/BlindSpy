// Write the report DOCX from the block list made by scripts/build_report.py.
// Usage (called by build_report.py): NODE_PATH=<dir>/node_modules node report_docx.js blocks.json out.docx
const fs = require("fs");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat, Packer,
  PageBreak, PageNumber, Paragraph, ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const [specPath, outPath] = process.argv.slice(2);
const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));

// A4 with the PDF's margins (18 mm top/bottom, 16 mm left/right), in DXA (1440 per inch)
const PAGE = { width: 11906, height: 16838 };
const MARGIN = { top: 1020, bottom: 1020, left: 907, right: 907 };
const CONTENT = PAGE.width - MARGIN.left - MARGIN.right;
const MAX_IMG_PX = 640; // about the content width at 96 px per inch

function runs(list, size) {
  return list.map((r) => new TextRun({
    text: r.text, bold: !!r.b, italics: !!r.i, size,
    ...(r.code ? { font: "Menlo", shading: { type: ShadingType.CLEAR, color: "auto", fill: "F2F2F2" } } : {}),
  }));
}

function tableBlock(b) {
  const plain = b.plain;
  const ncol = plain[0].length;
  const size = Math.round(b.font_pt * 2); // half-points, chosen by build_report.py
  const chars = [...Array(ncol).keys()].map((j) =>
    Math.min(40, Math.max(5, ...plain.map((r) => (r[j] || "").length))));
  const total = chars.reduce((s, c) => s + c, 0);
  const widths = chars.map((c) => Math.floor((CONTENT * c) / total));
  widths[ncol - 1] += CONTENT - widths.reduce((s, w) => s + w, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const rows = b.rows.map((row, i) => new TableRow({
    tableHeader: i === 0,
    cantSplit: true,
    children: widths.map((w, j) => new TableCell({
      width: { size: w, type: WidthType.DXA },
      borders,
      margins: { top: 30, bottom: 30, left: 50, right: 50 },
      ...(i === 0 ? { shading: { type: ShadingType.CLEAR, color: "auto", fill: "E8EEF5" } } : {}),
      children: [new Paragraph({ children: runs(row[j] || [], size).map((r) => r) })],
    })),
  }));
  return new Table({ width: { size: CONTENT, type: WidthType.DXA }, columnWidths: widths, rows });
}

const children = [
  new Paragraph({ spacing: { before: 4000, after: 300 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: spec.title, bold: true, size: 60 })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: spec.subtitle, size: 28 })] }),
  ...spec.front.map((t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: t, size: 22 })] })),
  new Paragraph({ children: [new PageBreak()] }),
];
let listCount = 0;
for (const b of spec.blocks) {
  if (b.t === "h") {
    const level = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][b.level - 1];
    children.push(new Paragraph({ heading: level, children: runs(b.runs) }));
  } else if (b.t === "p") {
    children.push(new Paragraph({ spacing: { after: 120 }, children: runs(b.runs) }));
  } else if (b.t === "ul" || b.t === "ol") {
    listCount += 1;
    for (const item of b.items) {
      children.push(new Paragraph({
        numbering: { reference: b.t === "ul" ? "bullets" : "numbers", level: 0, instance: listCount },
        spacing: { after: 60 }, children: runs(item),
      }));
    }
  } else if (b.t === "table") {
    children.push(tableBlock(b));
    children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  } else if (b.t === "img") {
    const scale = Math.min(1, MAX_IMG_PX / b.w);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, keepNext: true, children: [
      new ImageRun({ type: "png", data: fs.readFileSync(b.path),
        transformation: { width: Math.round(b.w * scale), height: Math.round(b.h * scale) },
        altText: { title: b.alt, description: b.alt, name: b.alt } }),
    ] }));
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: b.caption, italics: true, size: 17 })] }));
  } else if (b.t === "pagebreak") {
    children.push(new Paragraph({ children: [new PageBreak()] }));
  }
}

const doc = new Document({
  title: spec.title,
  creator: "BlindSpy team",
  lastModifiedBy: "BlindSpy team",
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true }, paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true }, paragraph: { spacing: { before: 280, after: 100 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: PAGE, margin: MARGIN } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 16 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => fs.writeFileSync(outPath, buf));
