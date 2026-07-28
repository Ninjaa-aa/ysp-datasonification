/**
 * Builds the YSP progress deck.
 *
 * Typography follows Dr. Malaska's 2026-07-23 review of the previous deck:
 *   - Nothing below 18 pt (16 pt allowed for slide numbers only), which means
 *     the text has to be cut, not shrunk.
 *   - No italics — he was advised they are hard for some people to read.
 *   - At most 3-4 style changes per slide; the code font is exempt.
 *   - A space between every value and its unit ("200 ms", not "200ms").
 *     Percentages are the exception ("56.6%").
 *   - Slide numbers, low on the slide.
 *   - Output filename carries name + project + date stamp.
 * Kept from his "what worked" list: ALL-CAPS titles with an action verb, one
 * self-contained story per slide, varied layouts, orange for problems and
 * blue for results, oversized numerals, main text plus subtext.
 */

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Hammad Zahid";
pres.title = "Sounds of Deep Ice Fluorescence — Sound Quality";

// Palette: two accents only — blue for results, orange for problems.
const BG = "0A0F1E";
const WHITE = "FFFFFF";
const GLACIER = "5BC8E8";
const ORANGE = "E8935B";
const DIM_WHITE = "AABBCC";
const DARK_PANEL = "111829";

const FONT = "Arial";
const MONO = "Courier New";

// Minimum body size is 18 pt; 16 pt is reserved for the slide number.
const SZ = { title: 28, lead: 22, body: 18, big: 40, num: 16 };

const ROOT = path.resolve(__dirname, "..");
const img = (p) => path.join(ROOT, p);

let slideNo = 0;

function newSlide(titleText, subtitle) {
  const s = pres.addSlide();
  s.background = { color: BG };
  slideNo += 1;

  if (titleText) {
    s.addText(titleText, {
      x: 0.6, y: 0.3, w: 9, h: 0.6,
      fontSize: SZ.title, fontFace: FONT, color: WHITE, bold: true,
    });
  }
  if (subtitle) {
    s.addText(subtitle, {
      x: 0.6, y: 0.95, w: 9, h: 0.4,
      fontSize: SZ.body, fontFace: FONT, color: DIM_WHITE,
    });
  }
  // Slide number, low left.
  s.addText(String(slideNo), {
    x: 0.6, y: 5.15, w: 0.6, h: 0.3,
    fontSize: SZ.num, fontFace: FONT, color: DIM_WHITE,
  });
  return s;
}

function panel(slide, x, y, w, h) {
  slide.addShape(pres.ShapeType.rect, {
    x, y, w, h, fill: { color: DARK_PANEL }, rectRadius: 0.08,
  });
}

// =========================================================================
// 1 — Title
// =========================================================================
let s1 = pres.addSlide();
s1.background = { color: BG };
slideNo += 1;

s1.addText("SOUNDS OF DEEP ICE\nFLUORESCENCE", {
  x: 0.8, y: 1.15, w: 8.4, h: 2.2,
  fontSize: SZ.big, fontFace: FONT, color: WHITE, bold: true,
  lineSpacingMultiple: 1.1,
});
s1.addShape(pres.ShapeType.rect, {
  x: 0.8, y: 3.45, w: 2.0, h: 0.04, fill: { color: GLACIER },
});
s1.addText("Making the sonification pleasant to listen to", {
  x: 0.8, y: 3.7, w: 8, h: 0.45,
  fontSize: SZ.lead, fontFace: FONT, color: GLACIER,
});
s1.addText("Hammad Zahid   |   BMSIS Young Scientist Program   |   July 2026", {
  x: 0.8, y: 4.4, w: 8.5, h: 0.4,
  fontSize: SZ.body, fontFace: FONT, color: DIM_WHITE,
});
s1.addNotes(
  "The pipeline bugs are fixed, but the output still did not sound good. " +
  "This update is about finding out why — and the answer was something none of our existing checks could detect."
);

// =========================================================================
// 2 — The problem our tests could not see
// =========================================================================
let s2 = newSlide("OUR TESTS PASSED, THE AUDIO DID NOT", null);

const checks = [
  { v: "0.998", l: "Articulation — notes cleanly separated", c: GLACIER },
  { v: "5.0/s", l: "Onset rate — one attack per row", c: GLACIER },
  { v: "0%", l: "Clipping — signal integrity clean", c: GLACIER },
];
let cy = 1.5;
checks.forEach((k) => {
  panel(s2, 0.6, cy, 8.9, 0.78);
  s2.addText(k.v, {
    x: 0.85, y: cy + 0.08, w: 1.5, h: 0.6,
    fontSize: SZ.lead, fontFace: FONT, color: k.c, bold: true, valign: "middle",
  });
  s2.addText(k.l, {
    x: 2.5, y: cy + 0.08, w: 6.8, h: 0.6,
    fontSize: SZ.body, fontFace: FONT, color: DIM_WHITE, valign: "middle",
  });
  cy += 0.9;
});

panel(s2, 0.6, 4.2, 8.9, 0.85);
s2.addText(
  "Every metric measured notes in TIME.\nNone measured notes sounding TOGETHER.",
  {
    x: 0.85, y: 4.28, w: 8.4, h: 0.7,
    fontSize: SZ.lead, fontFace: FONT, color: ORANGE, bold: true, valign: "middle",
  }
);
s2.addNotes(
  "This is the key lesson. Our guards checked whether notes were separated in time — attack, decay, silence between them. They all passed. " +
  "Nothing checked whether the notes sounding at the same moment were consonant with each other. A dense clashing cluster chord scores a perfect articulation."
);

// =========================================================================
// 3 — Diagnosis
// =========================================================================
let s3 = newSlide("DIAGNOSIS: A PERMANENT CLUSTER CHORD",
  "Measured with the Sethares sensory dissonance model.");

const findings = [
  { n: "74%", t: "of rows sounded 5 or more voices at once",
    b: "A real wind chime strikes one tube at a time." },
  { n: "3.6x", t: "more dissonant than the same notes as pure sines",
    b: "The scale was never the problem — the timbre was." },
  { n: "66%", t: "of it came from overtones hitting other channels",
    b: "Bell harmonics landed 55-75 Hz from other fundamentals." },
];
let fy = 1.55;
findings.forEach((f) => {
  panel(s3, 0.6, fy, 8.9, 1.05);
  s3.addText(f.n, {
    x: 0.8, y: fy + 0.2, w: 1.35, h: 0.65,
    fontSize: 26, fontFace: FONT, color: GLACIER, bold: true, valign: "middle",
  });
  s3.addText(f.t, {
    x: 2.3, y: fy + 0.12, w: 7.0, h: 0.4,
    fontSize: SZ.body, fontFace: FONT, color: WHITE, bold: true,
  });
  s3.addText(f.b, {
    x: 2.3, y: fy + 0.55, w: 7.0, h: 0.4,
    fontSize: SZ.body, fontFace: FONT, color: DIM_WHITE,
  });
  fy += 1.15;
});
s3.addNotes(
  "The Sethares model quantifies how rough two partials sound together based on how close they sit inside a critical band. " +
  "Roughness peaks around 55 to 75 Hz apart in this frequency region — exactly where our bell overtones were landing relative to other channels' fundamentals. " +
  "Only 1.4 percent of rows sounded a single voice."
);

// =========================================================================
// 4 — The fix
// =========================================================================
let s4 = newSlide("TWO CHANGES CUT HARSHNESS BY 87%", null);

const fixes = [
  { t: "Limit to the 3 loudest voices per row",
    c: "--max-voices 3",
    e: "Keeps 64% of amplitude." },
  { t: "Soften the low timbre",
    c: "bell -> soft",
    e: "Overtones clear their neighbours." },
];
let xy = 1.4;
fixes.forEach((f) => {
  panel(s4, 0.6, xy, 5.7, 1.45);
  s4.addText(f.t, {
    x: 0.85, y: xy + 0.1, w: 5.2, h: 0.35,
    fontSize: SZ.body, fontFace: FONT, color: GLACIER, bold: true,
  });
  s4.addText(f.c, {
    x: 0.85, y: xy + 0.5, w: 5.2, h: 0.32,
    fontSize: SZ.body, fontFace: MONO, color: WHITE,
  });
  s4.addText(f.e, {
    x: 0.85, y: xy + 0.87, w: 5.2, h: 0.55,
    fontSize: SZ.body, fontFace: FONT, color: DIM_WHITE, valign: "top",
  });
  xy += 1.6;
});

panel(s4, 6.6, 1.35, 2.9, 3.15);
s4.addText("ROUGHNESS", {
  x: 6.7, y: 1.5, w: 2.7, h: 0.35,
  fontSize: SZ.body, fontFace: FONT, color: GLACIER, bold: true, align: "center",
});
s4.addText("0.371\n\u2193\n0.049", {
  x: 6.7, y: 1.9, w: 2.7, h: 1.5,
  fontSize: 28, fontFace: FONT, color: WHITE, bold: true, align: "center",
  lineSpacingMultiple: 1.15,
});
s4.addText("chime preset", {
  x: 6.7, y: 3.42, w: 2.7, h: 0.35,
  fontSize: SZ.body, fontFace: FONT, color: DIM_WHITE, align: "center",
});
s4.addText("Articulation held\nat 0.998", {
  x: 6.7, y: 3.85, w: 2.7, h: 0.6,
  fontSize: SZ.body, fontFace: FONT, color: GLACIER, align: "center",
});
s4.addNotes(
  "Voice limiting is the big one — an 89 percent reduction on its own. I checked the fidelity cost: we keep about 64 percent of the amplitude, " +
  "and because the loudest band varies constantly the dominant-band information is preserved. " +
  "I also tested raising the pitch, which looked better on the model but pushed 15 percent of the energy above 3 kHz and became piercing, so the pitches are unchanged."
);

// =========================================================================
// 5 — Before and after
// =========================================================================
let s5 = newSlide("SAME DATA, CLEANER SOUND", null);

s5.addText("BEFORE", {
  x: 0.6, y: 1.35, w: 4.4, h: 0.35,
  fontSize: SZ.body, fontFace: FONT, color: ORANGE, bold: true, align: "center",
});
s5.addImage({ path: img("outputs/spectrogram_legacy.png"), x: 0.6, y: 1.75, w: 4.4, h: 1.75 });

s5.addText("AFTER", {
  x: 5.2, y: 1.35, w: 4.4, h: 0.35,
  fontSize: SZ.body, fontFace: FONT, color: GLACIER, bold: true, align: "center",
});
s5.addImage({ path: img("outputs/spectrogram_tuned.png"), x: 5.2, y: 1.75, w: 4.4, h: 1.75 });

const stats = [
  { l: "Voices per row", v: "5.8 -> 3.0" },
  { l: "Roughness", v: "0.371 -> 0.049" },
  { l: "Articulation", v: "0.998 held" },
];
stats.forEach((st, i) => {
  const sx = 0.6 + i * 3.05;
  panel(s5, sx, 3.75, 2.8, 0.95);
  s5.addText(st.v, {
    x: sx, y: 3.85, w: 2.8, h: 0.4,
    fontSize: SZ.lead, fontFace: FONT, color: WHITE, align: "center", bold: true,
  });
  s5.addText(st.l, {
    x: sx, y: 4.28, w: 2.8, h: 0.32,
    fontSize: SZ.body, fontFace: FONT, color: DIM_WHITE, align: "center",
  });
});
s5.addNotes(
  "Same descent on both sides. On the left the partials smear together; on the right the tones are cleanly separated. " +
  "The old rendering is still available as preset chime-legacy, so nothing you have already listened to becomes irreproducible."
);

// =========================================================================
// 6 — Status of your requests
// =========================================================================
let s6 = newSlide("YOUR REQUESTS — WHERE THEY STAND", null);

const reqs = [
  { s: "DONE", c: GLACIER, t: "Threshold study reproduced exactly",
    b: "--target-tones solves the threshold from the data." },
  { s: "DONE", c: GLACIER, t: "Two-function design implemented",
    b: "Trigger separated from intensity encoding." },
  { s: "DONE", c: GLACIER, t: "Stray frequencies explained",
    b: "You were right — 97% were independent of the data." },
  { s: "OPEN", c: ORANGE, t: "Sustain / tail-out",
    b: "Both attempts measured worse. Needs a synthesis change." },
];
let ry = 1.25;
reqs.forEach((r) => {
  panel(s6, 0.6, ry, 8.9, 0.84);
  s6.addText(r.s, {
    x: 0.8, y: ry + 0.1, w: 1.1, h: 0.68,
    fontSize: SZ.body, fontFace: FONT, color: r.c, bold: true, valign: "middle",
  });
  s6.addText(r.t, {
    x: 2.0, y: ry + 0.08, w: 7.3, h: 0.36,
    fontSize: SZ.body, fontFace: FONT, color: WHITE, bold: true,
  });
  s6.addText(r.b, {
    x: 2.0, y: ry + 0.46, w: 7.3, h: 0.36,
    fontSize: SZ.body, fontFace: FONT, color: DIM_WHITE,
  });
  ry += 0.9;
});

s6.addText("Next: which event threshold should be the default — 400, 600 or 900?", {
  x: 0.6, y: 4.95, w: 8.9, h: 0.4,
  fontSize: SZ.body, fontFace: FONT, color: GLACIER, align: "center",
});
s6.addNotes(
  "Three requests closed. The sustain one I want to flag honestly — both implementations measured worse than no tail at all, " +
  "so it ships switched off rather than degrading the sound. It needs notes that can ring across row boundaries. " +
  "And the open question for you is which threshold becomes the default; I have attached renders at all three."
);

// =========================================================================
// Export — filename carries name, project and date stamp
// =========================================================================
// Local date, not toISOString() — that reports UTC and can stamp yesterday.
const now = new Date();
const stamp = [
  now.getFullYear(),
  String(now.getMonth() + 1).padStart(2, "0"),
  String(now.getDate()).padStart(2, "0"),
].join("-");
const base = `Hammad_Zahid_YSP_Update_${stamp}`;
const outDir = path.join(ROOT, "outputs");
const outPath = path.join(outDir, `${base}.pptx`);

pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`Presentation saved to: ${outPath}`);
  // Remove any earlier build so the dated file is unambiguous.
  const legacy = path.join(outDir, "ysp_weekly_presentation.pptx");
  if (fs.existsSync(legacy)) {
    fs.unlinkSync(legacy);
    console.log("Removed previous undated build.");
  }
  console.log(`\nExport a PDF too (his .pptx would not open):`);
  console.log(`  libreoffice --headless --convert-to pdf --outdir outputs "${outPath}"`);
});
