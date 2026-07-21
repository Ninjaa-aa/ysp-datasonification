const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Hammad Zahid";
pres.title = "Sounds of Deep Ice Fluorescence — Weekly Progress";

// Color palette
const BG = "0A0F1E";
const WHITE = "FFFFFF";
const GLACIER = "5BC8E8";
const ORANGE = "E8935B";
const DIM_WHITE = "AABBCC";
const DARK_PANEL = "111829";

// Helper: absolute path from project root
const ROOT = path.resolve(__dirname, "..");
function img(relPath) {
  return path.join(ROOT, relPath);
}

// =========================================================================
// SLIDE 1: Title
// =========================================================================
let slide1 = pres.addSlide();
slide1.background = { color: BG };

slide1.addText("SOUNDS OF DEEP ICE\nFLUORESCENCE", {
  x: 0.8, y: 1.2, w: 8.4, h: 2.2,
  fontSize: 40, fontFace: "Arial", color: WHITE,
  bold: true, lineSpacingMultiple: 1.1,
});

// Accent line
slide1.addShape(pres.ShapeType.rect, {
  x: 0.8, y: 3.5, w: 2.0, h: 0.04,
  fill: { color: GLACIER },
});

slide1.addText("Weekly Progress — July 17–23, 2025", {
  x: 0.8, y: 3.75, w: 8, h: 0.5,
  fontSize: 20, fontFace: "Arial", color: GLACIER,
});

slide1.addText("Hammad Zahid", {
  x: 0.8, y: 4.5, w: 8, h: 0.4,
  fontSize: 18, fontFace: "Arial", color: WHITE,
});

slide1.addText("BMSIS Young Scientist Program", {
  x: 0.8, y: 4.95, w: 8, h: 0.35,
  fontSize: 14, fontFace: "Arial", color: DIM_WHITE,
});

slide1.addNotes(
  "Good morning Dr. Malaska and Dayana. This week I focused on diagnosing and fixing the audio quality issues in our sonification pipeline. " +
  "I'll walk you through what the problem was, how I found the root cause, the three fixes I applied, and the results."
);

// =========================================================================
// SLIDE 2: The Problem
// =========================================================================
let slide2 = pres.addSlide();
slide2.background = { color: BG };

slide2.addText("THE AUDIO SOUNDED LIKE NOISE", {
  x: 0.6, y: 0.3, w: 9, h: 0.6,
  fontSize: 28, fontFace: "Arial", color: WHITE, bold: true,
});

slide2.addText(
  "Last week's chime preset output produced constant rhythmic clicking — " +
  "every 200ms row fired at full volume regardless of whether the ice was fluorescing or silent.",
  {
    x: 0.6, y: 0.95, w: 9, h: 0.55,
    fontSize: 13, fontFace: "Arial", color: DIM_WHITE,
  }
);

// Before spectrogram
slide2.addText("BEFORE FIX", {
  x: 0.6, y: 1.65, w: 4.3, h: 0.35,
  fontSize: 12, fontFace: "Arial", color: ORANGE, bold: true, align: "center",
});
slide2.addImage({
  path: img("outputs/spectrogram_before.png"),
  x: 0.6, y: 2.0, w: 4.3, h: 1.7,
});
slide2.addText("Wall of orange — no silence gaps, every row at full blast", {
  x: 0.6, y: 3.75, w: 4.3, h: 0.35,
  fontSize: 10, fontFace: "Arial", color: DIM_WHITE, align: "center",
});

// After spectrogram
slide2.addText("AFTER FIX", {
  x: 5.2, y: 1.65, w: 4.3, h: 0.35,
  fontSize: 12, fontFace: "Arial", color: GLACIER, bold: true, align: "center",
});
slide2.addImage({
  path: img("outputs/spectrogram_chime_quiet_section.png"),
  x: 5.2, y: 2.0, w: 4.3, h: 1.7,
});
slide2.addText("Distinct pulses against genuine silence — data structure audible", {
  x: 5.2, y: 3.75, w: 4.3, h: 0.35,
  fontSize: 10, fontFace: "Arial", color: DIM_WHITE, align: "center",
});

// Key question
slide2.addShape(pres.ShapeType.rect, {
  x: 0.6, y: 4.35, w: 8.9, h: 0.55,
  fill: { color: DARK_PANEL },
  rectRadius: 0.08,
});
slide2.addText("Why was silence being rendered as loud sound?", {
  x: 0.6, y: 4.35, w: 8.9, h: 0.55,
  fontSize: 16, fontFace: "Arial", color: GLACIER, align: "center", italic: true,
});

slide2.addNotes(
  "The before spectrogram on the left shows what the audio looked like — a solid wall of orange meaning constant energy at every frequency and every time step. " +
  "After the fix, on the right, you can see distinct bright pulses separated by dark purple silence. The question driving this week's work was: why was silence being rendered as loud sound?"
);

// =========================================================================
// SLIDE 3: Diagnostic Investigation
// =========================================================================
let slide3 = pres.addSlide();
slide3.background = { color: BG };

slide3.addText("FOUR INVESTIGATIONS, ONE ROOT CAUSE", {
  x: 0.6, y: 0.25, w: 9, h: 0.55,
  fontSize: 26, fontFace: "Arial", color: WHITE, bold: true,
});

// 2x2 grid of investigation boxes — compact
const boxW = 4.15;
const boxH = 0.82;
const gap = 0.15;
const leftX = 0.6;
const rightX = leftX + boxW + gap;
const topY = 0.9;
const botY = topY + boxH + gap;

const investigations = [
  {
    title: "1 — Raw Data Quality",
    body: "58.6% of all values are zero. Sparse data with occasional spikes up to 4,921.",
    color: GLACIER, x: leftX, y: topY,
  },
  {
    title: "2 — Pipeline Values  ⚠ SMOKING GUN",
    body: "After gain step, every value near 1.0. True zeros mapped to 95% amplitude.",
    color: ORANGE, x: rightX, y: topY,
  },
  {
    title: "3 — Synthesis Output",
    body: "Every row fired at full volume because all channels were near-maximum.",
    color: GLACIER, x: leftX, y: botY,
  },
  {
    title: "4 — ADSR Envelope Shape",
    body: "Confirmed mathematically correct. Not the source of the bug.",
    color: GLACIER, x: rightX, y: botY,
  },
];

investigations.forEach((inv) => {
  slide3.addShape(pres.ShapeType.rect, {
    x: inv.x, y: inv.y, w: boxW, h: boxH,
    fill: { color: DARK_PANEL },
    rectRadius: 0.08,
  });
  slide3.addText(inv.title, {
    x: inv.x + 0.15, y: inv.y + 0.06, w: boxW - 0.3, h: 0.3,
    fontSize: 11, fontFace: "Arial", color: inv.color, bold: true,
  });
  slide3.addText(inv.body, {
    x: inv.x + 0.15, y: inv.y + 0.36, w: boxW - 0.3, h: 0.4,
    fontSize: 10, fontFace: "Arial", color: DIM_WHITE, valign: "top",
  });
});

// Pipeline stages image — positioned below the grid
const imgTopY = botY + boxH + 0.2;
slide3.addImage({
  path: img("diagnostics/outputs/investigation_2_pipeline_stages.png"),
  x: 0.6, y: imgTopY, w: 5.8, h: 2.45,
});

// Bug callout arrow/label
slide3.addShape(pres.ShapeType.rect, {
  x: 6.6, y: imgTopY + 0.6, w: 3.0, h: 1.0,
  fill: { color: "1A1208" },
  line: { color: ORANGE, width: 1.5 },
  rectRadius: 0.08,
});
slide3.addText("← THE BUG\nPanels 3 & 4 should show\nblue (silence) but show\norange (near-maximum)", {
  x: 6.7, y: imgTopY + 0.6, w: 2.8, h: 1.0,
  fontSize: 10, fontFace: "Arial", color: ORANGE, valign: "middle",
});

slide3.addNotes(
  "I ran four separate diagnostic investigations. Investigation 1 confirmed the raw data is 58.6% zeros — genuinely sparse. " +
  "Investigation 2 was the smoking gun: the pipeline heatmap shows panels 3 and 4 as solid orange, meaning the gain normalization step was mapping zeros to near-maximum amplitude. " +
  "Investigation 3 confirmed every row was firing at full volume because of this. Investigation 4 ruled out the ADSR envelope as the cause."
);

// =========================================================================
// SLIDE 4: The Fix
// =========================================================================
let slide4 = pres.addSlide();
slide4.background = { color: BG };

slide4.addText("THREE FIXES, CONFIRMED BY SPECTROGRAM", {
  x: 0.6, y: 0.25, w: 9, h: 0.55,
  fontSize: 26, fontFace: "Arial", color: WHITE, bold: true,
});

const fixes = [
  {
    title: "Fix 1 — Gain Normalization",
    file: "mapping.py",
    before: "work = work - work.min()",
    after: "work = np.clip(work, 0, None)",
    effect: "Zeros stay silent instead of being shifted to 95% amplitude",
  },
  {
    title: "Fix 2 — ADSR Decay Shape",
    file: "synth.py",
    before: "np.linspace(1.0, 0.0, d)  — linear",
    after: "np.exp(-t / 0.3)  — exponential",
    effect: "Notes decay like struck bells, not electronic test tones",
  },
  {
    title: "Fix 3 — Rebinning Aggregation",
    file: "preprocess.py",
    before: "np.mean()  — averages signal with zeros",
    after: "np.max()  — if any band fires, bin fires",
    effect: "Signal/silence contrast preserved through channel grouping",
  },
];

let fixY = 0.9;
fixes.forEach((fix, i) => {
  const fh = 1.1;
  // Background panel
  slide4.addShape(pres.ShapeType.rect, {
    x: 0.6, y: fixY, w: 5.8, h: fh,
    fill: { color: DARK_PANEL },
    rectRadius: 0.08,
  });

  slide4.addText(`${fix.title}  (${fix.file})`, {
    x: 0.8, y: fixY + 0.05, w: 5.4, h: 0.28,
    fontSize: 12, fontFace: "Arial", color: GLACIER, bold: true,
  });

  slide4.addText(`Before:  ${fix.before}`, {
    x: 0.8, y: fixY + 0.32, w: 5.4, h: 0.22,
    fontSize: 10, fontFace: "Courier New", color: ORANGE,
  });

  slide4.addText(`After:   ${fix.after}`, {
    x: 0.8, y: fixY + 0.52, w: 5.4, h: 0.22,
    fontSize: 10, fontFace: "Courier New", color: GLACIER,
  });

  slide4.addText(`→ ${fix.effect}`, {
    x: 0.8, y: fixY + 0.78, w: 5.4, h: 0.22,
    fontSize: 10, fontFace: "Arial", color: DIM_WHITE, italic: true,
  });

  fixY += fh + 0.15;
});

// ADSR shapes image on right
slide4.addText("ADSR Envelope Shapes (after fix)", {
  x: 6.6, y: 0.9, w: 3.0, h: 0.3,
  fontSize: 11, fontFace: "Arial", color: GLACIER, align: "center",
});
slide4.addImage({
  path: img("diagnostics/outputs/investigation_4_adsr_shapes.png"),
  x: 6.6, y: 1.25, w: 3.0, h: 1.5,
});

// Stats box
slide4.addShape(pres.ShapeType.rect, {
  x: 6.6, y: 3.0, w: 3.0, h: 1.6,
  fill: { color: DARK_PANEL },
  rectRadius: 0.08,
});
slide4.addText("VERIFICATION", {
  x: 6.7, y: 3.05, w: 2.8, h: 0.3,
  fontSize: 11, fontFace: "Arial", color: GLACIER, bold: true, align: "center",
});
slide4.addText(
  "Mean amplitude:\n  0.94 → 0.51\n\n" +
  "Near-zero values:\n  3.1% → matching raw data\n\n" +
  "Tests: 80/80 passing",
  {
    x: 6.75, y: 3.35, w: 2.7, h: 1.2,
    fontSize: 10, fontFace: "Arial", color: DIM_WHITE, valign: "top",
  }
);

slide4.addNotes(
  "Three one-line fixes resolved all the issues. Fix 1 was the critical one — replacing the min-shift with a zero-clip in the gain normalization so that true zeros stay at zero amplitude. " +
  "Fix 2 replaced the linear ADSR decay with exponential, giving a natural bell-like sound. Fix 3 changed rebinning from mean to max so sparsity is preserved. " +
  "The ADSR plot on the right shows the new exponential curve shapes. All 80 unit tests pass after the changes."
);

// =========================================================================
// SLIDE 5: Results
// =========================================================================
let slide5 = pres.addSlide();
slide5.background = { color: BG };

slide5.addText("THE ICE IS NOW AUDIBLE", {
  x: 0.6, y: 0.25, w: 9, h: 0.55,
  fontSize: 26, fontFace: "Arial", color: WHITE, bold: true,
});

// Left spectrogram — quiet section
slide5.addText("QUIET SECTION  (rows 1500–1700, depth ~94m)", {
  x: 0.6, y: 0.9, w: 4.4, h: 0.3,
  fontSize: 11, fontFace: "Arial", color: GLACIER, bold: true, align: "center",
});
slide5.addImage({
  path: img("outputs/spectrogram_chime_quiet_section.png"),
  x: 0.6, y: 1.25, w: 4.4, h: 1.75,
});
slide5.addText("Deep purple = genuine silence between fluorescence events", {
  x: 0.6, y: 3.05, w: 4.4, h: 0.3,
  fontSize: 10, fontFace: "Arial", color: DIM_WHITE, align: "center",
});

// Right spectrogram — full dataset
slide5.addText("FULL BOREHOLE DESCENT  (4000 rows, ~800 seconds)", {
  x: 5.2, y: 0.9, w: 4.4, h: 0.3,
  fontSize: 11, fontFace: "Arial", color: GLACIER, bold: true, align: "center",
});
slide5.addImage({
  path: img("outputs/spectrogram_chime_full.png"),
  x: 5.2, y: 1.25, w: 4.4, h: 1.75,
});
slide5.addText("Dense zone at start; sparse geology visible at depth", {
  x: 5.2, y: 3.05, w: 4.4, h: 0.3,
  fontSize: 10, fontFace: "Arial", color: DIM_WHITE, align: "center",
});

// Three stat callout boxes
const statBoxW = 2.8;
const statBoxH = 0.7;
const statY = 3.6;
const stats = [
  { label: "Active Zone RMS", value: "0.119", desc: "Fluorescence-rich ice" },
  { label: "Quiet Section RMS", value: "0.088", desc: "Sparse ice at depth" },
  { label: "Full Dataset RMS", value: "0.072", desc: "89m–96m descent avg" },
];

stats.forEach((st, i) => {
  const sx = 0.6 + i * (statBoxW + 0.25);
  slide5.addShape(pres.ShapeType.rect, {
    x: sx, y: statY, w: statBoxW, h: statBoxH,
    fill: { color: DARK_PANEL },
    rectRadius: 0.08,
  });
  slide5.addText(st.label, {
    x: sx, y: statY + 0.05, w: statBoxW, h: 0.22,
    fontSize: 10, fontFace: "Arial", color: GLACIER, align: "center", bold: true,
  });
  slide5.addText(st.value, {
    x: sx, y: statY + 0.25, w: statBoxW, h: 0.22,
    fontSize: 14, fontFace: "Arial", color: WHITE, align: "center", bold: true,
  });
  slide5.addText(st.desc, {
    x: sx, y: statY + 0.46, w: statBoxW, h: 0.2,
    fontSize: 9, fontFace: "Arial", color: DIM_WHITE, align: "center",
  });
});

// Conclusion line
slide5.addText(
  "The tool now faithfully represents what the ice was actually doing during the laser descent.",
  {
    x: 0.6, y: 4.55, w: 9, h: 0.35,
    fontSize: 13, fontFace: "Arial", color: WHITE, italic: true, align: "center",
  }
);

slide5.addNotes(
  "These two spectrograms are the proof that the pipeline is working correctly. The quiet section on the left shows distinct chime pulses separated by deep purple silence — this is what sparse fluorescence data should sound like. " +
  "The full dataset on the right shows the initial dense zone where the ice was genuinely active, followed by sparser regions at depth. " +
  "The RMS values decrease as the ice gets quieter, confirming the audio faithfully represents the data."
);

// =========================================================================
// SLIDE 6: Next Steps
// =========================================================================
let slide6 = pres.addSlide();
slide6.background = { color: BG };

slide6.addText("WHAT'S NEXT", {
  x: 0.6, y: 0.3, w: 9, h: 0.55,
  fontSize: 28, fontFace: "Arial", color: WHITE, bold: true,
});

const nextSteps = [
  {
    num: "01",
    title: "Tone sustain / tail-out feature",
    body: "Have each tone sustain and tail into the next row to prevent abrupt silences — as Dr. Malaska suggested.",
  },
  {
    num: "02",
    title: "Full borehole video",
    body: "Generate the complete 4000-row video with trail, minimap, colorbar, and the fixed audio pipeline.",
  },
  {
    num: "03",
    title: "Share with Dayana",
    body: "Coordinate GitHub access and sound files so both team members can listen and compare outputs.",
  },
  {
    num: "04",
    title: "Test on additional datasets",
    body: 'Try the "glommed" multi-borehole dataset to validate the tool works generically.',
  },
];

let nsY = 1.15;
nextSteps.forEach((ns) => {
  const nsH = 0.85;
  slide6.addShape(pres.ShapeType.rect, {
    x: 0.6, y: nsY, w: 8.9, h: nsH,
    fill: { color: DARK_PANEL },
    rectRadius: 0.08,
  });

  slide6.addText(ns.num, {
    x: 0.75, y: nsY + 0.12, w: 0.8, h: 0.55,
    fontSize: 24, fontFace: "Arial", color: GLACIER, bold: true, valign: "middle",
  });

  slide6.addText(ns.title, {
    x: 1.4, y: nsY + 0.1, w: 7.8, h: 0.3,
    fontSize: 15, fontFace: "Arial", color: WHITE, bold: true,
  });

  slide6.addText(ns.body, {
    x: 1.4, y: nsY + 0.42, w: 7.8, h: 0.35,
    fontSize: 12, fontFace: "Arial", color: DIM_WHITE,
  });

  nsY += nsH + 0.12;
});

slide6.addNotes(
  "The top priority for next week is the tone sustain feature that Dr. Malaska suggested — having each note tail out longer so the transitions feel more natural. " +
  "I also want to generate the full-length video with all visual features enabled, share files with Dayana so we can coordinate, " +
  "and test the pipeline on the glommed multi-borehole dataset to make sure it generalizes."
);

// =========================================================================
// EXPORT
// =========================================================================
const outPath = path.join(ROOT, "outputs", "ysp_weekly_presentation.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`Presentation saved to: ${outPath}`);
});
