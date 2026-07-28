const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Hammad Zahid";
pres.title = "Sounds of Deep Ice Fluorescence — Sound Quality";

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

function slideTitle(slide, text, subtitle) {
  slide.background = { color: BG };
  slide.addText(text, {
    x: 0.6, y: 0.25, w: 9, h: 0.55,
    fontSize: 26, fontFace: "Arial", color: WHITE, bold: true,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.6, y: 0.82, w: 9, h: 0.4,
      fontSize: 12, fontFace: "Arial", color: DIM_WHITE,
    });
  }
}

function panel(slide, x, y, w, h) {
  slide.addShape(pres.ShapeType.rect, {
    x, y, w, h, fill: { color: DARK_PANEL }, rectRadius: 0.08,
  });
}

// =========================================================================
// SLIDE 1: Title
// =========================================================================
let s1 = pres.addSlide();
s1.background = { color: BG };

s1.addText("SOUNDS OF DEEP ICE\nFLUORESCENCE", {
  x: 0.8, y: 1.2, w: 8.4, h: 2.2,
  fontSize: 40, fontFace: "Arial", color: WHITE, bold: true,
  lineSpacingMultiple: 1.1,
});
s1.addShape(pres.ShapeType.rect, {
  x: 0.8, y: 3.5, w: 2.0, h: 0.04, fill: { color: GLACIER },
});
s1.addText("Making the sonification pleasant to listen to — July 2026", {
  x: 0.8, y: 3.75, w: 8, h: 0.5,
  fontSize: 20, fontFace: "Arial", color: GLACIER,
});
s1.addText("Hammad Zahid", {
  x: 0.8, y: 4.5, w: 8, h: 0.4,
  fontSize: 18, fontFace: "Arial", color: WHITE,
});
s1.addText("BMSIS Young Scientist Program", {
  x: 0.8, y: 4.95, w: 8, h: 0.35,
  fontSize: 14, fontFace: "Arial", color: DIM_WHITE,
});
s1.addNotes(
  "Since the last update the pipeline bugs are fixed, but the output still did not sound good. " +
  "This session was about finding out why — and the answer turned out to be something none of our existing checks could see."
);

// =========================================================================
// SLIDE 2: The problem our tests could not see
// =========================================================================
let s2 = pres.addSlide();
slideTitle(s2, "OUR QUALITY CHECKS COULD NOT HEAR HARSHNESS",
  "The audio scored near-perfect on every metric we had, and still sounded bad.");

const checks = [
  { t: "Articulation  0.998 / 1.0", b: "Notes ARE cleanly separated by silence. Passed.", c: GLACIER },
  { t: "Onset rate  5.0 / sec", b: "Exactly one attack per row. Passed.", c: GLACIER },
  { t: "Clipping 0%   Clicks 0", b: "Signal integrity is clean. Passed.", c: GLACIER },
  { t: "…but it sounded harsh", b: "Every metric measured notes in TIME. None measured notes sounding TOGETHER.", c: ORANGE },
];
let cy = 1.35;
checks.forEach((k) => {
  panel(s2, 0.6, cy, 8.9, 0.72);
  s2.addText(k.t, {
    x: 0.8, y: cy + 0.06, w: 3.4, h: 0.6,
    fontSize: 13, fontFace: "Arial", color: k.c, bold: true, valign: "middle",
  });
  s2.addText(k.b, {
    x: 4.2, y: cy + 0.06, w: 5.1, h: 0.6,
    fontSize: 11, fontFace: "Arial", color: DIM_WHITE, valign: "middle",
  });
  cy += 0.82;
});

s2.addText(
  "A dense, clashing cluster chord scores a perfect articulation score. That is exactly what we were shipping.",
  {
    x: 0.6, y: 4.75, w: 8.9, h: 0.4,
    fontSize: 13, fontFace: "Arial", color: GLACIER, italic: true, align: "center",
  }
);
s2.addNotes(
  "This is the key lesson. Our guards checked whether notes were separated in time — attack, decay, silence between them. They all passed. " +
  "But nothing checked whether the notes sounding at the same moment were consonant with each other."
);

// =========================================================================
// SLIDE 3: Diagnosis
// =========================================================================
let s3 = pres.addSlide();
slideTitle(s3, "DIAGNOSIS: A PERMANENT 20-PARTIAL CLUSTER CHORD",
  "Measured with the Sethares / Plomp-Levelt sensory dissonance model.");

const findings = [
  {
    n: "1", t: "Too many voices at once",
    b: "74% of rows sounded 5+ voices; 23.8% sounded all eight. Only 1.4% sounded a single voice — a real wind chime strikes one tube at a time.",
  },
  {
    n: "2", t: "The scale was never the problem",
    b: "The same pentatonic pitches as pure sines measure 0.179 dissonance. The preset as it stood measured 0.649 — worse than a major triad.",
  },
  {
    n: "3", t: "Overtones collided with other channels",
    b: "The bell voice's 2x/3x/4x harmonics landed 55–75 Hz from OTHER channels' fundamentals (550 Hz vs 495 Hz) — the peak of the roughness curve. 66% of all dissonance.",
  },
];
let fy = 1.35;
findings.forEach((f) => {
  panel(s3, 0.6, fy, 8.9, 1.05);
  s3.addText(f.n, {
    x: 0.78, y: fy + 0.22, w: 0.5, h: 0.55,
    fontSize: 24, fontFace: "Arial", color: GLACIER, bold: true, valign: "middle",
  });
  s3.addText(f.t, {
    x: 1.35, y: fy + 0.1, w: 7.9, h: 0.3,
    fontSize: 14, fontFace: "Arial", color: WHITE, bold: true,
  });
  s3.addText(f.b, {
    x: 1.35, y: fy + 0.42, w: 7.9, h: 0.55,
    fontSize: 11, fontFace: "Arial", color: DIM_WHITE, valign: "top",
  });
  fy += 1.15;
});
s3.addNotes(
  "I used the Sethares model, which quantifies how rough two partials sound together based on how close they sit within a critical band. " +
  "Roughness peaks when tones are around 55 to 75 Hz apart in this frequency region — exactly where our bell overtones were landing relative to other channels' fundamentals."
);

// =========================================================================
// SLIDE 4: The fix
// =========================================================================
let s4 = pres.addSlide();
slideTitle(s4, "TWO CHANGES, EACH MEASURED INDEPENDENTLY", null);

const fixes = [
  {
    t: "Fix 1 — Voice limiting  (--max-voices 3)",
    b: "Only the loudest 3 channels sound per row.",
    e: "Retains 64% of per-row amplitude. The loudest channel uses all 8 bands and changes on 82% of rows, so the data survives.",
  },
  {
    t: "Fix 2 — Softer low timbre",
    b: "low group: bell (1,2,3,4x) -> soft (1x + quiet octave)",
    e: "Overtones no longer sit inside a neighbouring channel's critical band.",
  },
];
let xy = 1.15;
fixes.forEach((f) => {
  panel(s4, 0.6, xy, 5.9, 1.35);
  s4.addText(f.t, {
    x: 0.8, y: xy + 0.08, w: 5.5, h: 0.3,
    fontSize: 13, fontFace: "Arial", color: GLACIER, bold: true,
  });
  s4.addText(f.b, {
    x: 0.8, y: xy + 0.42, w: 5.5, h: 0.28,
    fontSize: 10, fontFace: "Courier New", color: WHITE,
  });
  s4.addText(f.e, {
    x: 0.8, y: xy + 0.74, w: 5.5, h: 0.5,
    fontSize: 10, fontFace: "Arial", color: DIM_WHITE, valign: "top",
  });
  xy += 1.5;
});

panel(s4, 6.7, 1.15, 2.8, 2.85);
s4.addText("ROUGHNESS", {
  x: 6.8, y: 1.25, w: 2.6, h: 0.3,
  fontSize: 12, fontFace: "Arial", color: GLACIER, bold: true, align: "center",
});
s4.addText(
  "chime\n  0.371 -> 0.049\n  (-87%)\n\nambient\n  1.031 -> 0.066\n  (-94%)\n\nArticulation held at\n0.998 — the drone\nfailure mode did\nnot return.",
  {
    x: 6.85, y: 1.6, w: 2.5, h: 2.3,
    fontSize: 11, fontFace: "Arial", color: DIM_WHITE, valign: "top",
  }
);

s4.addText(
  "Pitches deliberately unchanged. Raising the root scored better on the model, but on real audio it pushed 15% of the energy above 3 kHz — piercing.",
  {
    x: 0.6, y: 4.25, w: 8.9, h: 0.45,
    fontSize: 11, fontFace: "Arial", color: ORANGE, italic: true,
  }
);
s4.addNotes(
  "Voice limiting is the big one — an 89 percent reduction on its own. I checked the fidelity cost carefully: we keep about 64 percent of the amplitude, " +
  "and because the loudest band varies constantly, the information about which band is dominant is preserved. " +
  "I also tested raising the pitch, which looked better on paper but made it piercing on real audio, so I left the pitches alone."
);

// =========================================================================
// SLIDE 5: Before / after
// =========================================================================
let s5 = pres.addSlide();
slideTitle(s5, "BEFORE AND AFTER", "Same data, same pitches, same descent.");

s5.addText("BEFORE  (--preset chime-legacy)", {
  x: 0.6, y: 1.15, w: 4.4, h: 0.3,
  fontSize: 11, fontFace: "Arial", color: ORANGE, bold: true, align: "center",
});
s5.addImage({ path: img("outputs/spectrogram_legacy.png"), x: 0.6, y: 1.5, w: 4.4, h: 1.75 });
s5.addText("Roughness 0.302 — dense overlapping partials", {
  x: 0.6, y: 3.3, w: 4.4, h: 0.3,
  fontSize: 10, fontFace: "Arial", color: DIM_WHITE, align: "center",
});

s5.addText("AFTER  (--preset chime)", {
  x: 5.2, y: 1.15, w: 4.4, h: 0.3,
  fontSize: 11, fontFace: "Arial", color: GLACIER, bold: true, align: "center",
});
s5.addImage({ path: img("outputs/spectrogram_tuned.png"), x: 5.2, y: 1.5, w: 4.4, h: 1.75 });
s5.addText("Roughness 0.049 — clean, separated tones", {
  x: 5.2, y: 3.3, w: 4.4, h: 0.3,
  fontSize: 10, fontFace: "Arial", color: DIM_WHITE, align: "center",
});

const stats = [
  { l: "Voices per row", v: "5.8 -> 3.0" },
  { l: "Roughness", v: "0.371 -> 0.049" },
  { l: "Articulation", v: "0.998 (held)" },
];
stats.forEach((st, i) => {
  const sx = 0.6 + i * 3.05;
  panel(s5, sx, 3.75, 2.8, 0.75);
  s5.addText(st.l, {
    x: sx, y: 3.82, w: 2.8, h: 0.25,
    fontSize: 10, fontFace: "Arial", color: GLACIER, align: "center", bold: true,
  });
  s5.addText(st.v, {
    x: sx, y: 4.08, w: 2.8, h: 0.3,
    fontSize: 14, fontFace: "Arial", color: WHITE, align: "center", bold: true,
  });
});
s5.addText("Both renderings ship — chime-legacy reproduces anything you have already heard.", {
  x: 0.6, y: 4.65, w: 8.9, h: 0.35,
  fontSize: 12, fontFace: "Arial", color: WHITE, italic: true, align: "center",
});
s5.addNotes(
  "The spectrograms show the same descent. On the left the partials smear together; on the right the tones are cleanly separated. " +
  "I kept the old rendering available as chime-legacy so nothing you have already listened to becomes irreproducible."
);

// =========================================================================
// SLIDE 6: Your requests — status
// =========================================================================
let s6 = pres.addSlide();
slideTitle(s6, "YOUR REQUESTS — STATUS", null);

const reqs = [
  {
    s: "DONE", c: GLACIER, t: "Threshold study (Jul 9)",
    b: "sonify/events.py reproduces your spreadsheet exactly. --target-tones inverts it: ask for 25 tones, it solves the threshold.",
  },
  {
    s: "DONE", c: GLACIER, t: "Two-function design (Jul 24)",
    b: "Trigger (which peaks sound) fully separated from intensity encoding (how loud).",
  },
  {
    s: "DONE", c: GLACIER, t: "Frequencies not in the dataset (Jul 24)",
    b: "You were right. 97% of the old spectrum was independent of the data; event mode inverts this to 88% data-driven.",
  },
  {
    s: "OPEN", c: ORANGE, t: "Sustain / tail-out (Jul 9)",
    b: "Two implementations measured worse, not better. Needs note-level envelopes spanning rows — a synthesis change, scoped separately.",
  },
  {
    s: "NEED", c: ORANGE, t: "Other borehole / glommed datasets",
    b: "The engine is dataset-agnostic and ready — we just need the files.",
  },
];
let ry = 1.15;
reqs.forEach((r) => {
  panel(s6, 0.6, ry, 8.9, 0.72);
  s6.addText(r.s, {
    x: 0.75, y: ry + 0.06, w: 0.8, h: 0.6,
    fontSize: 11, fontFace: "Arial", color: r.c, bold: true, valign: "middle",
  });
  s6.addText(r.t, {
    x: 1.6, y: ry + 0.06, w: 7.7, h: 0.28,
    fontSize: 12, fontFace: "Arial", color: WHITE, bold: true,
  });
  s6.addText(r.b, {
    x: 1.6, y: ry + 0.34, w: 7.7, h: 0.36,
    fontSize: 10, fontFace: "Arial", color: DIM_WHITE, valign: "top",
  });
  ry += 0.8;
});

s6.addText(
  "A/B pack ready: tuned vs legacy chime, ambient, and event mode at thresholds 400 / 600 / 900 — your pick for the default.",
  {
    x: 0.6, y: 4.85, w: 8.9, h: 0.4,
    fontSize: 12, fontFace: "Arial", color: GLACIER, italic: true, align: "center",
  }
);
s6.addNotes(
  "Three of your requests are closed. The sustain one I want to flag honestly — I tried two implementations and both measured worse than no tail at all, " +
  "so I have left it switched off rather than ship something that degrades the sound. It needs a synthesis architecture change. " +
  "And whenever you can send the other borehole datasets, the tool is ready for them."
);

// =========================================================================
// EXPORT
// =========================================================================
const outPath = path.join(ROOT, "outputs", "ysp_weekly_presentation.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`Presentation saved to: ${outPath}`);
});
