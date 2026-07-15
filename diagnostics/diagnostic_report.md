# DIAGNOSTIC REPORT

## 1. DATA QUALITY
- **Sparsity**: The raw dataset is highly sparse. Exactly **58.6%** of all values are zeros, and **59.6%** are near-zero (< 1).
- **Channels**: The sparsity is extreme in higher index channels. For example, Channels 9-17 are consistently 65% to 83% zero. Channels 1 (27% zero) and 3 (25% zero) are less sparse, but silence is still very prevalent across the dataset.
- **Dynamic Range**: The values are highly skewed. The global mean is ~20.7, but the maximum value is 4921.4. The 75th percentile is only 56.7, while the 99th percentile jumps to 405.8. The data consists mostly of silence punctuated by huge spikes.
- **Verdict**: The data itself is highly sparse, mostly consisting of zero or near-zero values with occasional large spikes.

![Data Quality](/home/hammad/.gemini/antigravity-ide/brain/2586337b-9b03-438a-8488-854111771798/investigation_1_data_quality.png)

## 2. PIPELINE VALUES
- **Reasonable Values**: The amplitude matrix entering synthesis does **not** have reasonable values. Due to a critical bug, true zeros (silence) are being forced to ~94-95% maximum amplitude.
- **% Near-zero**: In the final matrix (after gain scaling and smoothing), only **3.1%** of values are near-zero. This makes absolutely no sense given that 59.6% of the raw data was near-zero! The mean amplitude of the entire dataset becomes an astonishing **0.941** out of 1.0.
- **NaN/Inf**: No NaN or Inf values were found. 
- **Verdict**: The pipeline is producing degenerate amplitude values. The gain normalization (`max_log` mode) applies a double-log transform (since the values were already logged) and incorrectly handles the log min-shift. It takes a raw value of 0 (which becomes `-10` in log space) and shifts/scales it to map to `~0.95` amplitude. This completely destroys the dynamic range and forces silence to be almost as loud as the absolute maximum peaks.

![Pipeline Stages](/home/hammad/.gemini/antigravity-ide/brain/2586337b-9b03-438a-8488-854111771798/investigation_2_pipeline_stages.png)
![Per-Channel Amplitude](/home/hammad/.gemini/antigravity-ide/brain/2586337b-9b03-438a-8488-854111771798/investigation_2_per_channel_amplitude.png)

## 3. SYNTHESIS OUTPUT
- **Peak and RMS**: The output WAV has a peak of `0.999969` (-0.0 dBFS) and an RMS of `0.133404` (-17.5 dBFS).
- **Crest Factor**: The crest factor is **17.5 dB**, which is high (bursty/sparse signal) compared to typical music (~10-15 dB).
- **Per-row Amplitude**: Because the pipeline is pushing all values to near-maximum amplitude, virtually *no* row is silent. The percentage of silence in each 200ms row segment sits consistently around 1.5% to 2.4%.
- **ADSR Envelope**: Every single row triggers a loud percussive burst (because all amplitudes are near 1.0). The tight ADSR envelope causes each 200ms segment to have a massive attack and immediate decay. 
- **Verdict**: The synthesizer is acting correctly based on its inputs, but because it is being fed near-maximum amplitude for every single row, it produces a constant stream of massive percussive hits, creating the relentless "tak tak tak" sound.

![Row Envelopes](/home/hammad/.gemini/antigravity-ide/brain/2586337b-9b03-438a-8488-854111771798/investigation_3_row_envelopes.png)
![Spectrogram](/home/hammad/.gemini/antigravity-ide/brain/2586337b-9b03-438a-8488-854111771798/investigation_3_spectrogram.png)

## 4. ADSR VERIFICATION
- **Linear or Curved**: The envelopes are properly curved (exponential). The R² scores for a linear fit are 0.4065 ("tight") and 0.6430 ("natural"), confirming they are not linear.
- **Tight Shape Decay**: The "tight" shape correctly decays to precisely zero, as intended.
- **Verdict**: The ADSR code is mathematically sound and is correctly generating curved envelopes. It is not the source of the issue.

![ADSR Shapes](/home/hammad/.gemini/antigravity-ide/brain/2586337b-9b03-438a-8488-854111771798/investigation_4_adsr_shapes.png)

## 5. ROOT CAUSE CONCLUSION
The primary problem is **B) The gain normalization is wrong**.

Because the data is so heavily skewed (as shown in Investigation 1), a logarithmic scale was necessary. However, the pipeline applies a double-log (once in `scale_values`, again in `apply_global_gain`). More disastrously, the shift applied in `apply_global_gain` when processing log data (`work - work_min`) takes the log of zero (which is clipped to `-10` due to epsilon) and shifts it to `+10`. When divided by the maximum reference (which is around `+10.48`), it maps raw zeros (absolute silence) to `10.0 / 10.48 = 0.954` amplitude.

This means empty space in the dataset is rendered at 95% volume. When this constant wall of sound is passed into the synthesizer's "tight" ADSR envelope, it results in a massive strike at the beginning of every single 200ms row, causing the unlistenable "tak tak tak" artifact.

## 6. RECOMMENDED FIXES (in priority order)
1. **Fix Gain Normalization (`apply_global_gain` in `sonify/mapping.py`)**:
   - The double-logging needs to be removed. If the data is already scaled to log by `scale_values`, it should not be logged again. 
   - The min-shift logic needs to be rewritten. A true zero in the dataset must remain an amplitude of `0.0`. The shift should preserve the zero-floor rather than shifting `-10` up to `0` and pushing it through the scaler.
2. **Handle Extreme Sparsity (Optional)**:
   - Given that 60% of the dataset is truly zero, fixing the gain will result in 60% of the audio being completely silent. While this is scientifically accurate, we may need to reconsider whether we want long stretches of pure silence in the sonification, or if a slight ambient drone should be present.
