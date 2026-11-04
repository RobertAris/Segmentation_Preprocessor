# DSA segmentation preprocessor

Align a 2D DSA sequence to one reference image, then trace vessels from several registered frames into **one static training mask per sequence** in 3D Slicer. Different contrast-filling stages contribute to the same vessel mask.

The pipeline estimates in-plane translations, saves the transforms, resamples the original intensities once, and produces an alignment review. Filtering is confined to the registration copies. It does not segment vessels automatically or establish that translation is sufficient for a given acquisition.

## Install

Run these commands from this project directory:

```sh
uv venv
source .venv/bin/activate
uv pip install -e .
```

If `uv` is unavailable, use `python3 -m venv .venv`, activate it, and run `python -m pip install -e .`.

## Start with the prepared examples

Three acquisitions have been prepared with explicit vessel regions for motion estimation:

- [sub-h002, run 1: alignment review](outputs/reviewed-roi/sub-h002_acq-unknown_run-1/report.html), using reference frame 15 and frames 3–24.
- [sub-h003, run 2: alignment review](outputs/reviewed-roi/sub-h003_acq-unknown_run-2/report.html).
- [sub-r006, hepatic run 2: alignment review](outputs/reviewed-roi/sub-r006_acq-hepatic_run-2/report.html), using 15 selected frames and reference frame 90.

Open a report in your browser to inspect alignment and adjust the next configuration. The editable starting configurations are in [configs/](configs/).

To start tracing, double-click **Start DSA Workflow.command** in this project folder and select one of the individual prepared run directories under `outputs/reviewed-roi/`. See the [Slicer instructions](slicer/README.md) for annotation and training-mask export. The example alignments still need review of the vessel branches you intend to label.

## Prepare one sequence

Inspect the existing organized dataset. This reads NRRD headers and sidecar metadata and leaves the source files unchanged:

```sh
dsa-preprocess inspect "$HOME/Desktop/Liver/Data/Liver/liver-dsa-organized" --subjects sub-h003
```

Create a configuration and initial review for one short acquisition:

```sh
dsa-preprocess configure \
  "$HOME/Desktop/Liver/Data/Liver/liver-dsa-organized/sub-h003/xa/sub-h003_acq-unknown_run-2_angio.nrrd" \
  -o configs/my-first-run.json

dsa-preprocess prepare \
  "$HOME/Desktop/Liver/Data/Liver/liver-dsa-organized/sub-h003/xa/sub-h003_acq-unknown_run-2_angio.nrrd" \
  --config configs/my-first-run.json \
  -o outputs/my-first-run

open outputs/my-first-run/report.html
```

The proposed reference frame is a starting point. In the report, find a frame with sharp shared vessels, choose complementary contrast-filled frames with the checkboxes, and use **Draw registration ROI** to outline branches visible in both. Select **Use displayed frame as next reference** when appropriate, then **Download updated JSON**. Prepare a **new** output directory with that configuration. Existing outputs are protected from accidental replacement.

The ROI matters: broad image regions can be dominated by stationary background or a catheter while the vessel tree moves. A high score in such a region can conceal vessel misalignment. Choose a vessel-containing region and check bifurcations throughout the image after registration. Report controls save the next configuration; they update the alignment only after you run `prepare` again.

All frame numbers refer to the original acquisition and start at **0**. Command-line overrides are available for quick refinement:

```sh
dsa-preprocess prepare \
  "$HOME/Desktop/Liver/Data/Liver/liver-dsa-organized/sub-h003/xa/sub-h003_acq-unknown_run-2_angio.nrrd" \
  --reference-frame 3 --frames 2-6 --roi 140 130 700 700 \
  -o outputs/sub-h003_acq-unknown_run-2-refined
```

These frame and ROI values demonstrate the syntax; choose them after examining the acquisition. The ROI is `[x, y, width, height]` in **full-resolution source pixels**, with x increasing right and y increasing down. Borders and text should fall outside the ROI.

## Review and correct alignment

The local HTML report provides original and registered frames, before/after overlays against the fixed reference, saved shifts, correlation scores, and automatically flagged frames. No image data are uploaded. Overlays show the reference in magenta and the selected frame in green; correctly aligned shared structures appear gray. Orange marks invalid pixels exposed by a shift. All previews share a display window estimated from the central 80% of the images so black detector borders do not wash out vessel contrast; saved image intensities are unchanged.

Check several bifurcations across the image, including branches outside the estimation ROI. A high correlation score is not proof of correct anatomy. Residual alignment error should be small relative to the narrowest vessel being traced. Rotation or different motion in different regions requires a more suitable registration model; this implementation deliberately provides translation only.

For difficult frames, edit the JSON configuration. For example:

```json
{
  "reference_frame": 3,
  "frames": [2, 3, 4, 5, 6],
  "roi": [140, 130, 700, 700],
  "max_shift_px": 32,
  "upsample_factor": 10,
  "normalization": "none",
  "polarity": "dark",
  "min_correlation": 0.35,
  "manual_shifts": {
    "5": [-1.5, 2.0]
  }
}
```

The numbers above are illustrative. A manual shift is `[dy, dx]`, applied to the moving frame to bring it into the reference grid: positive dx moves image content right, positive dy moves it down. Values can be fractional. Manual corrections still require visual review.

Alternatively, supply corresponding landmarks in original full-resolution pixel coordinates:

```json
{
  "reference_frame": 3,
  "frames": [2, 3, 4],
  "landmarks": {
    "4": {
      "reference": [[400, 320], [520, 450], [610, 510]],
      "moving": [[398, 322], [518, 452], [608, 512]]
    }
  }
}
```

Each point is `[x, y]`. Use the same bifurcations in the same order. Disagreement between landmark displacement vectors indicates that one translation may be insufficient. Keep the reference frame in the selected frame list.

## Annotate in 3D Slicer

Use the supplied [Slicer workflow](slicer/README.md) to load a prepared directory. It creates a sequence browser, an independent reference image, and one ordinary Segmentation node with fixed reference geometry. Browse time while painting additional branches into that same segmentation.

The registered sequence is a real `.seq.nrrd`: its frames are **time**, not a stack of anatomical slices. The source NRRDs in this dataset use their third axis for time; the preprocessor explicitly converts them into single-slice volumes plus a sequence axis.

Start with the clearest frame for a branch. A branch fading in a later frame is not a reason to erase it from the complete mask. Review junctions and branch widths against the frames where they are sharpest. Save the segmentation and export a binary mask through the Slicer helper; the workflow records the reference and frame provenance.

## Prepared files

| File | Purpose |
| --- | --- |
| `registered.seq.nrrd` | Registered original intensities, original frame indices retained |
| `reference.nrrd` | Single-slice image defining the annotation geometry |
| `validity.seq.nrrd` | Per-frame valid-pixel masks after resampling |
| `minip.nrrd` | Overview from eligible registered frames, excluding padded pixels |
| `minip_validity.nrrd` | Pixels supported by at least one eligible projection frame |
| `manifest.json` | Source, configuration, per-frame shifts, quality flags, and geometry |
| `config.json`, `transforms.tsv` | Reusable configuration and tabular per-frame transforms |
| `report.html`, `review/` | Portable local alignment review and PNG previews |

The MinIP is optional context. Individual aligned frames remain the primary annotation source. Automatic flags screen candidate alignments; every run needs anatomical review before labeling.

## Process a collection

After testing representative runs, prepare separate outputs for each acquisition:

```sh
dsa-preprocess batch "$HOME/Desktop/Liver/Data/Liver/liver-dsa-organized" \
  --subjects sub-h003 sub-h004 --limit 3 \
  --config-dir configs -o outputs/pilot-batch
```

Configurations are named `<run_id>.json`, where `run_id` is printed by `inspect`. Missing per-run configurations use automatic defaults. Omit `--limit` only when ready to process all matching runs. Each run has an independent output folder; `batch_summary.json` records failures and the command returns a nonzero status if any run fails. A failed run does not prevent the remaining runs from being processed. A new batch output directory is required.

## Dataset-specific handling and limits

- Only organized `*_angio.nrrd` sequences are discovered; MinIP derivatives and existing masks are excluded.
- Known Cios summary images at the end of rat runs are excluded by default. They are scanner composites, not additional exposures.
- Single-frame runs can be prepared, but cannot benefit from temporal alignment.
- Original spacing, in-plane directions, and origin define the output plane. Missing real pixel spacing remains a measurement limitation; a header value of 1 mm may be a placeholder.
- Inter-frame correction reduces movement between frames. It cannot recover a sharp vessel boundary from a frame that is itself blurred.
- New contrast-filled branches and subtraction artifacts can bias correlation. Select shared vessels and review the result.

The output is one complete vessel training label in the reference frame's image plane. Retain its provenance alongside the exported mask: it identifies the original reference frame and the registered frames available during annotation. This project writes preprocessing outputs to the directory you supply and leaves the source dataset unchanged.

The sequence layout follows [3D Slicer's sequence format](https://slicer.readthedocs.io/en/latest/user_guide/modules/sequences.html); translation estimation uses [scikit-image phase cross-correlation](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation).
