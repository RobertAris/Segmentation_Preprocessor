# Annotate aligned DSA frames in 3D Slicer

Each prepared run contains `registered.seq.nrrd` and a separate `reference.nrrd`.
The sequence has one single-slice volume per selected source frame. The reference
defines the annotation grid. This is a 2D projection mask, not a reconstructed 3D
vessel tree.

Double-click **Start DSA Workflow.command** in the project folder. It opens Slicer
and asks you to choose a prepared run folder containing `manifest.json`. The
chooser starts in your local output directory. Choose an individual run containing
`manifest.json`, not the parent folder.

You can also open Slicer, open **View → Python Interactor**, and paste:

```python
exec(open('/absolute/path/to/checkout/slicer/DSAWorkflow.py').read())
session = show_workflow('/absolute/path/to/prepared/run')
```

Alternatively call `show_workflow()` to choose the run folder. The folder must
contain `manifest.json`; do not choose the original dataset folder.

The helper opens Segment Editor with one ordinary static segmentation and makes
the aligned sequence proxy its source image. The segmentation is deliberately
absent from the sequence browser. Drawing on a later frame adds to the same mask.
Use the small DSA controls window or Slicer's sequence toolbar to browse time.
Source frame numbers are the original, zero-based acquisition indices.
The reference and aligned frames share a fixed display window estimated from the
central 80% of the reference, using its 0.5th–99.5th percentiles. This keeps black
detector borders from washing out the vessel contrast. Original intensities are
preserved; adjust the window manually in Slicer if another range is more useful.

Before drawing, review shared bifurcations both inside and outside the registration
ROI. The reference overlay slider helps compare anatomy visible in both frames.
Disappearing contrast is expected; shared vessel geometry must still agree.
Check the preprocessing report and correct the frame selection, ROI, or manual
landmarks when necessary. A registration score alone cannot confirm alignment.

Preparations using local deformation have manifest schema 2. Their report includes
full-resolution images, global/final comparisons, correspondence support,
confidence, displacement and deformation-grid views. The scalar shifts shown in
these runs describe coarse initialization; the saved field is the complete
transformation. A confidence score is a screening aid, not an anatomical probability.

For these preparations, review the current frame's shared branches and click
**Mark current frame as anatomically reviewed** before painting. The editor is
disabled while that review is pending or the frame has failed numerical checks.
A review decision cannot override failed, unsupported or folded transforms. The
reference also needs explicit review; its identity transform does not show which
branches are visible or establish full-tree completeness.

Record the reference regions whose anatomy and label coverage you reviewed using
**Add reviewed reference region for current frame**, with `[x, y, width, height]`
in full-resolution reference pixels. Regions are intersected with valid source
sampling and correspondence support. You can instead record regions in the local
HTML report, download its review JSON and use **Load anatomical review JSON from
this report**. Decisions are accepted only for the exact matching manifest and
annotation group. The unsupported-correspondence overlay helps inspect regions
that the registration could not substantiate.

Draw vessels on whichever aligned frame shows them best. A branch disappearing
on another frame is not a reason to erase its existing annotation. Keep the
**Vessels** segment selected, and add newly visible branches to it. Use
**Mark current frame as used / reviewed** to record contributing review frames.

**Save annotation snapshot** writes a new timestamped folder under the prepared
run's `annotations/` directory:

- `vessels_desc-multiframe_mask.nrrd`: binary 0/1 union of all segments, single
  slice, full reference spacing, origin, direction, and dimensions.
- `vessels_desc-multiframe.seg.nrrd`: editable Slicer segmentation with segment names.
- `annotation.json`: reference frame, preparation-manifest fingerprint, available
  frames, frames viewed/present during changes, explicitly marked frames, and notes.

Schema-2 snapshots additionally contain:

- `vessels_desc-reviewedcoverage_mask.nrrd`: pixels inside explicitly reviewed
  reference regions with source/correspondence support.
- `vessels_desc-unknown_mask.nrrd`: 1 outside that reviewed coverage. These pixels
  must be ignored as negative training labels.
- Extended `annotation.json`: method/group, contributing deformation hashes,
  anatomical decisions/regions and coverage semantics. Full-tree completeness is
  explicitly not inferred from image registration or from a reference frame.

An export is blocked if changes were made using an unreviewed/ineligible frame or
if positive label pixels lie outside the reviewed coverage. Review the required
regions or correct the preparation rather than interpreting unobserved anatomy as
background. Existing schema-1 preparations retain their translation workflow and
three-file snapshot format.

The exported mask is one training label for the whole sequence, in registered
reference coordinates. Keep these derivative files separate from source images.
Event tracking does not identify which frame established each individual pixel.

You can also export and navigate through Python:

```python
session.select_frame(8)  # original source index, must be in the prepared sequence
session.mark_current_frame()
files = session.export(notes='Shared bifurcations reviewed on frames 3, 8, 11.')
```

For a schema-2 preparation, also record the anatomical review and coverage:

```python
session.select_frame(8)
session.mark_current_frame(regions=[[150, 200, 400, 350]],
                           notes='Shared branch positions and label coverage checked.')
# Or import decisions downloaded from this exact preparation's report:
session.load_review_decisions('/absolute/path/to/downloaded-review.json')
session.show_correspondence_support(True)
```

The HTML report can also collect independent correspondences on the original
reference and original moving images. Download that landmark JSON and evaluate it
with `dsa-preprocess evaluate PREPARED --landmarks FILE -o RESULT.json`. Keep
evaluation points withheld from fitting. Mark missing/ambiguous visibility instead
of inventing coordinates; fine branches need placement repeatability assessment.

Resume an exported annotation with:

```python
session = load_prepared('/absolute/path/to/prepared/run',
    segmentation_file='/absolute/path/to/snapshot/vessels_desc-multiframe.seg.nrrd')
session.show_controls()
```

Keep `annotation.json` beside the `.seg.nrrd`; it restores the source-frame
provenance when resuming. For a portable full Slicer scene, also use **File → Save
→ Scene (.mrb)**. Keep the prepared run and original manifest for provenance.

The implementation uses Slicer's documented [sequence proxy model](https://slicer.readthedocs.io/en/latest/user_guide/modules/sequences.html)
and [Segment Editor source-volume and fixed-geometry behavior](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html).
