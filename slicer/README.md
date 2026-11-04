# Annotate aligned DSA frames in 3D Slicer

Each prepared run contains `registered.seq.nrrd` and a separate `reference.nrrd`.
The sequence has one single-slice volume per selected source frame. The reference
defines the annotation grid. This is a 2D projection mask, not a reconstructed 3D
vessel tree.

Double-click **Start DSA Workflow.command** in the project folder. It opens Slicer
and asks you to choose a prepared run folder containing `manifest.json`. The
chooser starts in `outputs/reviewed-roi/`. Choose an individual run, not the parent folder.

You can also open Slicer, open **View → Python Interactor**, and paste:

```python
exec(open('/Users/robertjahnke/Segmentation_Preprocessor/slicer/DSAWorkflow.py').read())
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

The exported mask is one training label for the whole sequence, in registered
reference coordinates. Keep these derivative files separate from source images.
Event tracking does not identify which frame established each individual pixel.

You can also export and navigate through Python:

```python
session.select_frame(8)  # original source index, must be in the prepared sequence
session.mark_current_frame()
files = session.export(notes='Shared bifurcations reviewed on frames 3, 8, 11.')
```

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
