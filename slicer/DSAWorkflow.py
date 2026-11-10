"""One fixed vessel segmentation while browsing a prepared DSA sequence.

Run inside 3D Slicer's Python console (not system Python)::

    exec(open('/path/to/slicer/DSAWorkflow.py').read())
    session = load_prepared('/path/to/prepared/run')
    session.show_controls()

The script does not alter source images or create a segmentation sequence.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import qt
import slicer
import vtk


_sessions = []  # Keep observers and nonmodal control windows alive.


def _matrix(volume):
    matrix = vtk.vtkMatrix4x4()
    volume.GetIJKToRASMatrix(matrix)
    return slicer.util.arrayFromVTKMatrix(matrix)


def _robust_display_window(reference):
    """Estimate one shared display range without letting detector borders dominate."""
    image = np.asarray(reference)[0]
    height, width = image.shape
    top, left = int(height * 0.1), int(width * 0.1)
    interior = image[top:height - top or None, left:width - left or None]
    finite = interior[np.isfinite(interior)]
    if not finite.size:
        raise ValueError("Reference image has no finite pixels for display windowing")
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    window = max(float(high - low), 1.0)
    level = float((low + high) / 2.0)
    return window, level


def _frame_review_problem(frame, schema_version):
    """Return a hard failure; anatomical review cannot override numerical failure."""
    status = frame.get("status")
    if schema_version == 1:
        return "Unaligned frame" if status in {"rejected", "uninformative"} else None
    if frame.get("registration_valid") is not True:
        return "Registration has no valid numerical/correspondence candidate"
    if status in {"failed", "unsupported", "rejected", "uninformative"}:
        return "Registration status is " + str(status)
    qa = frame.get("qa", {})
    for key in ("folded_cell_fraction", "folded_fraction", "fold_count"):
        value = qa.get(key)
        if value is not None and (not np.isfinite(value) or value > 0):
            return "Registration contains a detected fold"
    for key in ("min_jacobian", "min_cell_jacobian"):
        value = qa.get(key)
        if value is not None and (not np.isfinite(value) or value <= 0):
            return "Registration has a nonpositive or invalid Jacobian"
    return None


def _validated_regions(regions, shape):
    height, width = shape
    if not isinstance(regions, list):
        raise ValueError("Reviewed regions must be a list of reference [x,y,width,height] rectangles")
    for region in regions:
        if not isinstance(region, list) or len(region) != 4 or any(type(v) is not int for v in region):
            raise ValueError("Reviewed regions require integer [x,y,width,height] rectangles")
        x,y,w,h = region
        if min(x,y) < 0 or min(w,h) <= 0 or x+w > width or y+h > height:
            raise ValueError("Reviewed region lies outside the reference image")
    return [list(region) for region in regions]


def _validated_review_record(record, manifest, fingerprint):
    """Validate a downloaded review file against the exact immutable preparation."""
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Unsupported anatomical review record")
    if record.get("prepared_manifest_sha256") != fingerprint:
        raise ValueError("Review decisions belong to a different preparation manifest")
    if record.get("group_id") != manifest.get("group_id"):
        raise ValueError("Review decisions belong to a different annotation group")
    frames = {int(f["source_frame"]): f for f in manifest["frames"]}
    decisions = record.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Review decisions must be a list")
    result = {}
    for decision in decisions:
        if not isinstance(decision, dict) or type(decision.get("source_frame")) is not int:
            raise ValueError("Review source_frame must be an original integer index")
        frame = decision["source_frame"]
        if frame not in frames or frame in result:
            raise ValueError("Unknown or duplicate source frame in review decisions")
        value = decision.get("decision")
        if value not in {"pending", "accepted", "rejected"}:
            raise ValueError("Unknown anatomical review decision")
        regions = _validated_regions(decision.get("regions", []), manifest["spatial_shape"])
        if value != "accepted" and regions:
            raise ValueError("Only accepted review decisions may declare reviewed regions")
        if value == "accepted":
            problem = _frame_review_problem(frames[frame], manifest["schema_version"])
            if problem:
                raise ValueError("Cannot approve source frame %d: %s" % (frame, problem))
        result[frame] = {"source_frame":frame, "decision":value, "regions":regions,
                         "notes":str(decision.get("notes", ""))}
    return result


def _reviewed_coverage(shape, reviewed_regions, correspondence_by_frame):
    """Coverage needs an explicit region and source correspondence; identity is insufficient."""
    coverage = np.zeros(shape, dtype=bool)
    for frame, regions in reviewed_regions.items():
        regions = _validated_regions(regions, shape)
        if frame not in correspondence_by_frame:
            raise ValueError("Reviewed region has no matching correspondence support")
        support = np.asarray(correspondence_by_frame[frame], dtype=bool)
        if support.shape != tuple(shape):
            raise ValueError("Correspondence support does not match reference dimensions")
        for x,y,w,h in regions:
            coverage[y:y+h,x:x+w] |= support[y:y+h,x:x+w]
    return coverage


def _reviewed_contributors(manifest, edited_frames, reviewed_frames):
    """Check the contribution ledger before any annotation files are written."""
    frames = {int(frame["source_frame"]):frame for frame in manifest["frames"]}
    edited,reviewed = set(edited_frames),set(reviewed_frames)
    if (edited | reviewed) - set(frames):
        raise ValueError("Annotation provenance contains an unknown source frame")
    schema = manifest["schema_version"]
    bad = sorted(frame for frame in edited if _frame_review_problem(frames[frame],schema))
    if bad:
        raise RuntimeError("Segmentation changes occurred on ineligible source frames %s" % bad)
    if schema == 2:
        pending = sorted(edited - reviewed)
        if pending:
            raise RuntimeError("Explicit anatomical review is required for edited source frames %s" % pending)
        bad = sorted(frame for frame in reviewed if _frame_review_problem(frames[frame],schema))
        if bad:
            raise RuntimeError("Anatomical review cannot override failed source frames %s" % bad)
    return sorted(edited | reviewed)


class DSASession:
    """A prepared sequence, reference grid, and one ordinary segmentation node."""

    def __init__(self, directory, segmentation_file=None):
        self.directory = Path(directory).expanduser().resolve()
        manifest_file = self.directory / "manifest.json"
        self.manifest = json.loads(manifest_file.read_text())
        if self.manifest.get("schema_version") not in {1, 2}:
            raise ValueError("Unsupported or missing manifest schema_version")
        self.manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        self.edited_source_frames = set()
        self.marked_source_frames = set()
        self.viewed_source_frames = set()
        self.reviewed_regions = {}
        self.review_notes = {}
        self.rejected_source_frames = set()
        self.imported_review_records = []
        self.correspondence_by_frame = {}
        self.support_overlay = None
        self._show_support = False
        self.controls = None
        self._updating_controls = False
        self._initializing = True
        self.nodes = []
        try:
            self._load(segmentation_file)
        except Exception:
            for node in reversed(self.nodes):
                if node and node.GetScene():
                    slicer.mrmlScene.RemoveNode(node)
            raise
        finally:
            self._initializing = False

    def _file(self, key):
        filename = self.directory / self.manifest[key]
        if not filename.is_file():
            raise FileNotFoundError(filename)
        return str(filename)

    def _load(self, segmentation_file):
        name = self.directory.name
        self.reference = slicer.util.loadVolume(
            self._file("reference_file"), {"name": name + " reference (fixed grid)", "show": False})
        self.nodes.append(self.reference)
        self.sequence = slicer.util.loadSequence(
            self._file("sequence_file"), {"name": name + " registered frames", "show": False})
        self.nodes.append(self.sequence)
        if not self.sequence or self.sequence.GetNumberOfDataNodes() == 0:
            raise ValueError("The prepared sequence contains no frames")

        reference_shape = slicer.util.arrayFromVolume(self.reference).shape
        if reference_shape[0] != 1:
            raise ValueError("Expected a single-slice reference, not a 3D anatomy volume")
        reference_matrix = _matrix(self.reference)
        self._reference_matrix = reference_matrix.copy()
        self._reference_shape = tuple(reference_shape)
        for item in range(self.sequence.GetNumberOfDataNodes()):
            volume = self.sequence.GetNthDataNode(item)
            if slicer.util.arrayFromVolume(volume).shape != reference_shape:
                raise ValueError("Sequence frame dimensions differ from the fixed reference")
            if not np.allclose(_matrix(volume), reference_matrix, atol=1e-5):
                raise ValueError("Sequence frame geometry differs from the fixed reference")

        self.source_frames = [int(float(self.sequence.GetNthIndexValue(i)))
                              for i in range(self.sequence.GetNumberOfDataNodes())]
        self.frame_info = {int(frame["source_frame"]): frame
                           for frame in self.manifest.get("frames", [])}
        if self.manifest["schema_version"] == 2:
            expected = [int(frame["source_frame"]) for frame in self.manifest["frames"]]
            if expected != self.source_frames or len(set(expected)) != len(expected):
                raise ValueError("Sequence indices do not match the preparation manifest")
            self._load_dense_support(reference_shape, reference_matrix)

        logic = slicer.modules.sequences.logic()
        self.browser = logic.GetFirstBrowserNodeForSequenceNode(self.sequence)
        if not self.browser:
            self.browser = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSequenceBrowserNode", name + " DSA browser")
            self.nodes.append(self.browser)
            self.browser.SetAndObserveMasterSequenceNodeID(self.sequence.GetID())
        else:
            self.nodes.append(self.browser)
        self.browser.SetSaveChanges(self.sequence, False)
        self.browser.SetRecordingActive(False)
        self.proxy = self.browser.GetProxyNode(self.sequence)
        if self.proxy is None:
            logic.UpdateAllProxyNodes()
            self.proxy = self.browser.GetProxyNode(self.sequence)
        if self.proxy is None:
            raise RuntimeError("Slicer could not create the displayed sequence proxy")
        self.nodes.append(self.proxy)
        self.proxy.SetName(name + " aligned DSA (browse frames)")
        self.display_window, self.display_level = _robust_display_window(
            slicer.util.arrayFromVolume(self.reference))
        # Keep one window across time and for reference overlays. In particular,
        # black detector borders must not spread near-32768 DSA values over an
        # almost entirely white display. This changes display properties only.
        for volume in (self.reference, self.proxy):
            volume.CreateDefaultDisplayNodes()
            volume.GetDisplayNode().AutoWindowLevelOff()
            volume.GetDisplayNode().SetWindowLevel(self.display_window, self.display_level)

        if segmentation_file:
            segmentation_path = Path(segmentation_file).expanduser().resolve()
            sidecar = segmentation_path.parent / "annotation.json"
            if not sidecar.is_file():
                raise ValueError("Resuming requires the annotation.json saved beside the .seg.nrrd")
            previous = json.loads(sidecar.read_text())
            if previous.get("prepared_manifest_sha256") != self.manifest_sha256:
                raise ValueError("Saved segmentation belongs to a different prepared manifest")
            if "shape_kji" in previous and list(reference_shape) != previous["shape_kji"]:
                raise ValueError("Saved annotation dimensions differ from the fixed reference")
            if "ijk_to_ras" in previous and not np.allclose(np.asarray(previous["ijk_to_ras"]),reference_matrix,atol=1e-5):
                raise ValueError("Saved annotation geometry differs from the fixed reference")
            self.edited_source_frames.update(previous.get("frames_present_during_segmentation_changes", []))
            self.marked_source_frames.update(previous.get("explicitly_marked_source_frames", []))
            self.viewed_source_frames.update(previous.get("viewed_source_frames", []))
            if self.manifest["schema_version"] == 2:
                self.marked_source_frames.clear()
                self._apply_review_record({"schema_version":1,
                    "prepared_manifest_sha256":self.manifest_sha256,
                    "group_id":self.manifest.get("group_id"),
                    "decisions":previous.get("anatomical_review_decisions",[])})
                self.imported_review_records = previous.get("imported_review_records",[])
            self.segmentation = slicer.util.loadSegmentation(str(segmentation_path))
            if self.segmentation.GetParentTransformNode():
                raise ValueError("Saved segmentation has an active transform; resolve its geometry before resuming")
            stored_fingerprint = self.segmentation.GetAttribute("DSAWorkflow.ManifestSHA256")
            if stored_fingerprint and stored_fingerprint != self.manifest_sha256:
                raise ValueError("Segmentation node belongs to a different prepared manifest")
        else:
            self.segmentation = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", name + " complete vessels (static)")
            self.segmentation.CreateDefaultDisplayNodes()
            self.segmentation.GetSegmentation().AddEmptySegment("vessels", "Vessels", (1.0, 0.25, 0.15))
        self.nodes.append(self.segmentation)
        self.segmentation.SetReferenceImageGeometryParameterFromVolumeNode(self.reference)
        self.segmentation.SetAttribute("DSAWorkflow.ManifestSHA256", self.manifest_sha256)
        self.segmentation.SetAttribute("DSAWorkflow.PreparedDirectory", str(self.directory))
        self.segmentation.GetDisplayNode().SetOpacity2DFill(0.15)
        self.segmentation.GetDisplayNode().SetOpacity2DOutline(0.9)

        self.browser.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_frame_change)
        self.segmentation.GetSegmentation().AddObserver(
            slicer.vtkSegmentation.SegmentModified, self._on_segment_change)
        self.segmentation.GetSegmentation().AddObserver(
            slicer.vtkSegmentation.SegmentAdded, self._on_segment_change)
        self.segmentation.GetSegmentation().AddObserver(
            slicer.vtkSegmentation.SegmentRemoved, self._on_segment_change)

        self.editor = None
        if slicer.util.mainWindow():
            layout = slicer.app.layoutManager()
            layout.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
            red = layout.sliceWidget("Red")
            red.sliceLogic().GetSliceNode().SetOrientationToAxial()
            slicer.util.setSliceViewerLayers(background=self.proxy, foreground=None)
            slicer.util.resetSliceViews()
            slicer.modules.sequences.toolBar().setActiveBrowserNode(self.browser)
            slicer.modules.sequences.toolBar().show()
            slicer.util.selectModule("SegmentEditor")
            editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
            editor.setSegmentationNode(self.segmentation)
            editor.setSourceVolumeNode(self.proxy)
            editor.setCurrentSegmentID(self.segmentation.GetSegmentation().GetNthSegmentID(0))
            self.editor = editor
        reference_index = self.manifest.get("reference_frame")
        if reference_index in self.source_frames:
            self.browser.SetSelectedItemNumber(self.source_frames.index(reference_index))
        self._on_frame_change()

    def _load_dense_support(self, reference_shape, reference_matrix):
        """Validate support geometry and immutable deformation fingerprints before review."""
        mask_arrays = {}
        for key in ("validity_file", "correspondence_file", "confidence_file"):
            sequence = slicer.util.loadSequence(self._file(key), {"show":False})
            self.nodes.append(sequence)
            if sequence is None or sequence.GetNumberOfDataNodes() != len(self.source_frames):
                raise ValueError("Support sequence length differs from the image sequence")
            values = []
            for item,source in enumerate(self.source_frames):
                if int(float(sequence.GetNthIndexValue(item))) != source:
                    raise ValueError("Support sequence indices differ from original source indices")
                volume = sequence.GetNthDataNode(item)
                array = np.array(slicer.util.arrayFromVolume(volume), copy=True)
                if array.shape != reference_shape or not np.allclose(_matrix(volume),reference_matrix,atol=1e-5):
                    raise ValueError("Support sequence geometry differs from the fixed reference")
                if not np.isfinite(array).all():
                    raise ValueError("Support sequence contains nonfinite values")
                if key != "confidence_file" and not np.isin(array,[0,1]).all():
                    raise ValueError("Validity and correspondence support must be binary")
                if key == "confidence_file" and (np.any(array < 0) or np.any(array > 1)):
                    raise ValueError("Confidence scores must be in [0,1]")
                values.append(array[0])
            mask_arrays[key] = values
        for item,source in enumerate(self.source_frames):
            frame = self.frame_info[source]
            filename = frame.get("field_file")
            fingerprint = frame.get("field_sha256")
            if not filename or not fingerprint:
                raise ValueError("Deformation frame is missing its immutable field record")
            path = self.directory / filename
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
                raise ValueError("Deformation file fingerprint differs from the preparation manifest")
            with np.load(path,allow_pickle=False) as stored:
                field = stored["pull_displacement_yx"]
                if field.shape != (2,*reference_shape[1:]) or not np.isfinite(field).all():
                    raise ValueError("Invalid saved pull-displacement field")
            self.correspondence_by_frame[source] = (mask_arrays["validity_file"][item].astype(bool) &
                                                    mask_arrays["correspondence_file"][item].astype(bool))

    def _apply_review_record(self, record):
        decisions = _validated_review_record(record,self.manifest,self.manifest_sha256)
        for frame,decision in decisions.items():
            self.marked_source_frames.discard(frame)
            self.rejected_source_frames.discard(frame)
            self.reviewed_regions.pop(frame,None)
            self.review_notes[frame] = decision["notes"]
            if decision["decision"] == "accepted":
                self.marked_source_frames.add(frame)
                self.reviewed_regions[frame] = decision["regions"]
            elif decision["decision"] == "rejected":
                self.rejected_source_frames.add(frame)

    def load_review_decisions(self, filename):
        """Load anatomical decisions exported by this preparation's local HTML report."""
        path = Path(filename).expanduser().resolve()
        self._apply_review_record(json.loads(path.read_text()))
        self.imported_review_records.append({"file":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        self._on_frame_change()
        return sorted(self.marked_source_frames)

    @property
    def current_source_frame(self):
        return self.source_frames[self.browser.GetSelectedItemNumber()]

    def select_frame(self, source_frame):
        """Select by original, zero-based acquisition frame number."""
        self.browser.SetSelectedItemNumber(self.source_frames.index(int(source_frame)))

    def _on_frame_change(self, caller=None, event=None):
        if not getattr(self, "source_frames", None):
            return
        frame = self.current_source_frame
        self.viewed_source_frames.add(frame)
        details = self.frame_info.get(frame, {})
        problem = _frame_review_problem(details,self.manifest["schema_version"])
        needs_review = self.manifest["schema_version"] == 2 and frame not in self.marked_source_frames
        if self.manifest["schema_version"] == 2 and getattr(self,"editor",None):
            self.editor.setEnabled(not problem and not needs_review)
        if self._show_support:
            self.show_correspondence_support(True)
        if self.controls:
            self._updating_controls = True
            self.frame_slider.setValue(self.browser.GetSelectedItemNumber())
            status = details.get("status", "review required")
            unsafe = bool(problem)
            self.frame_label.setStyleSheet("color: #b00020; font-weight: bold" if unsafe else "")
            self.frame_label.setText(
                "%sSource frame %d  |  %s  |  %s y=%s, x=%s px" % (
                    "NOT ALIGNED — do not annotate. " if unsafe else ("REVIEW BEFORE ANNOTATING. " if needs_review else ""),
                    frame, status,
                    "coarse shift" if self.manifest["schema_version"] == 2 else "shift",
                    details.get("coarse_shift_y_px",details.get("shift_y_px", "?")),
                    details.get("coarse_shift_x_px",details.get("shift_x_px", "?"))))
            if hasattr(self,"review_label"):
                regions = self.reviewed_regions.get(frame,[])
                self.review_label.setText("Anatomical review: %s; %d reviewed reference region(s). Full-tree completeness is not established." % (
                    "recorded" if frame in self.marked_source_frames else "pending",len(regions)))
            self._updating_controls = False

    def _on_segment_change(self, caller=None, event=None):
        if not self._initializing and not getattr(self,"_exporting",False):
            self.edited_source_frames.add(self.current_source_frame)

    def mark_current_frame(self, regions=None, notes=""):
        """Record explicit anatomical review; regions are reference-grid [x,y,w,h].

        A frame review without regions does not establish known label coverage.
        """
        frame = self.current_source_frame
        problem = _frame_review_problem(self.frame_info.get(frame,{}),self.manifest["schema_version"])
        if problem:
            raise RuntimeError("Cannot mark source frame %d: %s" % (frame,problem))
        if regions is not None:
            checked = _validated_regions(regions,self.manifest["spatial_shape"])
            self.reviewed_regions.setdefault(frame,[]).extend(checked)
        self.marked_source_frames.add(frame)
        self.rejected_source_frames.discard(frame)
        if notes:
            self.review_notes[frame] = str(notes)
        self._on_frame_change()

    def show_correspondence_support(self, enabled=True):
        """Highlight unsupported regions; the overlay is not an anatomical label."""
        self._show_support = bool(enabled)
        if self.manifest["schema_version"] != 2:
            raise RuntimeError("Correspondence support is available only for deformation preparations")
        if not enabled:
            slicer.util.setSliceViewerLayers(background=self.proxy,foreground=None)
            return
        if self.support_overlay is None:
            self.support_overlay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode","DSA unsupported correspondence")
            self.nodes.append(self.support_overlay)
            self.support_overlay.CopyOrientation(self.reference)
            self.support_overlay.CreateDefaultDisplayNodes()
            display = self.support_overlay.GetDisplayNode()
            display.AutoWindowLevelOff()
            display.SetWindowLevel(1.,.5)
            display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeHot")
        unsupported = (~self.correspondence_by_frame[self.current_source_frame]).astype(np.uint8)[None]
        slicer.util.updateVolumeFromArray(self.support_overlay,unsupported)
        slicer.util.setSliceViewerLayers(background=self.proxy,foreground=self.support_overlay,foregroundOpacity=.5)

    def set_reference_overlay(self, opacity=0.0):
        """Overlay the reference on the current aligned frame to review shared vessels."""
        self._show_support = False
        slicer.util.setSliceViewerLayers(background=self.proxy,
                                         foreground=self.reference if opacity else None,
                                         foregroundOpacity=float(opacity))

    def export(self, output_directory=None, notes="", overwrite=False):
        """Save one snapshot without treating representation serialization as painting."""
        previous = getattr(self,"_exporting",False)
        self._exporting = True
        try:
            return self._export_snapshot(output_directory,notes,overwrite)
        finally:
            self._exporting = previous

    def _export_snapshot(self, output_directory=None, notes="", overwrite=False):
        """Save a binary union mask on the reference grid, editable segments, provenance.

        With no output directory, creates a unique timestamped snapshot under
        prepared-run/annotations/. The mask is in registered reference coordinates.
        It is not a label for any unregistered non-reference source frame.
        """
        if self.browser.GetSequenceNode(self.segmentation):
            raise RuntimeError("The segmentation was added to a sequence browser; remove it before export")
        if self.segmentation.GetParentTransformNode():
            raise RuntimeError("Harden or remove the segmentation transform before export")
        if self.reference.GetParentTransformNode() or tuple(slicer.util.arrayFromVolume(self.reference).shape) != self._reference_shape or not np.allclose(_matrix(self.reference),self._reference_matrix,atol=1e-5):
            raise RuntimeError("The fixed reference geometry changed; export cannot change annotation coordinates")
        if hashlib.sha256((self.directory / "manifest.json").read_bytes()).hexdigest() != self.manifest_sha256:
            raise RuntimeError("The preparation manifest changed during annotation; reload the correct preparation")
        contributors = _reviewed_contributors(self.manifest,self.edited_source_frames,self.marked_source_frames)
        schema2 = self.manifest["schema_version"] == 2
        if schema2:
            for frame in contributors:
                info = self.frame_info.get(frame,{})
                problem = _frame_review_problem(info,2)
                if problem:
                    raise RuntimeError("Ineligible contributing source frame %d: %s" % (frame,problem))
                if hashlib.sha256((self.directory / info["field_file"]).read_bytes()).hexdigest() != info["field_sha256"]:
                    raise RuntimeError("A contributing deformation changed during annotation")
        if output_directory is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            output_directory = self.directory / "annotations" / stamp
        destination = Path(output_directory).expanduser().resolve()
        source = Path(self.manifest["source_file"]).expanduser().resolve()
        if destination == source.parent or source.parent in destination.parents:
            raise ValueError("Choose a separate derivative folder outside the source image directory")
        files = {"mask": destination / "vessels_desc-multiframe_mask.nrrd",
                 "segmentation": destination / "vessels_desc-multiframe.seg.nrrd",
                 "provenance": destination / "annotation.json"}
        if schema2:
            files.update(coverage=destination / "vessels_desc-reviewedcoverage_mask.nrrd",
                         unknown=destination / "vessels_desc-unknown_mask.nrrd")
        if not overwrite and any(path.exists() for path in files.values()):
            raise FileExistsError("Annotation files exist; choose a new directory or overwrite=True")

        binary = np.zeros(slicer.util.arrayFromVolume(self.reference).shape, dtype=np.uint8)
        segmentation = self.segmentation.GetSegmentation()
        for index in range(segmentation.GetNumberOfSegments()):
            array = slicer.util.arrayFromSegmentBinaryLabelmap(
                self.segmentation, segmentation.GetNthSegmentID(index), self.reference)
            binary |= (array > 0).astype(np.uint8)
        if schema2:
            coverage = _reviewed_coverage(self._reference_shape[1:],self.reviewed_regions,self.correspondence_by_frame)[None]
            outside = (binary > 0) & ~coverage
            if outside.any():
                raise RuntimeError("The mask contains %d pixels outside reviewed correspondence regions. "
                    "Review and record reference regions covering those branches before export." % int(outside.sum()))
        destination.mkdir(parents=True, exist_ok=True)
        self.segmentation.SetAttribute("DSAWorkflow.EditedSourceFrames", json.dumps(sorted(self.edited_source_frames)))
        self.segmentation.SetAttribute("DSAWorkflow.MarkedSourceFrames", json.dumps(sorted(self.marked_source_frames)))
        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "DSA export")
        try:
            labelmap.CopyOrientation(self.reference)
            slicer.util.updateVolumeFromArray(labelmap, binary)
            if not slicer.util.saveNode(labelmap, str(files["mask"])):
                raise IOError("Could not save binary mask")
            if not slicer.util.saveNode(self.segmentation, str(files["segmentation"])):
                raise IOError("Could not save editable segmentation")
            if schema2:
                for key,array in (("coverage",coverage),("unknown",~coverage)):
                    slicer.util.updateVolumeFromArray(labelmap,array.astype(np.uint8))
                    if not slicer.util.saveNode(labelmap,str(files[key])):
                        raise IOError("Could not save annotation " + key + " mask")
        finally:
            slicer.mrmlScene.RemoveNode(labelmap)
        provenance = {
            "schema_version": 2 if schema2 else 1,
            "annotation_type": "multi_frame_static_vessel_mask",
            "coordinate_system": "registered_reference_grid",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source),
            "prepared_manifest": str(self.directory / "manifest.json"),
            "prepared_manifest_sha256": self.manifest_sha256,
            "reference_frame": self.manifest["reference_frame"],
            "available_source_frames": self.source_frames,
            "viewed_source_frames": sorted(self.viewed_source_frames),
            "frames_present_during_segmentation_changes": sorted(self.edited_source_frames),
            "explicitly_marked_source_frames": sorted(self.marked_source_frames),
            "tracking_note": "Change events identify frames present during edits; they do not prove per-pixel provenance.",
            "shape_kji": list(binary.shape),
            "ijk_to_ras": _matrix(self.reference).tolist(),
            "mask_values": [0, 1],
            "mask_nonzero_voxels": int(np.count_nonzero(binary)),
            "segment_names": [segmentation.GetSegment(segmentation.GetNthSegmentID(index)).GetName()
                              for index in range(segmentation.GetNumberOfSegments())],
            "mask_file": files["mask"].name,
            "segmentation_file": files["segmentation"].name,
            "notes": str(notes),
        }
        if schema2:
            provenance.update(
                group_id=self.manifest.get("group_id"),
                registration_method=self.manifest.get("registration_method"),
                transform_convention=self.manifest.get("transform_convention"),
                contributing_deformations=[{"source_frame":frame,
                    "field_file":self.frame_info[frame]["field_file"],
                    "field_sha256":self.frame_info[frame]["field_sha256"]}
                    for frame in sorted(self.edited_source_frames | self.marked_source_frames)],
                anatomical_review_decisions=[{"source_frame":frame,
                    "decision":"accepted" if frame in self.marked_source_frames else ("rejected" if frame in self.rejected_source_frames else "pending"),
                    "regions":self.reviewed_regions.get(frame,[]),"notes":self.review_notes.get(frame,"")}
                    for frame in self.source_frames],
                imported_review_records=self.imported_review_records,
                reviewed_coverage_file=files["coverage"].name,
                unknown_file=files["unknown"].name,
                coverage_definition="Explicitly reviewed reference rectangles intersected with source and correspondence support. Unknown=1 outside this coverage; unknown pixels must not be used as negative training labels.",
                full_tree_completeness_established=False,
                reviewed_coverage_fraction=float(coverage.mean()),
                unknown_fraction=float((~coverage).mean()),
                file_sha256={key:hashlib.sha256(path.read_bytes()).hexdigest()
                             for key,path in files.items() if key != "provenance"})
        files["provenance"].write_text(json.dumps(provenance, indent=2) + "\n")
        print("Saved multi-frame annotation to " + str(destination))
        return {key: str(value) for key, value in files.items()}

    def show_controls(self):
        """Open a small nonmodal frame/review/export panel beside Segment Editor."""
        if not slicer.util.mainWindow():
            raise RuntimeError("The controls window requires Slicer with its main window enabled")
        if self.controls:
            self.controls.show()
            self.controls.raise_()
            return self.controls
        self.controls = qt.QDialog(slicer.util.mainWindow())
        self.controls.setWindowTitle("DSA — one static vessel segmentation")
        layout = qt.QVBoxLayout(self.controls)
        if self.manifest["schema_version"] == 2:
            scroll = qt.QScrollArea()
            scroll.setWidgetResizable(True)
            content = qt.QWidget()
            scroll.setWidget(content)
            layout.addWidget(scroll)
            layout = qt.QVBoxLayout(content)
        instruction = qt.QLabel(
            "Browse aligned frames and keep adding to the same Vessels segment.\n"
            "Review shared bifurcations before joining branches.\n"
            "Automatic registration scores do not establish anatomical alignment.")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        self.frame_label = qt.QLabel()
        layout.addWidget(self.frame_label)
        self.frame_slider = qt.QSlider(qt.Qt.Horizontal)
        self.frame_slider.setRange(0, len(self.source_frames) - 1)
        self.frame_slider.connect("valueChanged(int)", self._select_item)
        layout.addWidget(self.frame_slider)
        mark = qt.QPushButton("Mark current frame as anatomically reviewed" if self.manifest["schema_version"] == 2 else "Mark current frame as used / reviewed")
        mark.connect("clicked()", self._mark_from_controls)
        layout.addWidget(mark)
        if self.manifest["schema_version"] == 2:
            layout.addWidget(qt.QLabel("Registration method: " + str(self.manifest.get("registration_method","unknown"))))
            self.review_label = qt.QLabel()
            self.review_label.setWordWrap(True)
            layout.addWidget(self.review_label)
            coverage_instruction = qt.QLabel("Before exporting a mask, record the reference regions whose anatomy and label coverage you reviewed. Use [x, y, width, height] in reference pixels. Unreviewed pixels remain unknown.")
            coverage_instruction.setWordWrap(True)
            layout.addWidget(coverage_instruction)
            self.region_edit = qt.QLineEdit()
            self.region_edit.setPlaceholderText("[x, y, width, height]")
            layout.addWidget(self.region_edit)
            add_region = qt.QPushButton("Add reviewed reference region for current frame")
            add_region.connect("clicked()",self._add_region_from_controls)
            layout.addWidget(add_region)
            load_review = qt.QPushButton("Load anatomical review JSON from this report")
            load_review.connect("clicked()",self._load_review_from_controls)
            layout.addWidget(load_review)
            support = qt.QPushButton("Show / hide unsupported correspondence")
            support.connect("clicked()",lambda:self.show_correspondence_support(not self._show_support))
            layout.addWidget(support)
        layout.addWidget(qt.QLabel("Reference overlay (for registration review)"))
        overlay = qt.QSlider(qt.Qt.Horizontal)
        overlay.setRange(0, 100)
        overlay.connect("valueChanged(int)", lambda value: self.set_reference_overlay(value / 100.0))
        layout.addWidget(overlay)
        layout.addWidget(qt.QLabel("Annotation notes (saved with export)"))
        self.notes_edit = qt.QPlainTextEdit()
        self.notes_edit.setMaximumHeight(90)
        layout.addWidget(self.notes_edit)
        save = qt.QPushButton("Save annotation snapshot")
        save.connect("clicked()", self._export_from_controls)
        layout.addWidget(save)
        self.save_label = qt.QLabel("Saves a binary mask, editable segmentation, reviewed coverage, unknown mask, and provenance." if self.manifest["schema_version"] == 2 else "Saves a binary mask, editable segmentation, and provenance.")
        self.save_label.setWordWrap(True)
        layout.addWidget(self.save_label)
        self.controls.resize(620, 700 if self.manifest["schema_version"] == 2 else 380)
        self._on_frame_change()
        self.controls.show()
        return self.controls

    def _mark_from_controls(self):
        try:
            self.mark_current_frame()
        except Exception as error:
            slicer.util.errorDisplay(str(error))

    def _add_region_from_controls(self):
        try:
            self.mark_current_frame(regions=[json.loads(str(self.region_edit.text))])
        except Exception as error:
            slicer.util.errorDisplay(str(error))

    def _load_review_from_controls(self):
        try:
            filename = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"Choose anatomical review decisions","","JSON files (*.json)")
            if isinstance(filename,(list,tuple)):
                filename = filename[0]
            if filename:
                self.load_review_decisions(filename)
        except Exception as error:
            slicer.util.errorDisplay(str(error))

    def _select_item(self, item):
        if not self._updating_controls:
            self.browser.SetSelectedItemNumber(item)

    def _export_from_controls(self):
        try:
            result = self.export(notes=self.notes_edit.toPlainText())
            self.save_label.setText("Saved: " + str(Path(result["mask"]).parent))
        except Exception as error:
            slicer.util.errorDisplay(str(error))


def load_prepared(directory, segmentation_file=None):
    """Load a prepared run without clearing existing Slicer data.

    To resume, pass the .seg.nrrd from a previous annotation snapshot as
    segmentation_file. Keep the returned session to navigate or export.
    """
    session = DSASession(directory, segmentation_file=segmentation_file)
    _sessions.append(session)
    return session


def show_workflow(directory=None):
    """Choose a prepared run and open annotation controls."""
    if directory is None:
        directory = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(), "Choose prepared DSA run containing manifest.json")
    if not directory:
        return None
    session = load_prepared(directory)
    session.show_controls()
    return session
