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


class DSASession:
    """A prepared sequence, reference grid, and one ordinary segmentation node."""

    def __init__(self, directory, segmentation_file=None):
        self.directory = Path(directory).expanduser().resolve()
        manifest_file = self.directory / "manifest.json"
        self.manifest = json.loads(manifest_file.read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported or missing manifest schema_version")
        self.manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        self.edited_source_frames = set()
        self.marked_source_frames = set()
        self.viewed_source_frames = set()
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
        for item in range(self.sequence.GetNumberOfDataNodes()):
            volume = self.sequence.GetNthDataNode(item)
            if slicer.util.arrayFromVolume(volume).shape != reference_shape:
                raise ValueError("Sequence frame dimensions differ from the fixed reference")
            if not np.allclose(_matrix(volume), reference_matrix, atol=1e-5):
                raise ValueError("Sequence frame geometry differs from the fixed reference")

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
            self.edited_source_frames.update(previous.get("frames_present_during_segmentation_changes", []))
            self.marked_source_frames.update(previous.get("explicitly_marked_source_frames", []))
            self.viewed_source_frames.update(previous.get("viewed_source_frames", []))
            self.segmentation = slicer.util.loadSegmentation(str(segmentation_path))
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

        self.source_frames = [int(float(self.sequence.GetNthIndexValue(i)))
                              for i in range(self.sequence.GetNumberOfDataNodes())]
        self.frame_info = {int(frame["source_frame"]): frame
                           for frame in self.manifest.get("frames", [])}
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
        if self.controls:
            self._updating_controls = True
            self.frame_slider.setValue(self.browser.GetSelectedItemNumber())
            details = self.frame_info.get(frame, {})
            status = details.get("status", "review required")
            unsafe = status in {"rejected", "uninformative"}
            self.frame_label.setStyleSheet("color: #b00020; font-weight: bold" if unsafe else "")
            self.frame_label.setText(
                "%sSource frame %d  |  %s  |  shift y=%s, x=%s px" % (
                    "NOT ALIGNED — do not annotate. " if unsafe else "",
                    frame, status,
                    details.get("shift_y_px", "?"), details.get("shift_x_px", "?")))
            self._updating_controls = False

    def _on_segment_change(self, caller=None, event=None):
        if not self._initializing:
            self.edited_source_frames.add(self.current_source_frame)

    def mark_current_frame(self):
        """Explicitly record a frame used to define/review vessel anatomy."""
        self.marked_source_frames.add(self.current_source_frame)

    def set_reference_overlay(self, opacity=0.0):
        """Overlay the reference on the current aligned frame to review shared vessels."""
        slicer.util.setSliceViewerLayers(background=self.proxy,
                                         foreground=self.reference if opacity else None,
                                         foregroundOpacity=float(opacity))

    def export(self, output_directory=None, notes="", overwrite=False):
        """Save a binary union mask on the reference grid, editable segments, provenance.

        With no output directory, creates a unique timestamped snapshot under
        prepared-run/annotations/. The mask is in registered reference coordinates.
        It is not a label for any unregistered non-reference source frame.
        """
        if self.browser.GetSequenceNode(self.segmentation):
            raise RuntimeError("The segmentation was added to a sequence browser; remove it before export")
        if self.segmentation.GetParentTransformNode():
            raise RuntimeError("Harden or remove the segmentation transform before export")
        unsafe_edits = [frame for frame in self.edited_source_frames
                        if self.frame_info.get(frame, {}).get("status") in {"rejected", "uninformative"}]
        if unsafe_edits:
            raise RuntimeError("Segmentation changes occurred on unaligned frames %s. Correct their alignment "
                               "and re-prepare the run before using these edits." % sorted(unsafe_edits))
        if output_directory is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            output_directory = self.directory / "annotations" / stamp
        destination = Path(output_directory).expanduser().resolve()
        source = Path(self.manifest["source_file"]).expanduser().resolve()
        if destination == source.parent or source.parent in destination.parents:
            raise ValueError("Choose a separate derivative folder outside the source image directory")
        destination.mkdir(parents=True, exist_ok=True)
        files = {"mask": destination / "vessels_desc-multiframe_mask.nrrd",
                 "segmentation": destination / "vessels_desc-multiframe.seg.nrrd",
                 "provenance": destination / "annotation.json"}
        if not overwrite and any(path.exists() for path in files.values()):
            raise FileExistsError("Annotation files exist; choose a new directory or overwrite=True")

        binary = np.zeros(slicer.util.arrayFromVolume(self.reference).shape, dtype=np.uint8)
        segmentation = self.segmentation.GetSegmentation()
        for index in range(segmentation.GetNumberOfSegments()):
            array = slicer.util.arrayFromSegmentBinaryLabelmap(
                self.segmentation, segmentation.GetNthSegmentID(index), self.reference)
            binary |= (array > 0).astype(np.uint8)
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
        finally:
            slicer.mrmlScene.RemoveNode(labelmap)
        provenance = {
            "schema_version": 1,
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
        mark = qt.QPushButton("Mark current frame as used / reviewed")
        mark.connect("clicked()", self.mark_current_frame)
        layout.addWidget(mark)
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
        self.save_label = qt.QLabel("Saves a binary mask, editable segmentation, and provenance.")
        self.save_label.setWordWrap(True)
        layout.addWidget(self.save_label)
        self.controls.resize(570, 380)
        self._on_frame_change()
        self.controls.show()
        return self.controls

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
