"""Portable, local-only registration review with linked image previews."""
from __future__ import annotations

from html import escape
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _gray(array: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.uint8(np.clip((np.asarray(array, dtype=float) - low) / max(high - low, 1e-8), 0, 1) * 255)


def _save_preview(path: Path, array: np.ndarray, maximum: int = 640) -> None:
    image = Image.fromarray(array)
    image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    image.save(path)


def _overlay(reference: np.ndarray, moving: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = np.stack((reference, moving, reference), axis=-1)
    if valid is not None:
        result[~valid.astype(bool)] = (235, 115, 25)
    return result


def _metric(frame: dict, *keys: str) -> str:
    for key in keys:
        value = frame.get(key)
        if value is not None:
            try:
                return f"{float(value):.3f}"
            except (ValueError, TypeError):
                pass
    return "—"


def _extra_arrays(directory, manifest, shape):
    """Load persisted screening maps without calling them anatomical validity."""
    from .io import read_sequence
    result = {}
    for name, key in (("support", "correspondence_file"), ("confidence", "confidence_file")):
        if manifest.get(key):
            value, _ = read_sequence(directory / manifest[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} for review")
            if name == "support" and not np.isin(value,[0,1]).all():
                raise ValueError("Correspondence support must be binary")
            if name == "confidence" and (np.any(value < 0) or np.any(value > 1)):
                raise ValueError("Confidence scores must be finite values in [0,1]")
            result[name] = value
    return result


def _field_views(directory, frame, original, low, high):
    """Create displacement/grid and optional coarse views from the saved pull map."""
    path = frame.get("field_file")
    if not path:
        return {}
    field_path = directory / path
    if frame.get("field_sha256") and hashlib.sha256(field_path.read_bytes()).hexdigest() != frame["field_sha256"]:
        raise ValueError("Transform fingerprint mismatch while generating review")
    with np.load(field_path, allow_pickle=False) as stored:
        field = np.asarray(stored["pull_displacement_yx"], dtype=float)
    if field.shape != (2, *original.shape) or not np.isfinite(field).all():
        raise ValueError("Invalid pull displacement for review")
    magnitude = np.linalg.norm(field, axis=0)
    scale = max(float(np.percentile(magnitude, 99)), 1.)
    heat = np.zeros((*original.shape, 3), dtype=np.uint8)
    strength = np.clip(magnitude / scale, 0, 1)
    heat[..., 0] = np.uint8(255 * strength)
    heat[..., 1] = np.uint8(210 * strength ** .5)
    heat[..., 2] = np.uint8(110 * (1-strength))
    grid_image = Image.fromarray(_gray(original, low, high)).convert("RGB")
    draw = ImageDraw.Draw(grid_image)
    height, width = original.shape
    spacing = max(12, min(height, width)//16)
    # The grid shows where fixed-grid points sample the original moving image.
    for y in range(spacing//2, height, spacing):
        points = [(float(x + field[1, y, x]), float(y + field[0, y, x])) for x in range(0, width, 3)]
        if len(points)>1:
            draw.line(points, fill=(40, 235, 220), width=1)
    for x in range(spacing//2, width, spacing):
        points = [(float(x + field[1, y, x]), float(y + field[0, y, x])) for y in range(0, height, 3)]
        if len(points)>1:
            draw.line(points, fill=(40, 235, 220), width=1)
    views = {"displacement": heat, "deformation": np.asarray(grid_image)}
    matrix = frame.get("coarse_pull_matrix_xy")
    if matrix is not None:
        from scipy.ndimage import map_coordinates
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3,3) or not np.isfinite(matrix).all():
            raise ValueError("Invalid coarse pull matrix for review")
        yy, xx = np.mgrid[:height, :width]
        # Avoid spurious floating-point warnings from small Accelerate BLAS products.
        points = np.einsum("ij,jk->ik", matrix,
                           np.stack((xx.ravel(), yy.ravel(), np.ones(height*width))),
                           optimize=False)
        if not np.allclose(points[2], 1):
            raise ValueError("Expected affine coarse pull coordinates")
        coarse = map_coordinates(np.asarray(original, dtype=np.float32),
                                 [points[1].reshape(height,width), points[0].reshape(height,width)],
                                 order=1, mode="constant", cval=float(np.median(original)), prefilter=False)
        views["coarse"] = _gray(coarse, low, high)
    return views


def write_report(output_dir, manifest: dict, originals: np.ndarray, registered: np.ndarray,
                 validity: np.ndarray) -> Path:
    """Write report.html and PNG previews, preserving original frame index labels.

    Arrays are in [selected frame, row, column] order, matching manifest['frames'].
    Contrast limits are shared by all views so before/after comparisons are fair.
    """
    directory = Path(output_dir)
    assets = directory / "review"
    assets.mkdir(exist_ok=True)
    originals, registered, validity = map(np.asarray, (originals, registered, validity))
    frames = manifest["frames"]
    if originals.shape != registered.shape or originals.shape != validity.shape or len(frames) != len(originals):
        raise ValueError("Review arrays must share [frame, row, column] shape and match manifest frames")
    reference_frame = int(manifest["reference_frame"])
    reference_index = next(i for i, frame in enumerate(frames) if int(frame["source_frame"]) == reference_frame)
    # Use detector interior for display windowing: a uniform black border can
    # otherwise drive the lower percentile to zero and hide all vessel contrast.
    # This affects previews only; annotation image intensities remain untouched.
    height, width = originals.shape[1:]
    x0, y0 = int(width * .1), int(height * .1)
    sample = originals[:, y0:height-y0:max(1, height // 128),
                       x0:width-x0:max(1, width // 128)]
    low, high = np.nanpercentile(sample, (0.5, 99.5))
    ref_gray = _gray(originals[reference_index], low, high)
    _save_preview(assets / "reference.png", ref_gray)
    schema2 = manifest.get("schema_version") == 2
    maps = _extra_arrays(directory, manifest, originals.shape) if schema2 else {}
    if schema2:
        Image.fromarray(ref_gray).save(assets / "reference-full.png")
    included = [i for i, frame in enumerate(frames) if frame.get("included_in_minip", True)]
    bright = manifest.get("config", {}).get("polarity", "dark") == "bright"
    reducer, padding = (np.max, -np.inf) if bright else (np.min, np.inf)
    if included:
        before_minip = reducer(originals[included], axis=0)
        projection_support = validity[included].astype(bool)
        if "support" in maps:
            projection_support = projection_support & maps["support"][included].astype(bool)
        after_minip = reducer(np.where(projection_support, registered[included], padding), axis=0)
    else:
        before_minip = np.full(originals.shape[1:], high)
        after_minip = before_minip.copy()
    after_minip[~np.isfinite(after_minip)] = high
    _save_preview(assets / "minip-before.png", _gray(before_minip, low, high))
    _save_preview(assets / "minip-after.png", _gray(after_minip, low, high))
    items = []
    rows = []
    for index, frame in enumerate(frames):
        original_gray = _gray(originals[index], low, high)
        aligned_gray = _gray(registered[index], low, high)
        prefix = f"frame-{index:04d}"
        images = {
            "original": original_gray,
            "registered": aligned_gray,
            "before": _overlay(ref_gray, original_gray),
            "after": _overlay(ref_gray, aligned_gray, validity[index]),
        }
        urls = {}
        full_urls = {}
        for key, pixels in images.items():
            filename = f"{prefix}-{key}.png"
            _save_preview(assets / filename, pixels)
            urls[key] = f"review/{filename}"
            if schema2:
                full_filename = f"{prefix}-{key}-full.png"
                Image.fromarray(pixels).save(assets / full_filename)
                full_urls[key] = f"review/{full_filename}"
        extra_urls = {}
        if schema2:
            extras = _field_views(directory, frame, originals[index], low, high)
            if "support" in maps:
                # Orange denotes unsupported correspondence, not zero-intensity image padding.
                support_overlay = np.repeat(aligned_gray[..., None], 3, axis=-1)
                effective_support = maps["support"][index].astype(bool) & validity[index].astype(bool)
                support_overlay[~effective_support] = (235,115,25)
                extras["support"] = support_overlay
            if "confidence" in maps:
                confidence = np.clip(maps["confidence"][index], 0, 1)
                extras["confidence"] = np.stack((np.uint8(255*(1-confidence)),
                                                   np.uint8(255*confidence), np.zeros_like(aligned_gray)),axis=-1)
            if "coarse" in extras:
                extras["coarse"] = _overlay(ref_gray, extras["coarse"])
            for key,pixels in extras.items():
                filename = f"{prefix}-{key}.png"
                Image.fromarray(pixels).save(assets / filename)
                extra_urls[key] = f"review/{filename}"
        status = str(frame.get("status", "review"))
        source = int(frame["source_frame"])
        dy = float(frame.get("coarse_shift_y_px", frame.get("shift_y_px", 0)))
        dx = float(frame.get("coarse_shift_x_px", frame.get("shift_x_px", 0)))
        before = _metric(frame, "ncc_before", "correlation_before", "before_ncc")
        after = _metric(frame, "ncc_after", "correlation_after", "after_ncc")
        reason = frame.get("reasons", frame.get("flags", frame.get("reason", [])))
        if isinstance(reason, list):
            reason = "; ".join(map(str, reason))
        item = {"index": index, "source": source, "status": status, "dx": dx, "dy": dy,
                "before_ncc": before, "after_ncc": after, "reason": str(reason), "urls": urls}
        if schema2:
            item.update(full_urls=full_urls, extra_urls=extra_urls,
                        registration_valid=frame.get("registration_valid") is True,
                        qa={key:frame[key] for key in ("qa", "inverse_qa", "match_count", "support_fraction",
                             "evaluation", "duplicate_of_source_frame", "field_sha256") if key in frame})
        items.append(item)
        flagged = status not in ("accepted", "reference", "manual", "ok", "estimated")
        rows.append(f'<tr class="{"flagged" if flagged else ""}"><td><button onclick="show({index})">{source}</button></td>'
                    f'<td>{escape(status)}</td><td>{dx:.2f}</td><td>{dy:.2f}</td>'
                    f'<td>{before}</td><td>{after}</td><td>{escape(str(reason))}</td></tr>')
    # JSON is inert inside this data block; escape '<' to prevent HTML termination.
    data = json.dumps(items, allow_nan=False).replace("<", "\\u003c")
    manifest_path = directory / "manifest.json"
    fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else ""
    editor_data = json.dumps({"config": manifest.get("config", {}),
                              "shape": list(originals.shape[1:]),
                              "run_id": manifest.get("run_id", "sequence"),
                              "group_id": manifest.get("group_id"),
                              "manifest_sha256": fingerprint,
                              "schema_version": manifest.get("schema_version",1),
                              "registration_method": manifest.get("registration_method", "translation")},
                             allow_nan=False).replace("<", "\\u003c")
    title = escape(str(manifest.get("run_id", Path(manifest.get("source_file", "DSA sequence")).stem)))
    configured_roi = manifest.get("config", {}).get("roi", manifest.get("roi"))
    roi = escape(str(configured_roi if configured_roi is not None else "automatic central 80%"))
    document = _HTML.replace("__TITLE__", title).replace("__REFERENCE__", str(reference_frame))
    document = document.replace("__ROI__", roi).replace("__ROWS__", "\n".join(rows)).replace("__DATA__", data)
    document = document.replace("__EDITOR_DATA__", editor_data)
    document = document.replace(
        "The same eligible frames contribute to both images. Invalid shifted borders are excluded after registration.",
        "These overviews use eligible frames. The registered view additionally excludes invalid source sampling and, "
        "for local registration, unsupported correspondence. Contributions can differ by pixel; "
        "use the matched-frame comparison above to inspect changes in sharpness.")
    projection = manifest.get("projection_comparison")
    if projection:
        label = escape(str(projection["label"]))
        viewer = escape(str(projection["paths"]["viewer"]), quote=True)
        projection_link = (f'<section class="editor"><h2>{label} and sharpness comparison</h2>'
                           f'<p><a href="{viewer}">Open native before/after {label} comparison and blink</a>. '
                           'Both images use every selected frame and the same contrast window. '
                           'Only invalid source sampling is omitted after alignment; failed or excluded '
                           'frames remain in this diagnostic comparison.</p>'
                           '<p><a href="minip.png">Curated projection PNG</a> · '
                           '<a href="minip.nrrd">Curated projection NRRD</a>. '
                           'The curated projection uses eligible frames and available matching support; '
                           'apparent sharpness does not establish anatomical accuracy.</p></section>')
        document = document.replace('<section class="editor" aria-label="Configuration editor">',
                                    projection_link+'\n<section class="editor" aria-label="Configuration editor">')
    if schema2:
        document = document.replace('<script id="frame-data"', _DENSE_SECTION+'\n<script id="frame-data"')
        document = document.replace("</body>", _DENSE_SCRIPT+"\n</body>")
        document = document.replace("<th>dx (px)</th>", "<th>Coarse dx (px)</th>")
        document = document.replace("<th>dy (px)</th>", "<th>Coarse dy (px)</th>")
        document = document.replace("dx and dy are pixel translations applied to the moving frame to bring it into the reference grid. Positive dx moves pixels right; positive dy moves them down.",
                                    "Coarse dx/dy summarize global initialization only. The complete local transform is the saved pull-displacement field; rotation/scale and local deformation cannot be summarized by these two numbers.")
        document = document.replace("Orange marks invalid shifted borders.","Orange in the main overlay marks invalid source sampling.")
    path = directory / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSA review · __TITLE__</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;background:#111720;color:#e4ebf3;max-width:1550px;margin:auto;padding:28px}
h1{font-size:25px;overflow-wrap:anywhere;margin-bottom:8px}h2{font-size:20px;margin-top:28px}p{max-width:1000px}a{color:#8dc7ff}button,select,input{font:inherit}button,select{background:#263445;color:#f4f7fb;border:1px solid #6d7c8f;border-radius:5px;padding:5px 10px;cursor:pointer}button:hover{background:#374d65}input[type=range]{width:min(550px,65vw);vertical-align:middle}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0 8px}.panes{display:grid;grid-template-columns:1fr 1fr;gap:15px}.pane{margin:0;background:#070b11;padding:10px;border:1px solid #354153;border-radius:7px}.pane img{display:block;width:100%;max-height:70vh;object-fit:contain;image-rendering:auto}.pane figcaption{font-size:14px;color:#bac8d8;margin-bottom:8px}.status{padding:10px 14px;background:#243249;border-radius:6px}.note{color:#bdcbdc}.magenta{color:#ef94ef}.green{color:#99e69f}.orange{color:#ffa760}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:7px 12px;border-bottom:1px solid #354153}th{color:#acc0d9}.flagged{background:#49331a}td button{padding:1px 10px}.contact{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:10px}.contact-card{border:1px solid #354153;border-radius:5px;padding:4px}.contact-card button{padding:3px;width:100%}.contact img{display:block;width:100%;height:110px;object-fit:contain}.current{outline:3px solid #72b8ff}.small{font-size:13px}details{margin-top:20px}summary{cursor:pointer;font-weight:600}.editor{border:1px solid #5a7b99;background:#182533;padding:14px;border-radius:7px;margin:18px 0}.editor h2{margin:0 0 6px;font-size:18px}.editor p{margin:7px 0}.editor input[type=number]{width:85px;background:#0d1723;color:#e4ebf3;border:1px solid #5a7b99;border-radius:4px;padding:4px}.stage{position:relative}.roi-box{position:absolute;border:2px solid #79e9fa;background:#79e9fa15;pointer-events:none;display:none}.drawing img{cursor:crosshair;touch-action:none}.include-label{display:block;padding:4px;font-size:13px}#config-message{color:#a4d2fc}@media(max-width:700px){body{padding:15px}.panes{grid-template-columns:1fr}.pane img{max-height:55vh}}
</style></head><body>
<h1>DSA alignment review · __TITLE__</h1>
<p>Reference: original frame <strong>__REFERENCE__</strong>. All indices use the original acquisition and start at zero.
Registration ROI (x, y, width, height): <strong>__ROI__</strong>.</p>
<p class="note">Review shared bifurcations inside and outside the ROI. Compare residual movement with the width of the smallest branch you intend to trace. Scores and automatic flags are screening aids; this report does not establish anatomical accuracy. Previews share one robust display window estimated from the central 80% of the selected images to avoid black detector borders. Display windowing does not change the saved image intensities. These previews are reduced in size; inspect thin vessels at full resolution in Slicer.</p>
<section class="editor" aria-label="Configuration editor"><h2>Prepare a refined alignment</h2><p>Choose useful frames with the checkboxes below. Browse to a sharp frame and use it as the next reference. Draw a region around shared vessels to estimate their movement.</p><div class="controls"><button id="use-reference">Use displayed frame as next reference</button><strong id="next-reference"></strong><button id="draw-roi">Draw registration ROI</button><button id="reset-roi">Reset to central 80%</button></div><div class="controls"><label>x <input id="roi-x" type="number" min="0" step="1"></label><label>y <input id="roi-y" type="number" min="0" step="1"></label><label>width <input id="roi-w" type="number" min="8" step="1"></label><label>height <input id="roi-h" type="number" min="8" step="1"></label><button id="apply-roi">Apply pixel coordinates</button><button id="download-config">Download updated JSON</button></div><p class="small">ROI values are full-resolution source pixels: [x, y, width, height]. Downloading saves configuration only. Run <code>dsa-preprocess prepare --config downloaded.json ... -o NEW_OUTPUT</code> to recompute alignment. Current overlays retain the saved reference __REFERENCE__ until then. Changing the reference clears manual shifts and landmarks tied to the old reference.</p><p id="config-message" role="status"></p></section>
<div class="controls"><button id="previous" aria-label="Previous frame">← Previous</button><input id="scrub" type="range" min="0" value="0" aria-label="Selected frame"><button id="next" aria-label="Next frame">Next →</button><label>View <select id="mode"><option value="overlay">Reference overlays</option><option value="gray">Original / registered</option></select></label></div>
<p id="status" class="status" aria-live="polite"></p>
<p class="small">Overlay: <span class="magenta">reference in magenta</span>, <span class="green">selected frame in green</span>; matching intensities appear gray. <span class="orange">Orange marks invalid shifted borders.</span> Use ← / → keys to browse.</p>
<div class="panes"><figure class="pane"><figcaption id="left-caption"></figcaption><div class="stage" id="left-stage"><a id="left-link" target="_blank"><img id="left" alt="Before alignment" draggable="false"></a><div id="roi-box" class="roi-box"></div></div></figure><figure class="pane"><figcaption id="right-caption"></figcaption><a id="right-link" target="_blank"><img id="right" alt="After alignment"></a></figure></div>
<h2>Selected frames</h2><div id="contact" class="contact"></div>
<details><summary>Per-frame shifts and screening scores</summary><p class="small">dx and dy are pixel translations applied to the moving frame to bring it into the reference grid. Positive dx moves pixels right; positive dy moves them down. Normalized correlation is measured within the registration ROI on processing copies.</p><div class="tablewrap"><table><thead><tr><th>Original frame</th><th>Status</th><th>dx (px)</th><th>dy (px)</th><th>Before NCC</th><th>After NCC</th><th>Review reason</th></tr></thead><tbody>__ROWS__</tbody></table></div></details>
<details><summary>Intensity projection overview</summary><p class="small">The same eligible frames contribute to both images. Invalid shifted borders are excluded after registration. Dark vessels use a minimum projection (MinIP), bright vessels a maximum projection. If no frames are eligible, these previews are empty. Use individual registered frames for tracing.</p><div class="panes"><figure class="pane"><figcaption>Original eligible frames</figcaption><a href="review/minip-before.png" target="_blank"><img src="review/minip-before.png" alt="Original intensity projection" loading="lazy"></a></figure><figure class="pane"><figcaption>Registered eligible frames, valid pixels only</figcaption><a href="review/minip-after.png" target="_blank"><img src="review/minip-after.png" alt="Registered intensity projection" loading="lazy"></a></figure></div></details>
<p class="note">Review <a href="manifest.json">manifest.json</a> for saved configuration, transforms, and frame provenance. To change the reference, ROI, selected frames, or manual corrections, edit the JSON configuration and prepare a new output directory. Source images remain unchanged.</p>
<script id="frame-data" type="application/json">__DATA__</script>
<script id="editor-data" type="application/json">__EDITOR_DATA__</script>
<script>
const frames=JSON.parse(document.getElementById('frame-data').textContent);let selected=0;
const scrub=document.getElementById('scrub'),mode=document.getElementById('mode'),contact=document.getElementById('contact');scrub.max=frames.length-1;
const editor=JSON.parse(document.getElementById('editor-data').textContent),config=JSON.parse(JSON.stringify(editor.config));const height=editor.shape[0],width=editor.shape[1];const included=new Set(frames.map(f=>f.source));config.reference_frame=Number(config.reference_frame===undefined?__REFERENCE__:config.reference_frame);let drawing=false,start=null;const leftImage=document.getElementById('left'),stage=document.getElementById('left-stage'),box=document.getElementById('roi-box'),message=document.getElementById('config-message');
frames.forEach((f,i)=>{const card=document.createElement('div');card.className='contact-card';const button=document.createElement('button');button.title='Show original frame '+f.source;button.onclick=()=>show(i);const image=document.createElement('img');image.src=f.urls.original;image.loading='lazy';image.alt='Original frame '+f.source;button.append(image,document.createTextNode('Frame '+f.source));const label=document.createElement('label');label.className='include-label';const check=document.createElement('input');check.type='checkbox';check.checked=true;check.dataset.frame=f.source;check.onchange=()=>{if(!check.checked&&f.source===config.reference_frame){check.checked=true;message.textContent='The next reference must stay included. Choose another reference before excluding this frame.';return}if(check.checked)included.add(f.source);else included.delete(f.source);updateEditor()};label.append(check,document.createTextNode(' Include frame'));card.append(button,label);contact.append(card)});
function show(i){selected=Math.max(0,Math.min(frames.length-1,i));const f=frames[selected];scrub.value=selected;const overlay=mode.value==='overlay';const left=f.urls[overlay?'before':'original'],right=f.urls[overlay?'after':'registered'];leftImage.src=left;document.getElementById('right').src=right;document.getElementById('left-link').href=left;document.getElementById('right-link').href=right;document.getElementById('left-caption').textContent=overlay?'Before: selected frame over saved reference':'Original selected frame · draw shared-vessel ROI here';document.getElementById('right-caption').textContent=overlay?'After: registered frame over saved reference':'Registered selected frame';document.getElementById('status').textContent='Original frame '+f.source+' · '+f.status+' · dx '+f.dx.toFixed(2)+' px, dy '+f.dy.toFixed(2)+' px · NCC '+f.before_ncc+' → '+f.after_ncc+(f.reason?' · '+f.reason:'');document.getElementById('previous').disabled=selected===0;document.getElementById('next').disabled=selected===frames.length-1;Array.from(contact.children).forEach((b,j)=>b.classList.toggle('current',j===selected));renderROI();}
function effectiveROI(){return config.roi||[Math.floor(width*.1),Math.floor(height*.1),width-2*Math.floor(width*.1),height-2*Math.floor(height*.1)]}
function contentRect(){const r=leftImage.getBoundingClientRect(),ratio=width/height;let w=r.width,h=w/ratio;if(h>r.height){h=r.height;w=h*ratio}return{x:r.left+(r.width-w)/2,y:r.top+(r.height-h)/2,w,h}}
function sourcePoint(event){const r=contentRect();return[Math.max(0,Math.min(width,Math.round((event.clientX-r.x)/r.w*width))),Math.max(0,Math.min(height,Math.round((event.clientY-r.y)/r.h*height)))]}
function renderROI(){if(mode.value!=='gray'){box.style.display='none';return}const roi=effectiveROI(),r=contentRect(),s=stage.getBoundingClientRect();if(!r.w||!r.h)return;Object.assign(box.style,{display:'block',left:(r.x-s.left+roi[0]/width*r.w)+'px',top:(r.y-s.top+roi[1]/height*r.h)+'px',width:(roi[2]/width*r.w)+'px',height:(roi[3]/height*r.h)+'px',borderStyle:config.roi?'solid':'dashed'})}
function updateEditor(){const roi=effectiveROI();['x','y','w','h'].forEach((name,i)=>document.getElementById('roi-'+name).value=roi[i]);document.getElementById('next-reference').textContent='Next reference: '+config.reference_frame+' · '+included.size+' frame(s)';renderROI()}
document.getElementById('use-reference').onclick=()=>{const frame=frames[selected].source;if(frame!==config.reference_frame){config.manual_shifts={};config.landmarks={}}config.reference_frame=frame;included.add(frame);contact.querySelector('input[data-frame="'+frame+'"]').checked=true;message.textContent='Next reference set to original frame '+frame+'. Download the configuration and prepare a new output to update the alignment.';updateEditor()};
document.getElementById('draw-roi').onclick=()=>{drawing=true;stage.classList.add('drawing');mode.value='gray';show(selected);message.textContent='Drag a rectangle on the ORIGINAL image at left around vessels visible in the chosen frames. Keep text, borders, and large subtraction artifacts outside.';stage.scrollIntoView({behavior:'smooth',block:'center'})};
leftImage.addEventListener('pointerdown',event=>{if(!drawing||mode.value!=='gray')return;event.preventDefault();start=sourcePoint(event);config.roi=[start[0],start[1],0,0];leftImage.setPointerCapture(event.pointerId)});
leftImage.addEventListener('pointermove',event=>{if(!start)return;const end=sourcePoint(event);config.roi=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.abs(end[0]-start[0]),Math.abs(end[1]-start[1])];updateEditor()});
leftImage.addEventListener('pointerup',event=>{if(!start)return;event.preventDefault();start=null;drawing=false;stage.classList.remove('drawing');if(config.roi[2]<8||config.roi[3]<8){config.roi=null;message.textContent='ROI must be at least 8 × 8 source pixels; restored the central default.'}else message.textContent='ROI saved in pending configuration: ['+config.roi.join(', ')+']. Download JSON and prepare again.';updateEditor()});
leftImage.addEventListener('pointercancel',()=>{start=null;drawing=false;stage.classList.remove('drawing');if(config.roi&&Math.min(config.roi[2],config.roi[3])<8)config.roi=null;updateEditor()});
document.getElementById('left-link').onclick=event=>{if(drawing||start){event.preventDefault();return}if(mode.value==='gray'){event.preventDefault()}};
document.getElementById('reset-roi').onclick=()=>{config.roi=null;drawing=false;start=null;stage.classList.remove('drawing');message.textContent='Pending ROI reset to the central 80%.';updateEditor()};
document.getElementById('apply-roi').onclick=()=>{const roi=['x','y','w','h'].map(name=>Number(document.getElementById('roi-'+name).value));if(!roi.every(Number.isInteger)||roi[0]<0||roi[1]<0||roi[2]<8||roi[3]<8||roi[0]+roi[2]>width||roi[1]+roi[3]>height){message.textContent='Enter an integer ROI at least 8 × 8 pixels and inside the '+width+' × '+height+' source image.';return}config.roi=roi;mode.value='gray';show(selected);updateEditor();message.textContent='ROI saved in pending configuration. Download JSON and prepare again.'};
document.getElementById('download-config').onclick=()=>{const result=JSON.parse(JSON.stringify(config));result.frames=Array.from(included).sort((a,b)=>a-b);for(const key of ['manual_shifts','landmarks']){result[key]=Object.fromEntries(Object.entries(result[key]||{}).filter(([frame])=>included.has(Number(frame))&&Number(frame)!==result.reference_frame))}result.exclude_from_minip=(result.exclude_from_minip||[]).filter(frame=>included.has(frame));const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=String(editor.run_id).replace(/[^a-zA-Z0-9_.-]/g,'_')+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message.textContent='Configuration downloaded. Use it with dsa-preprocess prepare and a new output directory.'};
leftImage.addEventListener('load',renderROI);window.addEventListener('resize',renderROI);
scrub.oninput=()=>show(Number(scrub.value));mode.onchange=()=>show(selected);document.getElementById('previous').onclick=()=>show(selected-1);document.getElementById('next').onclick=()=>show(selected+1);document.addEventListener('keydown',e=>{if(['SELECT','INPUT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();show(selected-1)}if(e.key==='ArrowRight'){e.preventDefault();show(selected+1)}});show(0);
updateEditor();
</script></body></html>'''


_DENSE_SECTION = r'''
<section class="editor" aria-label="Local deformation review">
<h2>Local registration and anatomical review</h2>
<p id="dense-method"></p>
<p>Coarse dx/dy describe initialization only. The saved displacement field contains the complete transform.
Orange in the correspondence view marks unsupported matching; the confidence view is a screening score,
not a probability of anatomical correctness. Click an image or its full-resolution link to inspect small vessels.</p>
<div class="controls"><label>Additional view <select id="extra-mode">
<option value="support">Correspondence support</option><option value="confidence">Matching confidence (red low, green high)</option>
<option value="coarse">Coarse global overlay</option><option value="displacement">Displacement magnitude (relative color scale)</option>
<option value="deformation">Pull grid on original moving image</option></select></label>
<button id="blink-final">Blink reference / final</button><a id="final-full" target="_blank">Final overlay at full resolution</a>
<a id="original-full" target="_blank">Original source at full resolution</a><a href="review/reference-full.png" target="_blank">Original reference at full resolution</a></div>
<figure class="pane"><figcaption id="extra-caption"></figcaption><a id="extra-link" target="_blank"><img id="extra-image" alt="Registration screening map"></a></figure>
<details><summary>Saved numerical quality diagnostics</summary><pre id="dense-qa" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details>
<h2>Record a review decision</h2>
<p>A numerical candidate remains unapproved until you review shared branches. Accepted regions are in the
fixed reference image coordinates, [x, y, width, height]. Mark only regions whose anatomy and label coverage you
actually inspected. Leaving regions empty records a frame decision without declaring label coverage.
A reference identity transform never establishes full-tree completeness.</p>
<div class="controls"><label>Displayed frame <select id="review-decision"><option value="pending">Pending</option><option value="accepted">Anatomically reviewed</option><option value="rejected">Rejected</option></select></label>
<label>Reviewed reference regions <input id="review-regions" size="42" placeholder="[[100,200,150,120]]" aria-label="Reviewed reference regions"></label></div>
<label>Review notes <input id="review-notes" size="75" aria-label="Anatomical review notes"></label>
<div class="controls"><button id="save-decision">Record displayed frame decision</button><button id="download-decisions">Download review decisions</button></div>
<p id="review-message" role="status"></p>
</section>
<details class="editor"><summary>Collect independent anatomical correspondences</summary>
<p>Use the original reference and original moving frame below. Click the same bifurcation in each image,
then check the coordinates. Do not use evaluation points as fitting constraints. Different filling or a
projected crossing is a reason to mark visibility ambiguous. Browser zoom or the full-resolution image links
help place small branch points; these displays do not establish subpixel reader accuracy.</p>
<div class="panes"><figure class="pane landmark-pane"><figcaption>Original reference <a href="review/reference-full.png" target="_blank">full resolution</a></figcaption>
<img id="point-reference" src="review/reference-full.png" alt="Original reference for independent landmarks" style="max-height:none;height:auto;cursor:crosshair"></figure>
<figure class="pane landmark-pane"><figcaption id="point-moving-caption">Original moving frame</figcaption><img id="point-moving" alt="Original moving image for independent landmarks" style="max-height:none;height:auto;cursor:crosshair"></figure></div>
<div class="controls"><label>Point ID <input id="point-id" value="branch-1" size="16"></label>
<label>Role <select id="point-role"><option value="evaluation">Evaluation (withheld)</option><option value="fitting">Fitting constraint</option></select></label>
<label>Visibility <select id="point-visibility"><option value="shared">Shared</option><option value="reference_only">Reference only</option><option value="moving_only">Moving only</option><option value="ambiguous">Ambiguous</option></select></label></div>
<div class="controls"><label>Reference x,y <input id="point-reference-xy" size="20" placeholder="[x,y]"></label>
<label>Moving x,y <input id="point-moving-xy" size="20" placeholder="[x,y]"></label>
<label>Reference width (px) <input id="point-width" type="number" min="0.01" step="any"></label></div>
<div class="controls"><label>Branch/region <input id="point-region" size="24"></label><label>Notes <input id="point-notes" size="50"></label></div>
<div class="controls"><button id="add-point">Record correspondence</button><button id="remove-point">Remove last correspondence</button><button id="download-points">Download landmark JSON</button></div>
<p id="point-message" role="status"></p><pre id="point-list" style="white-space:pre-wrap"></pre>
</details>'''


_DENSE_SCRIPT = r'''<script>
const decisions=new Map(),landmarks=[];
let blinkTimer=null,blinkOn=false,pointSource=null;
const byId=id=>document.getElementById(id);
byId('dense-method').textContent='Method: '+editor.registration_method+' · group '+(editor.group_id||editor.run_id)+'. All methods require anatomical review.';
function downloadJSON(value,suffix){if(!editor.manifest_sha256){throw new Error('The preparation manifest fingerprint is unavailable. Regenerate this report before recording evidence.')}const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=editor.run_id.replace(/[^a-zA-Z0-9_.-]/g,'_')+suffix;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function regionsFromText(text){const value=text.trim()?JSON.parse(text):[];if(!Array.isArray(value)||value.some(r=>!Array.isArray(r)||r.length!==4||!r.every(Number.isInteger)||r[0]<0||r[1]<0||r[2]<=0||r[3]<=0||r[0]+r[2]>width||r[1]+r[3]>height))throw new Error('Regions must be integer [x,y,width,height] rectangles inside the reference image.');return value}
function renderExtra(){const f=frames[selected],key=byId('extra-mode').value,url=f.extra_urls[key];byId('extra-image').style.display=url?'block':'none';byId('extra-image').src=url||'';byId('extra-link').href=url||'#';byId('extra-caption').textContent='Source frame '+f.source+' · '+byId('extra-mode').selectedOptions[0].text+(url?'':' · unavailable');byId('dense-qa').textContent=JSON.stringify(f.qa,null,2);byId('final-full').href=f.full_urls.after;byId('original-full').href=f.full_urls.original}
function stopBlink(){if(blinkTimer!==null){clearInterval(blinkTimer);blinkTimer=null}byId('blink-final').textContent='Blink reference / final'}
const originalShow=show;
show=function(i){stopBlink();originalShow(i);const f=frames[selected],overlay=mode.value==='overlay';byId('left-link').href=f.full_urls[overlay?'before':'original'];byId('right-link').href=f.full_urls[overlay?'after':'registered'];byId('status').textContent='Original frame '+f.source+' · '+f.status+' · coarse dx '+f.dx.toFixed(2)+' px, dy '+f.dy.toFixed(2)+' px · NCC '+f.before_ncc+' → '+f.after_ncc+(f.reason?' · '+f.reason:'');renderExtra();const d=decisions.get(f.source)||{decision:'pending',regions:[],notes:''};byId('review-decision').value=d.decision;byId('review-decision').options[1].disabled=!f.registration_valid;byId('review-regions').value=JSON.stringify(d.regions);byId('review-notes').value=d.notes;if(pointSource!==f.source){byId('point-reference-xy').value='';byId('point-moving-xy').value='';byId('point-width').value='';pointSource=f.source}byId('point-moving').src=f.full_urls.original;byId('point-moving-caption').textContent='Original source frame '+f.source};
byId('extra-mode').onchange=renderExtra;
const oldReferenceClick=byId('use-reference').onclick;
byId('use-reference').onclick=()=>{const old=config.reference_frame;oldReferenceClick();if(old!==config.reference_frame&&config.registration){delete config.registration.fitting_landmarks;delete config.registration.paths;message.textContent+=' Deformable fitting landmarks and registration paths tied to the old reference were cleared.'}};
const oldConfigClick=byId('download-config').onclick;
byId('download-config').onclick=()=>{let removed=0;if(config.registration){if(config.registration.fitting_landmarks)config.registration.fitting_landmarks=Object.fromEntries(Object.entries(config.registration.fitting_landmarks).filter(([frame])=>included.has(Number(frame))&&Number(frame)!==config.reference_frame));if(config.registration.paths){const entries=Object.entries(config.registration.paths);const kept=entries.filter(([frame,path])=>included.has(Number(frame))&&Number(frame)!==config.reference_frame&&Array.isArray(path)&&path.every(source=>included.has(Number(source))));removed=entries.length-kept.length;config.registration.paths=Object.fromEntries(kept)}}oldConfigClick();if(removed)message.textContent+=' Removed '+removed+' registration path(s) that referenced excluded frames.'};
byId('blink-final').onclick=()=>{if(blinkTimer!==null){stopBlink();show(selected);return}byId('blink-final').textContent='Stop blink';blinkOn=false;blinkTimer=setInterval(()=>{blinkOn=!blinkOn;byId('right').src=blinkOn?'review/reference-full.png':frames[selected].full_urls.registered;byId('right-caption').textContent=blinkOn?'Original reference':'Final registered source frame '+frames[selected].source},500)};
byId('save-decision').onclick=()=>{try{const f=frames[selected],decision=byId('review-decision').value;if(decision==='accepted'&&!f.registration_valid)throw new Error('This transform failed numerical/support checks. A review decision cannot override a failed transform.');const regions=regionsFromText(byId('review-regions').value);if(decision!=='accepted'&&regions.length)throw new Error('Reviewed coverage regions require an accepted anatomical review decision.');decisions.set(f.source,{source_frame:f.source,decision,regions,notes:byId('review-notes').value});byId('review-message').textContent='Recorded original frame '+f.source+'. '+decisions.size+' frame decision(s) stored; download to preserve them.'}catch(e){byId('review-message').textContent=e.message}};
byId('download-decisions').onclick=()=>{try{downloadJSON({schema_version:1,prepared_manifest_sha256:editor.manifest_sha256,group_id:editor.group_id,decisions:Array.from(decisions.values())},'-review.json');byId('review-message').textContent='Review decisions downloaded. Import this file into the matching Slicer session.'}catch(e){byId('review-message').textContent=e.message}};
for(const side of ['reference','moving'])byId('point-'+side).onclick=event=>{const r=event.currentTarget.getBoundingClientRect();const x=Math.max(0,Math.min(width-1,(event.clientX-r.left)/r.width*width));const y=Math.max(0,Math.min(height-1,(event.clientY-r.top)/r.height*height));byId('point-'+side+'-xy').value=JSON.stringify([Number(x.toFixed(2)),Number(y.toFixed(2))])};
function parsePoint(id){const raw=byId(id).value.trim();if(!raw)return null;const p=JSON.parse(raw);if(!Array.isArray(p)||p.length!==2||!p.every(Number.isFinite)||p[0]<0||p[1]<0||p[0]>width-1||p[1]>height-1)throw new Error('Points must be finite [x,y] coordinates inside the original image.');return p}
function renderPoints(){byId('point-list').textContent=landmarks.map(p=>'Frame '+p.source_frame+' / '+p.point_id+' / '+p.role+' / '+p.visibility+' / '+JSON.stringify(p.reference)+' ↔ '+JSON.stringify(p.moving)).join('\n')}
byId('add-point').onclick=()=>{try{const f=frames[selected],visibility=byId('point-visibility').value,reference=parsePoint('point-reference-xy'),moving=parsePoint('point-moving-xy'),id=byId('point-id').value.trim();if(!id)throw new Error('Provide a branch point ID.');if(landmarks.some(p=>p.source_frame===f.source&&p.point_id===id))throw new Error('This frame/point ID is already recorded.');if(visibility==='shared'&&(!reference||!moving))throw new Error('Shared points require both reference and moving coordinates.');if(visibility==='reference_only'&&(!reference||moving))throw new Error('Reference-only points require reference coordinates and an empty moving field.');if(visibility==='moving_only'&&(!moving||reference))throw new Error('Moving-only points require moving coordinates and an empty reference field.');const raw=byId('point-width').value.trim(),w=raw?Number(raw):null;if(w!==null&&(!Number.isFinite(w)||w<=0))throw new Error('Vessel width must be positive when provided.');landmarks.push({source_frame:f.source,point_id:id,role:byId('point-role').value,visibility,reference,moving,reference_width_px:w,region:byId('point-region').value,notes:byId('point-notes').value});renderPoints();byId('point-message').textContent='Recorded '+landmarks.length+' correspondences; download to preserve them.';byId('point-id').value='branch-'+(landmarks.length+1)}catch(e){byId('point-message').textContent=e.message}};
byId('remove-point').onclick=()=>{landmarks.pop();renderPoints()};
byId('download-points').onclick=()=>{try{downloadJSON({schema_version:1,prepared_manifest_sha256:editor.manifest_sha256,coordinate_convention:'original_full_resolution_xy',landmarks},'-landmarks.json');byId('point-message').textContent='Landmark JSON downloaded. Use dsa-preprocess evaluate with this prepared directory.'}catch(e){byId('point-message').textContent=e.message}};
// The original-image click is reserved for drawing the pending registration ROI.
// Its full-resolution file remains available through the separate final/original links.
show(selected);
</script>'''
