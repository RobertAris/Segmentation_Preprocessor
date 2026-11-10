"""Automatic intensity projections and matched-frame sharpness inspection.

Projection arrays use original intensities already in memory. This module never
resamples an image, changes frame eligibility, or establishes anatomical accuracy.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import write_volume


def _project(images, validity, bright):
    """Reduce one frame at a time, avoiding an extra full sequence allocation."""
    initial = -np.inf if bright else np.inf
    result = np.full(images.shape[1:], initial, dtype=np.float32)
    counts = np.zeros(images.shape[1:], dtype=np.uint32)
    reducer = np.maximum if bright else np.minimum
    for index, image in enumerate(images):
        if not np.isfinite(image).all():
            raise ValueError("Projection intensities must be finite")
        valid = True if validity is None else validity[index]
        reducer(result, image, out=result, where=valid)
        counts += valid
    return result, counts


def _display_window(before, after, after_valid):
    height, width = before.shape
    y, x = int(height * .1), int(width * .1)
    central = (slice(y, height-y), slice(x, width-x))
    before_sample = before[central].ravel()
    after_sample = after[central][after_valid[central]]
    low, high = np.percentile(np.concatenate((before_sample, after_sample)), (.5, 99.5))
    if high <= low:
        high = low + max(abs(float(low)) * 1e-6, 1.)
    return float(low), float(high)


def _gray(array, low, high):
    return np.uint8(np.clip((np.asarray(array, dtype=float)-low)/(high-low), 0, 1)*255)


def write_projection_outputs(output_dir, manifest, originals, aligned, validity,
                             eligible_projection, eligible_validity, header):
    """Write default projection views and return JSON-safe manifest metadata.

    The existing curated projection remains separate. The matched comparison uses
    every selected frame, including numerically failed or explicitly excluded
    frames, and excludes only invalid source sampling after alignment. This keeps
    selective matching support from manufacturing an apparent sharpness gain.
    Call within preparation's temporary directory before saving its final manifest.
    """
    originals, aligned, validity = map(np.asarray, (originals, aligned, validity))
    frames = manifest["frames"]
    if (originals.ndim != 3 or not len(originals) or min(originals.shape[1:]) < 1
            or originals.shape != aligned.shape or aligned.shape != validity.shape
            or len(frames) != len(originals)):
        raise ValueError("Projection arrays must share nonempty [frame,row,column] shape and manifest order")
    if not np.isin(validity, [0, 1]).all():
        raise ValueError("Projection source validity must be binary")
    source_frames = [frame["source_frame"] for frame in frames]
    if (any(type(index) is not int or index < 0 for index in source_frames)
            or len(set(source_frames)) != len(source_frames)):
        raise ValueError("Projection frames must be unique original integer indices")
    eligible_projection, eligible_validity = map(np.asarray, (eligible_projection, eligible_validity))
    if (eligible_projection.shape != originals.shape[1:]
            or eligible_validity.shape != eligible_projection.shape
            or not np.isfinite(eligible_projection).all()
            or not np.isin(eligible_validity, [0, 1]).all()):
        raise ValueError("Eligible projection and binary validity must match the reference grid")
    polarity = manifest.get("config", {}).get("polarity", "dark")
    if polarity not in {"dark", "bright"}:
        raise ValueError("Projection polarity must be dark or bright")
    bright = polarity == "bright"
    operation, label = ("maximum", "MIP") if bright else ("minimum", "MinIP")
    before, _ = _project(originals, None, bright)
    after, counts = _project(aligned, validity.astype(bool, copy=False), bright)
    after_valid = counts > 0
    low, high = _display_window(before, after, after_valid)
    fill = low if bright else high
    after[~after_valid] = fill

    directory = Path(output_dir)
    comparison = directory / "minip-comparison"
    comparison.mkdir()
    paths = {"viewer": "minip-comparison/index.html", "before_png": "minip-comparison/before.png",
             "after_png": "minip-comparison/after.png", "comparison_png": "minip-comparison/comparison.png",
             "before_nrrd": "minip-comparison/before.nrrd", "after_nrrd": "minip-comparison/after.nrrd",
             "after_validity_nrrd": "minip-comparison/after-validity.nrrd",
             "after_frame_count_nrrd": "minip-comparison/after-frame-count.nrrd",
             "metadata": "minip-comparison/projection.json", "curated_png": "minip.png"}
    before_pixels, after_pixels = _gray(before, low, high), _gray(after, low, high)
    curated_pixels = _gray(eligible_projection, low, high)
    curated_pixels[~eligible_validity.astype(bool)] = 0 if bright else 255
    Image.fromarray(curated_pixels).save(directory / paths["curated_png"])
    Image.fromarray(before_pixels).save(directory / paths["before_png"])
    Image.fromarray(after_pixels).save(directory / paths["after_png"])
    for key, array in (("before_nrrd", before), ("after_nrrd", after),
                       ("after_validity_nrrd", after_valid.astype(np.uint8)),
                       ("after_frame_count_nrrd", counts)):
        write_volume(directory / paths[key], array, header)

    height, width = before.shape
    panel = Image.new("RGB", (2*width+36, height+58), "#f4f6f8")
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow 10.0 lacks the size argument.
        font = ImageFont.load_default()
    for column, (title, pixels) in enumerate((("Before alignment", before_pixels),
                                            ("After alignment", after_pixels))):
        x = 12 + column*(width+12)
        draw.text((x, 10), title, font=font, fill="#17212b")
        panel.paste(Image.fromarray(pixels).convert("RGB"), (x, 46))
    panel.save(directory / paths["comparison_png"])

    eligible_frames = [frame["source_frame"] for frame in frames if frame.get("included_in_minip", True)]
    ineligible_frames = [index for index in source_frames if index not in eligible_frames]
    metadata = {"operation": operation, "label": label, "polarity": polarity,
                "source_frames": source_frames, "frame_count": len(source_frames),
                "reference_frame": manifest["reference_frame"], "paths": paths,
                "display_window": {"low": low, "high": high, "mapping": "linear_shared_before_after_and_curated"},
                "after_valid_frame_count_range": [int(counts.min()), int(counts.max())],
                "curated_source_frames": eligible_frames,
                "included_in_comparison_but_excluded_from_curated": ineligible_frames,
                "comparison_support": "all selected frames; only invalid source interpolation samples excluded after alignment",
                "curated_support": ("eligible frames intersected with source and correspondence support"
                                    if manifest.get("schema_version") == 2 else "eligible frames with valid source sampling"),
                "no_additional_resampling": True, "sharpening_or_denoising": False,
                "anatomical_acceptance_established": False,
                "note": "Diagnostic comparison includes failed/excluded frames and unverified deformation regions. Apparent sharpness is not anatomical accuracy."}
    (directory / paths["metadata"]).write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    excluded = ", ".join(map(str, ineligible_frames)) or "none"
    title = escape(str(manifest.get("run_id", "DSA sequence")))
    document = (_HTML.replace("__TITLE__", title).replace("__LABEL__", label)
                .replace("__OPERATION__", operation).replace("__FRAMES__", ", ".join(map(str, source_frames)))
                .replace("__COUNT__", str(len(source_frames))).replace("__REFERENCE__", str(manifest["reference_frame"]))
                .replace("__EXCLUDED__", excluded).replace("__WIDTH__", str(width)).replace("__HEIGHT__", str(height)))
    (directory / paths["viewer"]).write_text(document)
    return metadata


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__LABEL__ comparison - __TITLE__</title>
<style>body{font:16px/1.5 system-ui,sans-serif;color:#17212b;background:#f4f6f8;max-width:1500px;margin:24px auto;padding:0 20px}h1{line-height:1.2}a{color:#075b9a}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0;background:white;padding:12px}figcaption{font-weight:600}img{max-width:100%;height:auto}button{font:inherit;padding:9px 16px;margin:12px 0}#blink-panel{display:none}.note{padding:14px;background:#fff0ce;border-radius:6px}p{max-width:1150px}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style></head><body>
<h1>__TITLE__: __LABEL__ before and after alignment</h1>
<p>__COUNT__ selected frames, reference __REFERENCE__. The __OPERATION__ over time uses the same original frame selection in both images, with one shared linear display window and no added sharpening or denoising. Invalid source sampling is omitted after alignment.</p>
<p class="note">This comparison includes every selected frame and deformation outside matching support. Original frame indices: __FRAMES__. Frames excluded from the curated projection but retained here: __EXCLUDED__. Review individual frames before annotating; apparent sharpness does not establish anatomical accuracy.</p>
<div class="pair"><figure><figcaption>Before alignment - all selected frames</figcaption><a href="before.png" target="_blank"><img src="before.png" alt="Original __LABEL__"></a></figure><figure><figcaption>After alignment - all selected frames, valid sampling only</figcaption><a href="after.png" target="_blank"><img src="after.png" alt="Aligned __LABEL__"></a></figure></div>
<p>Click an image for native __WIDTH__ x __HEIGHT__ pixels. Compare branch edges at native resolution.</p>
<button id="show-blink" aria-expanded="false">Show blink comparison</button> <button id="play" hidden>Start blinking</button>
<figure id="blink-panel"><figcaption id="blink-caption">Before alignment</figcaption><a id="blink-link" href="before.png" target="_blank"><img id="blink-image" src="before.png" alt="__LABEL__ blink comparison"></a></figure>
<h2>Saved projections</h2><p><a href="after.nrrd">Aligned all-frame __LABEL__ (NRRD)</a> · <a href="before.nrrd">Original all-frame __LABEL__ (NRRD)</a> · <a href="comparison.png">Combined comparison PNG</a> · <a href="../minip.png">Curated __LABEL__ PNG</a> · <a href="../minip.nrrd">Curated __LABEL__ NRRD</a> · <a href="after-validity.nrrd">Aligned sampling validity</a> · <a href="after-frame-count.nrrd">Contributing frame counts</a> · <a href="projection.json">Projection metadata</a> · <a href="../report.html">Frame-by-frame review</a></p>
<p>The curated projection excludes ineligible frames and, for local registration, unsupported correspondence. Fewer contributions can reduce apparent blur; it has different support from the main comparison.</p>
<script>
let timer=null,after=false;
const toggle=document.getElementById('show-blink'),panel=document.getElementById('blink-panel'),play=document.getElementById('play'),image=document.getElementById('blink-image'),caption=document.getElementById('blink-caption'),link=document.getElementById('blink-link');
function stop(){clearInterval(timer);timer=null;play.textContent='Start blinking'}
toggle.onclick=()=>{const show=panel.style.display!=='block';panel.style.display=show?'block':'none';play.hidden=!show;toggle.setAttribute('aria-expanded',String(show));toggle.textContent=show?'Hide blink comparison':'Show blink comparison';if(!show)stop()};
play.onclick=()=>{if(timer){stop();return}play.textContent='Stop blinking';timer=setInterval(()=>{after=!after;image.src=after?'after.png':'before.png';link.href=image.getAttribute('src');caption.textContent=after?'After alignment':'Before alignment'},700)};
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});
</script></body></html>'''
