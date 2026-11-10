"""Deterministic, bidirectionally checked local image correspondences.

Correspondences are screening proposals. In particular, a high patch score cannot
establish vessel identity during filling, washout, or projected overlap.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from skimage.feature import corner_peaks, corner_shi_tomasi, match_template
from skimage.filters import sato

from .registration import features, roi_slices
from .transforms import sample_support


def registration_features(image, polarity="dark", representation="local_contrast"):
    if representation == "local_contrast":
        result = features(image, polarity)
    elif representation == "vesselness":
        a = np.asarray(image, dtype=np.float32)
        lo,hi=np.percentile(a,[.5,99.5])
        if hi-lo < 1e-8:
            return np.zeros_like(a)
        a=np.clip((a-lo)/(hi-lo),0,1)
        result=sato(a,sigmas=(1,2,3),black_ridges=polarity=="dark")
    else:
        raise ValueError("feature must be local_contrast or vesselness")
    high=float(np.percentile(result,99.5))
    return np.clip(result/max(high,1e-8),0,1).astype(np.float32)


def region_mask(shape, regions, exclude_regions, roi=None):
    regions = regions or ([roi] if roi is not None else [None])
    mask=np.zeros(shape,dtype=bool)
    labels=np.full(shape,-1,dtype=np.int32)
    for i,region in enumerate(regions):
        slices=roi_slices(shape,region)
        mask[slices]=True
        labels[slices]=i
    for region in exclude_regions:
        slices=roi_slices(shape,region)
        mask[slices]=False
        labels[slices]=-1
    return mask,labels


def source_support_mask(shape, exclude_regions):
    """Source exclusions are separate from the reference matching ROI."""
    allowed=np.ones(shape,dtype=bool)
    for region in exclude_regions:
        allowed[roi_slices(shape,region)]=False
    return allowed


def patch_supported(support, center_yx, half):
    """Require every bilinear contributor of a subpixel-centered patch."""
    if support is None:
        return True
    dy,dx=np.mgrid[-half:half+1,-half:half+1]
    coords=np.stack([dy+center_yx[0],dx+center_yx[1]])
    return bool(sample_support(support,coords).all())


def _peak(response):
    index=np.unravel_index(np.argmax(response),response.shape)
    score=float(response[index])
    other=response.copy()
    y,x=index
    other[max(0,y-2):y+3,max(0,x-2):x+3]=-np.inf
    second=float(other.max()) if np.isfinite(other).any() else -1.
    offset=np.zeros(2,dtype=float)
    for axis,i in enumerate(index):
        if 0<i<response.shape[axis]-1:
            a=list(index);b=list(index)
            a[axis]-=1;b[axis]+=1
            left,right=float(response[tuple(a)]),float(response[tuple(b)])
            denominator=left-2*score+right
            if np.isfinite([left,right,denominator]).all() and denominator < -1e-8:
                offset[axis]=np.clip(.5*(left-right)/denominator,-.5,.5)
    return np.asarray(index,dtype=float)+offset,score,score-second


def _match(template_image, search_image, center_yx, search_center_yx, half, radius,
           template_support=None,search_support=None):
    cy,cx=np.rint(center_yx).astype(int)
    sy,sx=np.rint(search_center_yx).astype(int)
    h,w=template_image.shape
    if (cy-half<0 or cx-half<0 or cy+half>=h or cx+half>=w or
        sy-half-radius<0 or sx-half-radius<0 or
        sy+half+radius>=search_image.shape[0] or sx+half+radius>=search_image.shape[1]):
        return None
    if not patch_supported(template_support,center_yx,half):
        return None
    template=template_image[cy-half:cy+half+1,cx-half:cx+half+1]
    if float(template.std())<.01:
        return None
    search=search_image[sy-half-radius:sy+half+radius+1,
                        sx-half-radius:sx+half+radius+1]
    response=match_template(search,template,pad_input=False)
    if search_support is not None:
        local_support=search_support[sy-half-radius:sy+half+radius+1,
                                     sx-half-radius:sx+half+radius+1]
        center_support=ndimage.minimum_filter(local_support.astype(np.uint8),
            size=2*half+1,mode="constant",cval=0)[half:-half,half:-half].astype(bool)
        if center_support.sum()<9:
            return None
        response=np.where(center_support,response,-np.inf)
    location,score,margin=_peak(response)
    target=np.asarray([sy-radius,sx-radius])+location
    # The integer maximum can be supported while its subpixel refinement samples
    # across an exclusion boundary. Recheck all refined interpolation positions.
    if not patch_supported(search_support,target,half):
        return None
    return target,score,margin


def _single_level_correspondences(reference,moving,mask,labels,config,valid=None,
                                  guides=None):
    """Reference and already-coarsened moving features share one pixel grid.

    Return reference_xy, moving_xy, weights, and diagnostics. Moving positions are
    in the coarsened grid and must be transformed back to original coordinates.
    """
    half=config["patch_size"]//2
    radius=config["search_radius_px"]
    step=config["patch_step"]
    # Junction-like structure is more informative than the axis of a straight tube.
    corners=corner_shi_tomasi(reference,sigma=1.5)
    candidate=corner_peaks(corners,min_distance=max(3,step//2),
                           threshold_rel=.005,exclude_border=half+radius,
                           num_peaks=500,labels=mask.astype(np.uint8))
    yy,xx=np.mgrid[half+radius:reference.shape[0]-half-radius:step,
                   half+radius:reference.shape[1]-half-radius:step]
    grid=np.column_stack([yy.ravel(),xx.ravel()])
    if len(grid):
        grid=grid[mask[grid[:,0],grid[:,1]]]
    candidates=np.unique(np.vstack([candidate,grid]),axis=0) if len(grid) else candidate
    # Process high-information points first; the cap bounds runtime deterministically.
    if len(candidates):
        order=np.argsort(-corners[candidates[:,0],candidates[:,1]],kind="stable")
        candidates=candidates[order[:500]]
    points,targets,scores,region_ids=[],[],[],[]
    for center in candidates:
        cy,cx=center
        patch=mask[cy-half:cy+half+1,cx-half:cx+half+1]
        if patch.shape!=(2*half+1,2*half+1) or not patch.all():
            continue
        search_center=center.astype(float)
        if guides is not None:
            displacement=_guide_displacement(center[::-1],*guides,config["support_radius_px"],
                                               config["max_fit_error_px"],config["max_displacement_px"])
            if displacement is not None:
                search_center+=displacement[::-1]
        match=_match(reference,moving,center,search_center,half,radius,
                      template_support=mask,search_support=valid)
        if match is None:
            continue
        target,score,margin=match
        if score<config["min_match_ncc"] or margin<config["min_peak_margin"]:
            continue
        reverse=_match(moving,reference,target,center,half,radius,
                       template_support=valid,search_support=mask)
        if reverse is None:
            continue
        back,reverse_score,reverse_margin=reverse
        if (reverse_score<config["min_match_ncc"] or
            reverse_margin<config["min_peak_margin"] or
            np.linalg.norm(back-center)>config["max_forward_backward_error_px"]):
            continue
        points.append(center[::-1]);targets.append(target[::-1])
        scores.append(min(score,reverse_score));region_ids.append(int(labels[cy,cx]))
    if not points:
        return np.empty((0,2)),np.empty((0,2)),np.empty(0),{"candidate_count":int(len(candidates)),"match_count":0}
    region_ids=np.asarray(region_ids)
    weights=np.asarray(scores)
    # Equal total influence per represented region, rather than per trunk pixel.
    for region in np.unique(region_ids):
        subset=region_ids==region
        weights[subset]/=max(float(weights[subset].sum()),1e-8)
    weights*=len(weights)/max(len(np.unique(region_ids)),1)
    return np.asarray(points,float),np.asarray(targets,float),weights,{
        "candidate_count":int(len(candidates)),"match_count":len(points),
        "represented_region_count":int(len(np.unique(region_ids))),
        "median_match_ncc":float(np.median(scores))}


def _guide_displacement(point_xy,reference_xy,moving_xy,radius,max_error,maximum):
    """A local motion prior needs distributed agreement; one nearby vessel is insufficient."""
    distances=np.linalg.norm(reference_xy-point_xy,axis=1)
    chosen=np.flatnonzero(distances<=radius)
    if len(chosen)<4:
        return None
    chosen=chosen[np.argsort(distances[chosen],kind="stable")[:12]]
    offset=reference_xy[chosen]-point_xy
    spread=np.linalg.eigvalsh((offset-offset.mean(0)).T@(offset-offset.mean(0))/len(offset))
    if spread[0]<.25:
        return None
    try:
        hull=ConvexHull(offset)
    except QhullError:
        return None
    if (hull.equations[:,-1]>1e-8).any():
        return None
    design=np.column_stack([offset,np.ones(len(offset))])
    weights=1/np.maximum(distances[chosen],1.)
    delta=moving_xy[chosen]-reference_xy[chosen]
    coefficient=np.linalg.lstsq(design*np.sqrt(weights[:,None]),
                                delta*np.sqrt(weights[:,None]),rcond=None)[0]
    residual=np.linalg.norm(design@coefficient-delta,axis=1)
    if np.percentile(residual,75)>max_error:
        return None
    if np.linalg.norm(coefficient[2])>maximum:
        return None
    # This prior only chooses the center of a new search. It never supplies a
    # returned correspondence without a distinctive, reciprocal fine-scale match.
    return coefficient[2]


def _pyramid_level(image,mask,scale):
    if scale==1:
        return image,mask
    sigma=.5*np.sqrt(scale*scale-1)
    radius=int(np.ceil(4*sigma))
    filtered=ndimage.gaussian_filter(image,sigma,mode="nearest")[::scale,::scale]
    # Every antialias contributor must be supported, not merely the selected pixel.
    supported=ndimage.minimum_filter(mask.astype(np.uint8),size=2*radius+1,
                                     mode="constant",cval=0)[::scale,::scale].astype(bool)
    return filtered,supported


def local_correspondences(reference,moving,mask,labels,config,valid=None):
    """Match an optional image pyramid and require evidence at its finest level.

    Pyramid samples use image[::scale], so point and displacement coordinates map
    back by exactly multiplying by scale. search_radius_px is the finest-level
    radius; coarser levels use the same pixel radius, bounded by the configured
    full-resolution maximum displacement. Coarse matches only guide a fresh fine
    search, never replace it or interpolate missing vessel correspondences.
    """
    mask=np.asarray(mask,dtype=bool)
    if valid is None and config.get("exclude_regions"):
        raise ValueError("Moving/source support must be supplied when source exclusions are configured")
    moving_support=np.ones(moving.shape,dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
    if mask.shape!=reference.shape or moving_support.shape!=moving.shape:
        raise ValueError("Correspondence support masks must match their image grids")
    scales=config.get("pyramid_scales",[1])
    if (not isinstance(scales,list) or not scales or len(scales)>4 or scales[-1]!=1 or
        any(type(s) is not int or s not in {1,2,4,8} for s in scales) or
        any(a<=b for a,b in zip(scales,scales[1:]))):
        raise ValueError("pyramid_scales must be decreasing powers of two <=8 ending in 1")
    guides=None
    levels=[]
    for scale in scales:
        a,reference_support=_pyramid_level(reference,mask,scale)
        b,source_support=_pyramid_level(moving,moving_support,scale)
        level_config={**config,
            "patch_size":max(9,2*int(np.rint((config["patch_size"]//2)/scale))+1),
            "patch_step":max(4,int(np.ceil(config["patch_step"]/scale))),
            "search_radius_px":max(1,min(config["search_radius_px"],
                                          int(np.ceil(config["max_displacement_px"]/scale)))),
            "max_forward_backward_error_px":config["max_forward_backward_error_px"]/scale,
            "support_radius_px":config["support_radius_px"]/scale,
            "max_fit_error_px":config["max_fit_error_px"]/scale,
            "max_displacement_px":config["max_displacement_px"]/scale}
        level_guides=None if guides is None else (guides[0]/scale,guides[1]/scale)
        p,q,w,detail=_single_level_correspondences(a,b,reference_support,labels[::scale,::scale],
                                                  level_config,source_support,level_guides)
        p*=scale;q*=scale
        # Fine support is checked again on the original full-resolution source
        # masks, including each subpixel interpolation contributor.
        if len(p):
            good=np.asarray([patch_supported(mask,point[::-1],config["patch_size"]//2) and
                             patch_supported(moving_support,target[::-1],config["patch_size"]//2)
                             for point,target in zip(p,q)])
            p,q,w=p[good],q[good],w[good]
        levels.append({"scale":scale,"match_count":int(len(p)),
                       "search_radius_full_resolution_px":level_config["search_radius_px"]*scale})
        guides=(p,q) if len(p)>=4 else None
    detail.update(match_count=int(len(p)),pyramid_levels=levels,
                   finest_evidence_scale=1,
                   source_support_note="All source exclusions and interpolation contributors checked in original source sampling")
    return p,q,w,detail


def correspondence_support(shape, points_xy, weights, radius, region):
    """Bound support near measured proposals; do not bless global extrapolation."""
    seeds=np.zeros(shape,dtype=bool)
    strength=np.zeros(shape,dtype=np.float32)
    for point,weight in zip(points_xy,weights):
        x,y=np.rint(point).astype(int)
        if 0<=y<shape[0] and 0<=x<shape[1]:
            seeds[y,x]=True;strength[y,x]=max(strength[y,x],float(weight))
    if not seeds.any():
        return seeds,strength
    distance,nearest=ndimage.distance_transform_edt(~seeds,return_indices=True)
    confidence=np.exp(-.5*(distance/max(radius/2,1))**2)*strength[tuple(nearest)]
    confidence/=max(float(confidence.max()),1.)
    supported=(distance<=radius)&region
    confidence[~supported]=0
    return supported,confidence.astype(np.float32)
