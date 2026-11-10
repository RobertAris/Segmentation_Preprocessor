"""Opt-in vessel-feature registration with explicit numerical/correspondence QA.

An estimator result is a candidate for anatomical review. No image similarity,
smoothness, or inverse-consistency check automatically accepts an annotation.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr
from skimage.registration import optical_flow_tvl1, phase_cross_correlation

from .correspondence import (correspondence_support, local_correspondences,
                             region_mask, registration_features, source_support_mask)
from .registration import correlation
from .transforms import (affine_to_pull, apply_pull, field_quality,
                         identity_grid, invert_pull_field, map_points, sample_support, shift_to_pull)


DEFAULT_REGISTRATION = {
    "method": "bspline", "global_model": "translation", "regions": [],
    "exclude_regions": [], "feature": "local_contrast", "grid_spacings_px": [128,64],
    "regularization": 1., "patch_size": 31, "patch_step": 24,
    "search_radius_px": 12, "min_match_ncc": .55, "min_peak_margin": .02,
    "max_forward_backward_error_px": 1.5, "min_matches": 6,
    "max_fit_error_px": 2.,
    "pyramid_scales": [1],
    "support_radius_px": 48., "max_displacement_px": 32., "max_local_strain": .5,
    "min_jacobian": .2, "inverse_tolerance_px": .1, "inverse_iterations": 80,
    "tvl1_attachment": 15., "tvl1_tightness": .3, "tvl1_num_warp": 5,
    "tvl1_num_iter": 10, "demons_iterations": 50, "demons_smoothing": 1.5,
}
METHODS={"translation","rigid","similarity","affine","bspline","tvl1","demons"}


def validate_registration_config(config):
    if not isinstance(config,dict):
        raise ValueError("registration must be an object")
    unknown=set(config)-set(DEFAULT_REGISTRATION)
    if unknown:
        raise ValueError("Unknown registration keys: "+", ".join(sorted(unknown)))
    result={**DEFAULT_REGISTRATION,**config}
    if result["method"] not in METHODS:
        raise ValueError("Unknown registration method")
    if result["global_model"] not in {"translation","rigid","similarity","affine"}:
        raise ValueError("global_model must be translation, rigid, similarity or affine")
    if result["feature"] not in {"local_contrast","vesselness"}:
        raise ValueError("feature must be local_contrast or vesselness")
    for key in ["regions","exclude_regions"]:
        regions=result[key]
        if not isinstance(regions,list):
            raise ValueError(f"{key} must be a list of [x,y,width,height] regions")
        for region in regions:
            if (not isinstance(region,(list,tuple)) or len(region)!=4 or
                any(type(v) is not int for v in region) or min(region[:2])<0 or min(region[2:])<8):
                raise ValueError(f"{key} entries must be integer [x,y,width,height], size >=8")
        result[key]=[list(r) for r in regions]
    spacings=result["grid_spacings_px"]
    if (not isinstance(spacings,list) or not spacings or
        any(type(s) is not int or s<32 for s in spacings) or
        any(a<=b for a,b in zip(spacings,spacings[1:]))):
        raise ValueError("grid_spacings_px must be strictly decreasing integer spacings >=32")
    result["grid_spacings_px"]=list(spacings)
    scales=result["pyramid_scales"]
    if (not isinstance(scales,list) or not scales or len(scales)>4 or scales[-1]!=1 or
        any(type(s) is not int or s not in {1,2,4,8} for s in scales) or
        any(a<=b for a,b in zip(scales,scales[1:]))):
        raise ValueError("pyramid_scales must be decreasing powers of two <=8 ending in 1")
    result["pyramid_scales"]=list(scales)
    integer_limits={"patch_size":(9,151),"patch_step":(4,256),"search_radius_px":(1,128),
                    "min_matches":(3,500),"inverse_iterations":(1,1000),
                    "tvl1_num_warp":(1,100),"tvl1_num_iter":(1,1000),"demons_iterations":(1,1000)}
    for key,(low,high) in integer_limits.items():
        value=result[key]
        if type(value) is not int or not low<=value<=high:
            raise ValueError(f"{key} must be integer {low}..{high}")
    if result["patch_size"]%2!=1:
        raise ValueError("patch_size must be odd")
    for key in ["regularization","max_forward_backward_error_px","max_fit_error_px","support_radius_px",
                "max_displacement_px","max_local_strain","min_jacobian","inverse_tolerance_px",
                "tvl1_attachment","tvl1_tightness","demons_smoothing"]:
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
            raise ValueError(f"{key} must be a positive finite number")
        result[key]=float(value)
    for key in ["min_match_ncc","min_peak_margin"]:
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError(f"{key} must be finite in 0..1")
        result[key]=float(value)
    return result


def _global_shift(reference,moving,mask,labels,maximum,source_allowed):
    shifts=[]
    for region_id in np.unique(labels[mask]):
        yy,xx=np.nonzero(mask&(labels==region_id))
        if not len(yy):
            continue
        y0,y1,x0,x1=yy.min(),yy.max()+1,xx.min(),xx.max()+1
        a,b=reference[y0:y1,x0:x1].copy(),moving[y0:y1,x0:x1].copy()
        local=mask[y0:y1,x0:x1]
        a[~local]=0;b[~local|~source_allowed[y0:y1,x0:x1]]=0
        if a.std()<.001 or b.std()<.001:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            shift,_,_=phase_cross_correlation(a,b,normalization=None,upsample_factor=10)
        if np.isfinite(shift).all() and np.linalg.norm(shift)<=maximum:
            shifts.append(shift)
    return np.median(shifts,axis=0) if shifts else np.zeros(2)


def _fit_global(reference_xy,moving_xy,weights,model):
    """Deterministic robust weighted global fit, in [x,y] coordinates."""
    p,q=np.asarray(reference_xy),np.asarray(moving_xy)
    base=np.asarray(weights,dtype=float)
    weights=base.copy()
    matrix=np.eye(3)
    if model in {"rigid","similarity","affine"} and np.linalg.matrix_rank(p-p.mean(0))<2:
        raise ValueError("Global fit requires spatially distributed correspondences")
    for _ in range(8):
        total=max(weights.sum(),1e-8)
        pc=(weights[:,None]*p).sum(0)/total
        qc=(weights[:,None]*q).sum(0)/total
        if model=="translation":
            matrix[:2,:2]=np.eye(2)
            matrix[:2,2]=(weights[:,None]*(q-p)).sum(0)/total
        elif model in {"rigid","similarity"}:
            pp,qq=p-pc,q-qc
            covariance=(pp*weights[:,None]).T@qq
            u,s,vh=np.linalg.svd(covariance)
            sign=np.ones(2)
            if np.linalg.det(vh.T@u.T)<0:sign[-1]=-1
            rotation=vh.T@np.diag(sign)@u.T
            scale=1.
            if model=="similarity":
                scale=float(np.dot(s,sign)/max(np.sum(weights*np.sum(pp*pp,axis=1)),1e-8))
            matrix[:2,:2]=scale*rotation
            matrix[:2,2]=qc-matrix[:2,:2]@pc
        else:
            design=np.column_stack([p,np.ones(len(p))])
            fitted=np.linalg.lstsq(design*np.sqrt(weights[:,None]),
                                   q*np.sqrt(weights[:,None]),rcond=None)[0]
            matrix[:2]=fitted.T
        prediction=p@matrix[:2,:2].T+matrix[:2,2]
        residual=np.linalg.norm(prediction-q,axis=1)
        delta=max(1.,float(np.median(residual))*1.5)
        weights=base*np.minimum(1.,delta/np.maximum(residual,1e-8))
    return matrix


def _basis(points_xy,spacing,grid_shape):
    ny,nx=grid_shape
    coordinate=np.asarray(points_xy)/spacing+1
    base=np.floor(coordinate).astype(int)-1
    f=coordinate-np.floor(coordinate)
    basis=[(1-f)**3/6,(3*f**3-6*f*f+4)/6,
           (-3*f**3+3*f*f+3*f+1)/6,f**3/6]
    rows=[];cols=[];values=[]
    for j in range(4):
        for i in range(4):
            xx=base[:,0]+i;yy=base[:,1]+j
            rows.extend(range(len(points_xy)))
            cols.extend((yy*nx+xx).tolist())
            values.extend((basis[i][:,0]*basis[j][:,1]).tolist())
    return sparse.csr_matrix((values,(rows,cols)),shape=(len(points_xy),ny*nx))


def _bending(grid_shape):
    ny,nx=grid_shape
    def second(n):
        return sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],
                            [0,1,2],shape=(n-2,n),format="csr")
    def first(n):
        return sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format="csr")
    return sparse.vstack([sparse.kron(sparse.eye(ny),second(nx)),
                          sparse.kron(second(ny),sparse.eye(nx)),
                          np.sqrt(2)*sparse.kron(first(ny),first(nx))],format="csr")


def _bspline_residual(shape,points_xy,targets_xy,weights,config):
    field=np.zeros((2,*shape),dtype=np.float32)
    for spacing in config["grid_spacings_px"]:
        grid_shape=(int(np.ceil((shape[0]-1)/spacing))+4,
                    int(np.ceil((shape[1]-1)/spacing))+4)
        design=_basis(points_xy,spacing,grid_shape)
        predicted,_=map_points(field,points_xy)
        target=(targets_xy-predicted)[:,::-1]
        penalty=_bending(grid_shape)
        ridge=sparse.eye(np.prod(grid_shape),format="csr")
        robust=np.asarray(weights,dtype=float).copy()
        coefficients=np.zeros((2,np.prod(grid_shape)))
        for _ in range(5):
            weighted=design.multiply(np.sqrt(robust)[:,None])
            system=sparse.vstack([weighted,np.sqrt(config["regularization"])*penalty,
                                   np.sqrt(config["regularization"]*.005)*ridge],format="csr")
            for axis in range(2):
                right=np.concatenate([np.sqrt(robust)*target[:,axis],
                                       np.zeros(system.shape[0]-len(points_xy))])
                coefficients[axis]=lsqr(system,right,atol=1e-7,btol=1e-7,iter_lim=400)[0]
            residual=np.linalg.norm((design@coefficients.T)-target,axis=1)
            delta=max(.5,float(np.median(residual))*1.5)
            robust=weights*np.minimum(1.,delta/np.maximum(residual,1e-8))
        coords=identity_grid(shape)/spacing+1
        for axis in range(2):
            field[axis]+=ndimage.map_coordinates(coefficients[axis].reshape(grid_shape),coords,
                                                 order=3,prefilter=False,mode="nearest")
    return field


def _good_geometry(field,config):
    qa=field_quality(field)
    good=(qa["min_jacobian"]>=config["min_jacobian"] and
          qa["min_cell_jacobian"]>=config["min_jacobian"] and
          qa["folded_cell_fraction"]==0 and
          qa["max_local_strain"]<=config["max_local_strain"] and
          qa["max_displacement_px"]<=config["max_displacement_px"])
    return good,qa


def _validate_landmarks(landmarks,shape):
    if landmarks is None:
        return np.empty((0,2)),np.empty((0,2))
    if not isinstance(landmarks,dict) or set(landmarks)!={"reference","moving"}:
        raise ValueError("fitting_landmarks requires only reference and moving Nx2 arrays")
    p,q=np.asarray(landmarks["reference"],float),np.asarray(landmarks["moving"],float)
    if (p.ndim!=2 or p.shape[1:]!=(2,) or p.shape!=q.shape or len(p)<3 or
        not np.isfinite(p).all() or not np.isfinite(q).all()):
        raise ValueError("Fitting landmarks must be matching finite Nx2 [x,y] arrays, N>=3")
    for points in [p,q]:
        if (points<0).any() or (points[:,0]>shape[1]-1).any() or (points[:,1]>shape[0]-1).any():
            raise ValueError("Fitting landmark outside source image")
    return p,q


def estimate_pair(reference,moving,config,*,polarity="dark",roi=None,fitting_landmarks=None):
    """Estimate one reference->source pull map, with review-only eligibility.

    Global models use robust local correspondences. B-splines fit residual local
    proposals with bending/ridge regularization. TV-L1 and Demons are feature-driven
    comparators without a masked-force objective; support gating is external.
    """
    config=validate_registration_config(config)
    if polarity not in {"dark","bright"}:
        raise ValueError("polarity must be dark or bright")
    reference,moving=np.asarray(reference),np.asarray(moving)
    if (reference.ndim!=2 or reference.shape!=moving.shape or min(reference.shape)<8 or
        not np.isfinite(reference).all() or not np.isfinite(moving).all()):
        raise ValueError("Expected matching finite 2D frames")
    p_manual,q_manual=_validate_landmarks(fitting_landmarks,reference.shape)
    if config["method"]=="demons":
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ValueError("demons requires the optional SimpleITK dependency") from None
    mask,labels=region_mask(reference.shape,config["regions"],config["exclude_regions"],roi)
    source_allowed=source_support_mask(moving.shape,config["exclude_regions"])
    if len(p_manual):
        for points in [p_manual,q_manual]:
            allowed=sample_support(source_allowed,points[:,::-1].T)
            if not allowed.all():
                raise ValueError("Fitting landmark samples excluded source/reference content")
    ref=registration_features(reference,polarity,config["feature"])
    mov=registration_features(moving,polarity,config["feature"])
    identity=np.zeros((2,*reference.shape),dtype=np.float32)
    metrics={"status":"unsupported","reason":"", "method":config["method"],
             "global_model":config["global_model"],"ncc_before":correlation(ref,mov,mask),
             "ncc_after":None,"coarse_shift_y_px":0.,"coarse_shift_x_px":0.,
             "coarse_pull_matrix_xy":np.eye(3).tolist(),"match_count":0,
             "support_fraction":0.,"anatomical_acceptance":False,
             "support_note":"Local image correspondences are proposals, not verified branch identity"}
    def finish(field,support,confidence,valid,reason=None):
        metrics["reason"]=reason or metrics["reason"]
        metrics["support_fraction"]=float(support.mean())
        metrics["qa"]=field_quality(field)
        aligned,source_valid=apply_pull(mov,field)
        metrics["ncc_after"]=correlation(ref,aligned,mask&source_valid)
        return {"pull_displacement_yx":field.astype(np.float32),
                "correspondence_support":support.astype(bool),"confidence":confidence.astype(np.float32),
                "metrics":metrics,"registration_valid":bool(valid)}
    empty=np.zeros(reference.shape,dtype=bool)
    if not mask.any() or ref[mask].std()<.001 or mov[mask].std()<.001:
        return finish(identity,empty,empty.astype(np.float32),False,"Insufficient shared local contrast")
    if np.array_equal(reference,moving):
        if len(p_manual) and not np.allclose(p_manual,q_manual,atol=1e-6):
            raise ValueError("Nonidentity fitting landmarks conflict with pixel-identical images")
        metrics.update(status="review",duplicate_identity=True,ncc_after=1.,
                       inverse_qa={"converged_fraction":1.,"max_converged_residual_px":0.})
        return finish(identity,mask,mask.astype(np.float32),True,
                      "Pixel-identical images; preserve source index but do not count as independent evidence")
    shift=_global_shift(ref,mov,mask,labels,config["max_displacement_px"],source_allowed)
    metrics.update(coarse_shift_y_px=float(shift[0]),coarse_shift_x_px=float(shift[1]))
    coarse=shift_to_pull(reference.shape,shift)
    matrix=np.eye(3);matrix[:2,2]=-shift[::-1]
    coarsened,coarse_valid=apply_pull(mov,coarse)
    moving_support=coarse_valid&sample_support(source_allowed,identity_grid(reference.shape)+coarse)
    p,q,w,detail=local_correspondences(ref,coarsened,mask,labels,config,moving_support)
    metrics.update(detail)
    original_q,_=map_points(coarse,q)
    if len(p_manual):
        p=np.vstack([p,p_manual]);original_q=np.vstack([original_q,q_manual])
        w=np.r_[w,np.full(len(p_manual),max(float(w.sum()),1.)*5/len(p_manual))]
        metrics["fitting_landmark_count"]=len(p_manual)
    if len(p)<config["min_matches"]:
        return finish(coarse,empty,empty.astype(np.float32),False,
                      "Too few distinctive bidirectionally consistent correspondences")
    centered=p-p.mean(0)
    eigenvalues=np.linalg.eigvalsh(centered.T@centered/max(len(p),1))
    metrics["correspondence_spread_eigenvalues_px2"]=eigenvalues.tolist()
    if eigenvalues[0]<1. or eigenvalues[-1]<config["patch_size"]**2/4:
        return finish(coarse,empty,empty.astype(np.float32),False,
                      "Correspondences do not constrain motion across a two-dimensional region")
    global_model=config["method"] if config["method"] in {"translation","rigid","similarity","affine"} else config["global_model"]
    matrix=_fit_global(p,original_q,w,global_model)
    coarse=affine_to_pull(reference.shape,matrix)
    metrics["coarse_pull_matrix_xy"]=matrix.tolist()
    good,_=_good_geometry(coarse,config)
    if not good:
        metrics["status"]="failed"
        return finish(coarse,empty,empty.astype(np.float32),False,"Global fit violates displacement/strain/topology bounds")
    final=coarse
    if config["method"] in {"bspline","tvl1","demons"}:
        coarsened,coarse_valid=apply_pull(mov,coarse)
        moving_support=coarse_valid&sample_support(source_allowed,identity_grid(reference.shape)+coarse)
        p,q,w,detail=local_correspondences(ref,coarsened,mask,labels,config,moving_support)
        metrics.update(detail)
        if len(p_manual):
            # Exact affine inverse converts source landmarks into the coarsened grid.
            inv=np.linalg.inv(matrix)
            q_land=q_manual@inv[:2,:2].T+inv[:2,2]
            p=np.vstack([p,p_manual]);q=np.vstack([q,q_land])
            w=np.r_[w,np.full(len(p_manual),max(float(w.sum()),1.)*5/len(p_manual))]
        if len(p)<config["min_matches"]:
            return finish(coarse,empty,empty.astype(np.float32),False,
                          "Too few residual correspondences after global alignment")
        original_q,_=map_points(coarse,q)
        if config["method"]=="bspline":
            residual=_bspline_residual(reference.shape,p,q,w,config)
        elif config["method"]=="tvl1":
            residual=optical_flow_tvl1(ref,coarsened,attachment=config["tvl1_attachment"],
                tightness=config["tvl1_tightness"],num_warp=config["tvl1_num_warp"],
                num_iter=config["tvl1_num_iter"],prefilter=True)
            metrics["force_masking"]="unmasked_feature_comparator_with_external_support_gating"
        else:
            algorithm=sitk.DiffeomorphicDemonsRegistrationFilter()
            algorithm.SetNumberOfIterations(config["demons_iterations"])
            algorithm.SetStandardDeviations(config["demons_smoothing"])
            displacement=algorithm.Execute(sitk.GetImageFromArray(ref),sitk.GetImageFromArray(coarsened))
            residual=sitk.GetArrayFromImage(displacement)[...,[1,0]].transpose(2,0,1).astype(np.float32)
            metrics["force_masking"]="unmasked_feature_comparator_with_external_support_gating"
        found=False
        for fraction in [1.,.5,.25,.125,.0625]:
            # The affine is analytically defined beyond the reference rectangle;
            # do not clamp its values when residual coordinates leave that grid.
            linear_yx=matrix[:2,:2][::-1,::-1]
            proposal=coarse+np.einsum("ij,jhw->ihw",linear_yx,residual*fraction)
            good,_=_good_geometry(proposal,config)
            if good:
                final=proposal;found=True
                metrics["local_update_fraction"]=fraction
                break
        if not found:
            metrics["status"]="failed"
            return finish(coarse,empty,empty.astype(np.float32),False,
                          "No local update satisfies displacement/strain/topology bounds")
    inverse,inverse_valid,inverse_residual=invert_pull_field(final,
        tolerance_px=config["inverse_tolerance_px"],max_iterations=config["inverse_iterations"])
    grid=identity_grid(reference.shape)
    source_coords=grid+final
    inv_at_source=np.stack([ndimage.map_coordinates(v,source_coords,order=1,mode="nearest",prefilter=False) for v in inverse])
    roundtrip=np.linalg.norm(source_coords+inv_at_source-grid,axis=0)
    inverse_supported=sample_support(inverse_valid,source_coords)
    _,geometric=apply_pull(mov,final)
    final_source_allowed=sample_support(source_allowed,source_coords)
    # Agreement with fitting proposals is a screening diagnostic, never an
    # independent anatomical measurement. Disagreeing proposals create holes.
    recovered,point_domain=map_points(inverse,original_q)
    point_inverse_supported=sample_support(inverse_valid,original_q[:,::-1].T)
    proposal_error=np.linalg.norm(recovered-p,axis=1)
    kept=point_domain&point_inverse_supported&(proposal_error<=config["max_fit_error_px"])
    metrics.update(proposal_reference_error_median_px=float(np.median(proposal_error)),
                   proposal_reference_error_p95_px=float(np.percentile(proposal_error,95)),
                   proposal_reference_error_max_px=float(np.max(proposal_error)),
                   proposal_inlier_count=int(kept.sum()),
                   proposal_error_note="Residual to image/fitting proposals; not independent anatomical accuracy")
    support,confidence=correspondence_support(reference.shape,p[kept],w[kept],config["support_radius_px"],mask)
    if (~kept).any():
        conflicts,_=correspondence_support(reference.shape,p[~kept],np.ones((~kept).sum()),
                                          config["patch_size"]/2,mask)
        support &= ~conflicts
    support &= geometric&final_source_allowed&inverse_supported&(roundtrip<=config["inverse_tolerance_px"]*2)
    confidence[~support]=0
    metrics["inverse_qa"]={"converged_fraction":float(inverse_valid.mean()),
        "max_converged_residual_px":float(inverse_residual[inverse_valid].max()) if inverse_valid.any() else None,
        "max_supported_roundtrip_error_px":float(roundtrip[support].max()) if support.any() else None}
    metrics["match_count"]=len(p)
    enough=kept.sum()>=config["min_matches"]
    if enough:
        centered_kept=p[kept]-p[kept].mean(0)
        spread=np.linalg.eigvalsh(centered_kept.T@centered_kept/kept.sum())
        enough=bool(spread[0]>=1. and spread[-1]>=config["patch_size"]**2/4)
    valid=bool(support.any() and enough)
    if not valid:
        support[:]=False;confidence[:]=0
    metrics["status"]="review" if valid else "unsupported"
    return finish(final,support,confidence,valid,
                  "Candidate requires independent full-tree anatomical review" if valid else
                  "Too few spatially distributed proposals agree with a supported numerical inverse")
