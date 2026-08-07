
__all__ = ['visualize_LAF', 'epilines_to_start_end_points', 'draw_LAF_matches', 'draw_LAF_matches_from_result_dict',
           'draw_LAF_inliers_perspective_repjojected', 'draw_epipolar_errors_in_single_image', 'plot_images',
           'plot_lines', 'plot_color_line_matches']

import torch
import cv2
import numpy as np
import kornia

try:  # kornia >= 0.8.3
    from kornia.image import tensor_to_image, image_to_tensor
except ImportError:  # older kornia
    from kornia.utils import tensor_to_image, image_to_tensor
from typing import List, Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch.nn as nn


def visualize_LAF(img, LAF, img_idx = 0, color='r', linewidth=1,
                  draw_ori = True, fig=None,
                  ax = None, return_fig_ax = False, **kwargs):
    """Draw local affine frames (LAFs) over an image.

    Args:
        img: Batched image tensor of shape :math:`(B, C, H, W)`.
        LAF: LAFs of shape :math:`(B, N, 2, 3)`.
        img_idx: Index of the image in the batch to draw on.
        color: Matplotlib color for the LAF outlines.
        linewidth: Line width of the LAF outlines.
        draw_ori: If True, also draw the orientation line of each LAF.
        fig: Optional existing matplotlib figure to draw on.
        ax: Optional existing matplotlib axes to draw on.
        return_fig_ax: If True, return the figure and axes instead of None.
        **kwargs: Extra keyword arguments forwarded to ``plt.subplots``.

    Returns:
        Tuple of ``(fig, ax)`` if ``return_fig_ax`` is True, otherwise None.

    Example:
        >>> lafs, resp = laf_from_opencv_kpts(kps, mrSize=1.0, with_resp=True)
        >>> visualize_LAF(image_to_tensor(img, False), lafs, 0, 'y', draw_ori=False)
    """
    from kornia_moons.feature import to_numpy_image
    x, y = kornia.feature.laf.get_laf_pts_to_draw(kornia.feature.laf.scale_laf(LAF, 0.5), img_idx)
    if not draw_ori:
        x= x[1:]
        y= y[1:]
    if (fig is None and ax is None):
        fig, ax = plt.subplots(1,1, **kwargs)
    if (fig is not None and ax is None):
        ax = fig.add_axes([0, 0, 1, 1])
    
    ax.imshow(to_numpy_image(img[img_idx]))
    ax.plot(x, y, color, linewidth=linewidth)
    if return_fig_ax : return fig, ax
    return


def epilines_to_start_end_points(epi, h, w):
    """Clip epipolar lines to their image-boundary start/end points.

    Args:
        epi: Epipolar line coefficients ``(a, b, c)`` of shape :math:`(N, 3)`.
        h: Image height.
        w: Image width.

    Returns:
        Stacked start and end points of shape :math:`(2, N, 2)` for an
        ``h`` x ``w`` image, clipped to the image boundary.
    """
    num = len(epi)
    zeros = torch.zeros(num, device=epi.device, dtype=epi.dtype)
    ones = torch.ones(num, device=epi.device, dtype=epi.dtype)
    eps = 1e-8

    
    b = -epi[:,2] / (epi[:,1] + eps)
    k = -epi[:,0] / (epi[:,1] + eps)
    k_positive = k >= 0
    k_negative = k < 0
    
    
    # We find the points of crossing x = 0, x = w, y = 0, y = h
    p1 = torch.stack([zeros,  b], axis=-1)
    p2 = torch.stack([w*ones, k*w + b], axis =-1)
    p3 = torch.stack([-b/(k + eps), zeros], axis=-1)
    p4 = torch.stack([(h - b)/(k + eps), ones*h], axis=-1)
    
    # Now we select those, which are inside of image
    p1_is_start = (p1[:, 1:2] <= h) & (p1[:, 1:2] >= 0)
    p3_is_start = (k_positive * (~p1_is_start.view(-1))).view(-1,1)
    p4_is_start = (k_negative * (~p1_is_start.view(-1))).view(-1,1)
    none_is_start = ~(p1_is_start | p3_is_start |  p4_is_start)
    
    
    p2_is_fin  = (p2[:, 1:2] <= h) & (p2[:, 1:2] >= 0)
    p4_is_fin  = (k_positive * (~p2_is_fin.view(-1))).view(-1,1)
    p3_is_fin  = (k_negative * (~p2_is_fin.view(-1))).view(-1,1)
    
    
    start_lines = p1 * p1_is_start.float() + p3 * p3_is_start.float() + p4 * p4_is_start.float()
    fin_lines   = p2 * p2_is_fin.float()   + p4 * p4_is_fin.float() + p3 * p3_is_fin.float()
    return torch.stack ([start_lines, fin_lines])


def _promote_to_matching_channels(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if img1.ndim != img2.ndim:
        if img1.ndim == 2:
            img1 = np.stack([img1] * 3, axis=-1)
        if img2.ndim == 2:
            img2 = np.stack([img2] * 3, axis=-1)
    return img1, img2

import kornia as K
import kornia.feature as KF
import torch

def draw_LAF_matches(lafs1, lafs2, tent_idxs,  
                     img1, img2, inlier_mask = None, 
                        draw_dict={"inlier_color": (0.2, 1, 0.2),
                               "tentative_color": (0.8, 0.8, 0), 
                               "feature_color": (0.2, 0.5, 1),
                                  "vertical": False}, 
                        Fm: Optional[np.array] = None, H: Optional[np.array] = None,
                        fig = None, ax: Optional = None,
                        return_fig_ax=False):
    """This function draws LAFs, tentative matches, inliers epipolar lines (if F is provided),
    and image1 corners reprojection into image 2 (if H is provided)

    Args:
        lafs1: LAFs of image 1, shape :math:`(1, N1, 2, 3)`.
        lafs2: LAFs of image 2, shape :math:`(1, N2, 2, 3)`.
        tent_idxs: Tentative match indexes (query, train) of shape :math:`(M, 2)`.
        img1: First image, path/tensor/numpy array.
        img2: Second image, path/tensor/numpy array.
        inlier_mask: Optional boolean array/list of length M marking inliers
            among ``tent_idxs``.
        draw_dict: Drawing options. Keys: ``inlier_color`` (RGB tuple or
            None to skip drawing inliers), ``tentative_color`` (RGB tuple or
            None to skip drawing tentative matches), ``feature_color`` (RGB
            tuple or None to skip drawing all detected LAFs), ``vertical``
            (bool, stack images vertically instead of horizontally).
        Fm: Optional fundamental matrix of shape :math:`(3, 3)` used to draw
            inlier epipolar lines.
        H: Optional homography of shape :math:`(3, 3)` used to draw the
            reprojection of image 1's corners into image 2.
        fig: Optional existing matplotlib figure to draw on.
        ax: Optional existing matplotlib axes to draw on.
        return_fig_ax: If True, return the figure and axes instead of None.

    Returns:
        Tuple of ``(fig, ax)`` if ``return_fig_ax`` is True, otherwise None.

    Example:
        >>> match_dists, match_idxs = kornia.feature.match_snn(descs1, descs2, 0.98)
        >>> draw_LAF_matches(lafs1, lafs2, match_idxs, img1, img2, mask,
        ...     draw_dict={"inlier_color": (0.2, 1, 0.2),
        ...                "tentative_color": (0.8, 0.8, 0),
        ...                "feature_color": None, "vertical": False}, H=H)
    """
    from kornia_moons.feature import to_numpy_image, to_np, to_torch
    if inlier_mask is not None:
        inlier_mask = np.array(inlier_mask).reshape(-1)
    lafs1 = to_torch(lafs1).detach().cpu().float()
    lafs2 = to_torch(lafs2).detach().cpu().float()
    tent_idxs = to_torch(tent_idxs).detach().cpu().long()
    img1 = to_numpy_image(img1)
    img2 = to_numpy_image(img2)
    img1, img2 = _promote_to_matching_channels(img1, img2)

    h,w = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    corners_pts = np.float32([[0, 0],
                              [0, h-1],
                              [w-1, h-1],
                              [w-1, 0],
                              [0, 0] ]).reshape(-1,1,2)


    xy1 = KF.get_laf_center(lafs1).reshape(-1, 2)
    xy2 = KF.get_laf_center(lafs2).reshape(-1, 2)
    # If we have no axes, create one
    if (fig is None and ax is None):
        fig, ax = plt.subplots(1,1, figsize=(20,10))
    if (fig is not None and ax is None):
        ax = fig.add_axes([0, 0, 1, 1])

    tent_corrs = torch.stack([xy1[tent_idxs[:,0]], xy2[tent_idxs[:,1]]])
    try:
        vert = draw_dict['vertical']
    except:
        vert = False
    if vert:
        tent_corrs[1,:,1]+=h # shift for the 2nd image
    else:
        tent_corrs[1,:,0]+=w # shift for the 2nd image
    if H is not None:
        dst_corners = cv2.perspectiveTransform(corners_pts, H)
        if vert:
            dst_corners[...,1]+=h
        else:
            dst_corners[...,0]+=w
    # Prepraring canvas
    if not vert:
        if len(img1.shape) == 3:
            new_shape = (max(h, h2), w + w2, img1.shape[2])
        elif len(img1.shape) == 2:
            new_shape = (max(h, h2), w + w2)
    else:
        if len(img1.shape) == 3:
            new_shape = (h + h2,  max(w, w2), img1.shape[2])
        elif len(img1.shape) == 2:
            new_shape = (h + h2,  max(w, w2))        
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    if not vert:
        new_img[0:h, 0:w] = img1
        new_img[0:h2, w:w + w2] = img2
    else:
        new_img[0:h, 0:w] = img1
        new_img[h:h+h2, 0:w2] = img2

    x1, y1 = to_np(KF.laf.get_laf_pts_to_draw(lafs1, 0))
    x2, y2 = to_np(KF.laf.get_laf_pts_to_draw(lafs2, 0))
    if vert:
        y2+=h
    else:
        x2+=w
    # Drawing features
    try:
        fc = draw_dict['feature_color']
    except:
        fc = None
    if fc is not None:
        ax.plot(x1, y1, color=fc)
        ax.plot(x2, y2, color=fc)
        
    tent_corrs = tent_corrs.detach().cpu().numpy()
    #Drawing tentatives
    try:
        tc = draw_dict['tentative_color']
    except:
        tc = None
    if tc is not None:
        ax.plot(tent_corrs[...,0], tent_corrs[...,1], color=tc)
        ax.plot(x1[:, tent_idxs[:,0]], y1[:, tent_idxs[:,0]], color=tc)
        ax.plot(x2[:, tent_idxs[:,1]], y2[:, tent_idxs[:,1]], color=tc)
    try:
        ic = draw_dict['inlier_color']
    except:
        ic = None
    if (ic is not None) and (inlier_mask is not None):
        inlier_mask = inlier_mask > 0
        ax.plot(tent_corrs[..., inlier_mask, 0], tent_corrs[...,inlier_mask, 1], color=ic)
        ax.plot(x1[:, tent_idxs[inlier_mask,0]], y1[:, tent_idxs[inlier_mask,0]], color=ic)
        ax.plot(x2[:, tent_idxs[inlier_mask,1]], y2[:, tent_idxs[inlier_mask,1]], color=ic)
    if H is not None:
        ax.plot(corners_pts[:,0,0], corners_pts[:,0,1], color=(0,0,1))
        ax.plot(dst_corners[:,0,0], dst_corners[:,0,1], color=(0,0,1))
    if (Fm is not None):
        if inlier_mask is None:
            inlier_mask = [True for i in range(len(tent_idxs))]
        tent_corrs2 = torch.stack([xy1[tent_idxs[:,0]], xy2[tent_idxs[:,1]]])
        inl1 = tent_corrs2[0, inlier_mask]
        inl2 = tent_corrs2[1, inlier_mask]
        Ff = torch.from_numpy(Fm).float()
        epi1 = K.geometry.compute_correspond_epilines(inl2.reshape(1,-1, 2), Ff.t()[None])[0]
        epiline1 = epilines_to_start_end_points(epi1, h, w)
        
        epi2 = K.geometry.compute_correspond_epilines(inl1.reshape(1,-1, 2), Ff[None])[0]
        epiline2 = epilines_to_start_end_points(epi2, h2, w2)
        if vert:
            epiline2[...,1]+=h # shift for the 2nd image
        else:
            epiline2[...,0]+=w # shift for the 2nd image            
        
        ax.plot(epiline1[:,:,0], epiline1[:,:,1], color='purple')
        ax.plot(epiline2[:,:,0], epiline2[:,:,1], color='purple')
    # Finally clip the image
    ax.imshow(new_img)
    if not vert:
        ax.set_xlim([0,w+w2])
        ax.set_ylim([max(h,h2),0])
        ax.margins(0,0)
    else:
        ax.set_xlim([0,max(w,w2)])
        ax.set_ylim([h+h2, 0])
        ax.margins(0,0)
    if return_fig_ax : return fig, ax
    return 

def draw_LAF_matches_from_result_dict(result_dict,
                        img1, img2, 
                        draw_dict={"inlier_color": (0.2, 1, 0.2),
                               "tentative_color": (0.8, 0.8, 0), 
                               "feature_color": (0.2, 0.5, 1),
                                  "vertical": False}):
    """Draw matches from a result dictionary produced by a matching pipeline.

    Args:
        result_dict: Dictionary with keys ``feat_dict1`` and ``feat_dict2``
            (each ``{'lafs': tensor of shape (1, N1, 2, 3)}`` and ``(1, N2, 2, 3)`` respectively), ``tents_dict``
            (``{'idxs': long tensor of shape (M, 2)}`` with tentative match
            indexes), ``inlier_mask`` (list/array of M bools), and optionally
            ``'H'`` (homography, :math:`(3, 3)` np.array) and/or ``'F'``
            (fundamental matrix, :math:`(3, 3)` np.array).
        img1: First image, path/tensor/numpy array.
        img2: Second image, path/tensor/numpy array.
        draw_dict: Drawing options, see :func:`draw_LAF_matches`.

    Returns:
        None. Forwards to :func:`draw_LAF_matches` without ``return_fig_ax``.
    """
    if 'H' in result_dict:
        H = result_dict['H']
    else:
        H = None
    if 'F' in result_dict:
        Fm = result_dict['F']
    else:
        Fm = None
    return draw_LAF_matches(result_dict['feat_dict1']['lafs'], 
                        result_dict['feat_dict2']['lafs'],
                               result_dict['tents_dict']['idxs'],  
                        img1, img2, result_dict['inlier_mask'], 
                        draw_dict, Fm, H)

def draw_LAF_inliers_perspective_repjojected(lafs1, lafs2, tent_idxs,  
                        img1, img2, inlier_mask = None, 
                        draw_dict={"inlier_color": (0.2, 1, 0.2),
                               "reprojected_color": (0.2, 0.5, 1),
                                "vertical": False}, 
                        H: np.array = None,
                        fig = None, ax: Optional = None,
                        return_fig_ax=False):
    """This function draws tentative matches and inliers given the homography H

    Note that the function name keeps its original typo
    (``repjojected``) for backward compatibility.

    Args:
        lafs1: LAFs of image 1, shape :math:`(1, N, 2, 3)`.
        lafs2: LAFs of image 2, shape :math:`(1, N, 2, 3)`.
        tent_idxs: Tentative match indexes (query, train) of shape :math:`(M, 2)`.
        img1: First image, path/tensor/numpy array.
        img2: Second image, path/tensor/numpy array.
        inlier_mask: Boolean array/list of length M marking inliers among
            ``tent_idxs``.
        draw_dict: Drawing options. Keys: ``inlier_color`` (RGB tuple or
            None to skip drawing inliers), ``reprojected_color`` (RGB tuple
            or None to skip drawing the reprojected LAFs), ``vertical``
            (bool, stack images vertically instead of horizontally).
        H: Homography of shape :math:`(3, 3)` used to reproject LAFs between
            the two images.
        fig: Optional existing matplotlib figure to draw on.
        ax: Optional existing matplotlib axes to draw on.
        return_fig_ax: If True, return the figure and axes instead of None.

    Returns:
        Tuple of ``(fig, ax)`` if ``return_fig_ax`` is True, otherwise None.
    """
    import kornia as K
    import kornia.feature as KF
    import kornia.geometry as KG
    from kornia_moons.feature import to_numpy_image, to_np, to_torch
    inlier_mask = np.array(inlier_mask).reshape(-1)
    lafs1 = to_torch(lafs1).detach().cpu().float()
    lafs2 = to_torch(lafs2).detach().cpu().float()
    tent_idxs = to_torch(tent_idxs).detach().cpu().long()
    img1 = to_numpy_image(img1)
    img2 = to_numpy_image(img2)
    img1, img2 = _promote_to_matching_channels(img1, img2)

    h,w = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    lafs1_in2 = KF.perspective_transform_lafs(torch.from_numpy(H).float()[None],
                                              lafs1)
    lafs2_in1 = KF.perspective_transform_lafs(torch.inverse(torch.from_numpy(H).float()[None]),
                                              lafs2)
    xy1 = KF.get_laf_center(lafs1).reshape(-1, 2)
    xy2 = KF.get_laf_center(lafs2).reshape(-1, 2)
    
    # If we have no axes, create one
    if (fig is None and ax is None):
        fig, ax = plt.subplots(1,1, figsize=(20,10))
    if (fig is not None and ax is None):
        ax = fig.add_axes([0, 0, 1, 1])

    tent_corrs_in1 = torch.stack([xy1[tent_idxs[:,0]],
                                  KF.get_laf_center(lafs2_in1).reshape(-1, 2)[tent_idxs[:,1]]])
    
    tent_corrs_in2 = torch.stack([KF.get_laf_center(lafs1_in2).reshape(-1, 2)[tent_idxs[:,0]],
                                  xy2[tent_idxs[:,1]]])
    
    try:
        vert = draw_dict['vertical']
    except:
        vert = False
    if vert:
        tent_corrs_in2[:,:,1]+=h # shift for the 2nd image
    else:
        tent_corrs_in2[:,:,0]+=w # shift for the 2nd image
    # Prepraring canvas
    if not vert:
        if len(img1.shape) == 3:
            new_shape = (max(h, h2), w + w2, img1.shape[2])
        elif len(img1.shape) == 2:
            new_shape = (max(h, h2), w + w2)
    else:
        if len(img1.shape) == 3:
            new_shape = (h + h2,  max(w, w2), img1.shape[2])
        elif len(img1.shape) == 2:
            new_shape = (h + h2,  max(w, w2))        
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    if not vert:
        new_img[0:h, 0:w] = img1
        new_img[0:h2, w:w + w2] = img2
    else:
        new_img[0:h, 0:w] = img1
        new_img[h:h+h2, 0:w2] = img2

    x1,y1 = to_np(KF.laf.get_laf_pts_to_draw(lafs1, 0))
    x2,y2 = to_np(KF.laf.get_laf_pts_to_draw(lafs2, 0))
    
    x1in2, y1in2 = to_np(KF.laf.get_laf_pts_to_draw(lafs1_in2, 0))
    x2in1, y2in1 = to_np(KF.laf.get_laf_pts_to_draw(lafs2_in1, 0))
    
    if vert:
        y2+=h
        y1in2+=h
    else:
        x2+=w
        x1in2+=w
        
    try:
        ic = draw_dict['inlier_color']
    except:
        ic = None
    if (ic is not None) and (inlier_mask is not None):
        inlier_mask = inlier_mask > 0
        ax.plot(tent_corrs_in1[..., inlier_mask, 0], tent_corrs_in1[...,inlier_mask, 1], color=ic)#, markersize=15, marker='x')
        ax.plot(tent_corrs_in2[..., inlier_mask, 0],
                tent_corrs_in2[...,inlier_mask, 1], color=ic)#, markersize=15, marker='x')
        ax.plot(x1[:, tent_idxs[inlier_mask,0]], y1[:, tent_idxs[inlier_mask,0]], color=ic)#, markersize=15)
        ax.plot(x2[:, tent_idxs[inlier_mask,1]], y2[:, tent_idxs[inlier_mask,1]], color=ic)# markersize=15)
    try:
        rc = draw_dict['reprojected_color']
    except:
        rc = None
    if rc is not None:
        inlier_mask = inlier_mask > 0
        ax.plot(x1in2[:, tent_idxs[inlier_mask,0]], y1in2[:, tent_idxs[inlier_mask,0]], color=rc)#, markersize=15)
        ax.plot(x2in1[:, tent_idxs[inlier_mask,1]], y2in1[:, tent_idxs[inlier_mask,1]], color=rc)#, markersize=15)

    # Finally clip the image
    ax.imshow(new_img)
    if not vert:
        ax.set_xlim([0,w+w2])
        ax.set_ylim([max(h,h2),0])
        ax.margins(0,0)
    else:
        ax.set_xlim([0,max(w,w2)])
        ax.set_ylim([h+h2, 0])
        ax.margins(0,0)
    if return_fig_ax : return fig, ax
    return 

def draw_epipolar_errors_in_single_image(kp1: np.array, kp2: np.array,
                                         Fm1to2: np.array, img,
                                          draw_dict={"error_color": (1, 0.2, 0.2),
                                                     "feature_color": (0.2, 0.5, 1),
                                                     "figsize": (10,10),
                                                     "markersize": 8},
                                         img_index: int = 2,
                                         ax: Optional = None,
                                         title = None):
    """This function draws epipolar errors in single image

    Args:
        kp1: Keypoints in image 1, array of shape :math:`(N, 2)`.
        kp2: Keypoints in image 2, array of shape :math:`(N, 2)`.
        Fm1to2: Fundamental matrix mapping points from image 1 to image 2,
            of shape :math:`(3, 3)`.
        img: Image to draw on (the one indexed by ``img_index``), path,
            tensor, or numpy array.
        draw_dict: Drawing options. Keys: ``error_color`` (RGB tuple for the
            error lines and estimated projections), ``feature_color`` (RGB
            tuple for the ground-truth points), ``figsize`` (figure size
            used when ``ax`` is None), ``markersize`` (marker size for the
            scatter points).
        img_index: Which image (1 or 2) ``kp1``/``kp2``/``img`` refer to;
            when 1, ``kp1``/``kp2`` and ``Fm1to2`` are swapped/transposed
            accordingly.
        ax: Optional existing matplotlib axes to draw on.
        title: Optional plot title.

    Returns:
        The matplotlib axes the errors were drawn on.
    """
    from kornia.geometry.epipolar import get_closest_point_on_epipolar_line
    from kornia_moons.feature import to_numpy_image, cv2_matches_from_kornia, to_np, to_torch
    
    img = to_numpy_image(img)
    pts1 = to_torch(kp1)[None]
    pts2 = to_torch(kp2)[None]
    Fm = to_torch(Fm1to2)
    if len(Fm.shape) == 2:
        Fm = Fm[None]
    assert img_index in [1,2]
    if img_index == 1:
        pts1, pts2 = pts2, pts1
        Fm = Fm.transpose(1, 2)
    closest_pts = get_closest_point_on_epipolar_line(pts1, pts2, Fm)
    
    # If we have no axes, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=draw_dict["figsize"])
    ax.imshow(img)
    plt.scatter(pts2[0, :, 0].numpy(),
                pts2[0, :, 1].numpy(),
                s=draw_dict["markersize"]**2,
                label='GT points',
                color = draw_dict['feature_color'])
    plt.scatter(closest_pts[0, :, 0].numpy(),
                closest_pts[0, :, 1].numpy(),
                s=draw_dict["markersize"]**2,
                label='est. projection',
                color = draw_dict['error_color'], marker='x')

    plt.plot(torch.cat([pts2[0, :, 0].view(1,-1),
                        closest_pts[0, :, 0].view(1,-1)],dim=0).numpy(),
             torch.cat([pts2[0, :, 1].view(1,-1),
                        closest_pts[0, :, 1].view(1,-1)], dim=0).numpy(),
             label='error', color=draw_dict['error_color'])

    plt.legend(['GT points','est. projection','errors'])
    if title is not None:
        plt.title(title)
    return ax

def plot_images(imgs, titles=None, cmaps="gray", dpi=100, size=6, pad=0.5):
    """Plot a horizontal strip of images and establish the figure that
    :func:`plot_lines` and :func:`plot_color_line_matches` draw on.

    Args:
        imgs: List of NumPy or PyTorch images, RGB :math:`(H, W, 3)` or
            mono :math:`(H, W)`.
        titles: Optional list of strings, as titles for each image.
        cmaps: Colormap, or list of colormaps (one per image), used for
            monochrome images.
        dpi: Figure resolution in dots per inch.
        size: Width in inches allotted per image; the figure height is ``3/4`` of it.
        pad: Padding passed to ``fig.tight_layout``.

    Returns:
        None. The created figure and axes become the current matplotlib
        figure, ready for :func:`plot_lines` to draw on.
    """
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n
    figsize = (size * n, size * 3 / 4) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_lines(lines, line_colors="orange", point_colors="cyan", ps=4, lw=2, indices=(0, 1)):
    """Plot line segments and their endpoints on the current figure's axes.

    Must be called after :func:`plot_images`, which creates the figure and
    axes this function draws on.

    Args:
        lines: List of arrays of shape :math:`(N, 2, 2)`, one per image,
            with ``[..., 0]`` the y coordinate and ``[..., 1]`` the x
            coordinate of each endpoint.
        line_colors: Color, or list of colors (one per image), for the
            line segments.
        point_colors: Color, or list of colors (one per image), for the
            endpoints.
        ps: Size of the endpoint markers, in points.
        lw: Line width, in points.
        indices: Indexes of the current figure's axes to draw the lines on.

    Returns:
        None. The lines and points are added to the existing axes in place.
    """
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    if not isinstance(line_colors, list):
        line_colors = [line_colors] * len(lines)
    if not isinstance(point_colors, list):
        point_colors = [point_colors] * len(lines)

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines and junctions
    for a, l, lc, pc in zip(axes, lines, line_colors, point_colors):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D(
                (l[i, 1, 1], l[i, 0, 1]),
                (l[i, 1, 0], l[i, 0, 0]),
                zorder=1,
                c=lc,
                linewidth=lw,
            )
            a.add_line(line)
        pts = l.reshape(-1, 2)
        a.scatter(pts[:, 1], pts[:, 0], c=pc, s=ps, linewidths=0, zorder=2)


def plot_color_line_matches(lines, lw=2, indices=(0, 1)):
    """Plot matched line segments on the current figure's axes, giving each
    match its own color.

    Must be called after :func:`plot_images`, which creates the figure and
    axes this function draws on. Used to visualize SOLD2-style line matches
    between two images.

    Args:
        lines: List of arrays of shape :math:`(N, 2, 2)`, one per image,
            with ``[..., 0]`` the y coordinate and ``[..., 1]`` the x
            coordinate of each endpoint. Line ``i`` in each image is drawn
            with the same random color, so index ``i`` must denote the same
            match across images.
        lw: Line width, in points.
        indices: Indexes of the current figure's axes to draw the lines on.

    Returns:
        None. The lines are added to the existing axes in place.
    """
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    n_lines = len(lines[0])

    cmap = plt.get_cmap("nipy_spectral", lut=n_lines)
    colors = np.array([mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)])

    np.random.shuffle(colors)

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines
    for a, l in zip(axes, lines):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D(
                (l[i, 1, 1], l[i, 0, 1]),
                (l[i, 1, 0], l[i, 0, 0]),
                zorder=1,
                c=colors[i],
                linewidth=lw,
            )
            a.add_line(line)
