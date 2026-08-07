
__all__ = ['to_numpy_image', 'to_torch', 'to_np', 'laf_from_opencv_kpts', 'opencv_kpts_from_laf', 'laf_from_opencv_ORB_kpts',
           'laf_from_opencv_SIFT_kpts', 'opencv_SIFT_kpts_from_laf', 'opencv_ORB_kpts_from_laf',
           'cv2_matches_from_kornia', 'kornia_matches_from_cv2', 'make_keypoints_upright', 'OpenCVDetectorKornia',
           'OpenCVFeatureKornia', 'OpenCVDetectorWithAffNetKornia']

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

from .viz import visualize_LAF, draw_LAF_matches

def to_numpy_image(img: Union[str, np.array, torch.Tensor]):
    """Load or convert an image into an RGB numpy array.

    Args:
        img: Image file path, `(B, C, H, W)` / `(C, H, W)` tensor, or
            numpy array. Paths are read with OpenCV and converted to RGB.

    Returns:
        The image as an ``np.ndarray``.

    Raises:
        TypeError: If ``img`` is not a str, np.ndarray, or torch.Tensor.
        FileNotFoundError: If ``img`` is a path that cannot be read.
    """
    if type(img) is str:
        img_bgr = cv2.imread(img)
        if img_bgr is None:
            raise FileNotFoundError(f'Could not read image: {img}')
        img_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(img, torch.Tensor):
        img_out = tensor_to_image(img)
    elif isinstance(img, np.ndarray):
        img_out = img
    else:
        raise TypeError('img should be str, np.array or torch.Tensor')
    return img_out

def to_torch(x: Union[List, np.array, torch.Tensor]):
    """Convert a Python list or numpy array to a torch tensor, passing tensors through.

    Args:
        x: List, numpy array, or torch tensor to convert.

    Returns:
        ``x`` as a ``torch.Tensor``.

    Raises:
        TypeError: If ``x`` is not a list, np.ndarray, or torch.Tensor.
    """
    if isinstance(x, list):
        x_out = torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        x_out = x
    elif isinstance(x, np.ndarray):
        x_out = torch.from_numpy(x)
    else:
        raise TypeError('img should be List, np.array or torch.Tensor')
    return x_out

def to_np(array: Union[List, Tuple, np.ndarray, torch.Tensor]):
    """Convert a tensor or sequence to a numpy array.

    Args:
        array: List, tuple, numpy array, or torch tensor to convert.
            Tensors are detached and moved to CPU before conversion.

    Returns:
        ``array`` as an ``np.ndarray``.
    """
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)

def laf_from_opencv_kpts(kpts: List[cv2.KeyPoint], 
                         mrSize: float=6.0,
                         device: torch.device=torch.device('cpu'),
                         with_resp: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Convert OpenCV keypoints into kornia local affine frames (LAFs).

    Args:
        kpts: List of N OpenCV keypoints.
        mrSize: Measurement-region scale multiplier applied to the keypoint
            size. Use 6.0 for SIFT-like detectors and 1.0 for ORB-like
            detectors, matching the OpenCV description-region conventions.
        device: Device to place the output tensors on.
        with_resp: If True, also return the keypoint responses.

    Returns:
        LAFs of shape :math:`(1, N, 2, 3)`, and, if ``with_resp`` is True,
        a tuple of the LAFs and responses of shape :math:`(1, N, 1)`.

    Example:
        >>> img = cv2.imread('data/strahov.png')
        >>> kps, descs = cv2.ORB_create(500).detectAndCompute(img, None)
        >>> lafs, resp = laf_from_opencv_kpts(kps, mrSize=1.0, with_resp=True)
    """
    N = len(kpts)
    xy = torch.tensor([(x.pt[0], x.pt[1]) for x in kpts ], device=device, dtype=torch.float).view(1, N, 2)
    scales = torch.tensor([(mrSize * x.size) for x in kpts ], device=device, dtype=torch.float).view(1, N, 1, 1)
    angles = torch.tensor([(-x.angle) for x in kpts ], device=device, dtype=torch.float).view(1, N, 1)
    laf = kornia.feature.laf_from_center_scale_ori(xy, scales, angles).reshape(1, -1, 2, 3)
    if not with_resp:
        return laf.reshape(1, -1, 2, 3)
    resp = torch.tensor([x.response for x in kpts], device=device, dtype=torch.float).view(1, N, 1)
    return laf, resp


def opencv_kpts_from_laf(lafs: torch.Tensor,
                         mrSize: float = 1.0, 
                         resps: Optional[torch.Tensor] = None) -> List[cv2.KeyPoint]:
    """Convert kornia local affine frames (LAFs) back to OpenCV keypoints.

    Args:
        lafs: LAFs of shape :math:`(1, N, 2, 3)`.
        mrSize: Measurement-region scale multiplier to divide out of the LAF
            scale. Must match the ``mrSize`` used to create the LAFs (6.0
            for SIFT-like, 1.0 for ORB-like detectors).
        resps: Optional keypoint responses of shape :math:`(1, N, 1)`.

    Returns:
        List of N OpenCV keypoints.

    Note:
        A LAF does not store the OpenCV ``octave`` and ``class_id`` fields,
        so keypoints converted back get the defaults (``octave=0``,
        ``class_id=-1``). OpenCV descriptor extractors still work on such
        keypoints.

    Example:
        >>> lafs, resp = laf_from_opencv_kpts(kps, mrSize=1.0, with_resp=True)
        >>> kps_back = opencv_kpts_from_laf(lafs, 1.0, resp)
    """
    XY = kornia.feature.get_laf_center(lafs)
    S = kornia.feature.get_laf_scale(lafs)
    Ang = kornia.feature.get_laf_orientation(lafs)
    if resps is not None:
        assert resps.shape[:2] == lafs.shape[:2]
        cv_kpts = [cv2.KeyPoint(xy[0].item(), xy[1].item(), s.item()/mrSize, -a.item(), r.item()) 
                   for xy, s, a, r in zip(XY.view(-1,2), S.view(-1, 1), Ang.view(-1, 1), resps.view(-1, 1))]
    else:
        cv_kpts = [cv2.KeyPoint(xy[0].item(), xy[1].item(), s.item()/ mrSize, -a.item()) 
                   for xy, s, a in zip(XY.view(-1,2), S.view(-1, 1), Ang.view(-1, 1))]        
    return cv_kpts

def laf_from_opencv_ORB_kpts(kpts: List[cv2.KeyPoint], 
                             device: torch.device=torch.device('cpu'),
                             with_resp: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Convert ORB keypoints into kornia LAFs using the ORB `mrSize` convention.

    Args:
        kpts: List of N OpenCV ORB keypoints.
        device: Device to place the output tensors on.
        with_resp: If True, also return the keypoint responses.

    Returns:
        LAFs of shape :math:`(1, N, 2, 3)`, and, if ``with_resp`` is True,
        a tuple of the LAFs and responses of shape :math:`(1, N, 1)`.
    """
    return laf_from_opencv_kpts(kpts, 1.0, device, with_resp)

def laf_from_opencv_SIFT_kpts(kpts: List[cv2.KeyPoint], 
                              device: torch.device=torch.device('cpu'),
                              with_resp: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Convert SIFT keypoints into kornia LAFs using the SIFT `mrSize` convention.

    Args:
        kpts: List of N OpenCV SIFT keypoints.
        device: Device to place the output tensors on.
        with_resp: If True, also return the keypoint responses.

    Returns:
        LAFs of shape :math:`(1, N, 2, 3)`, and, if ``with_resp`` is True,
        a tuple of the LAFs and responses of shape :math:`(1, N, 1)`.
    """
    return laf_from_opencv_kpts(kpts, 6.0, device, with_resp)


def opencv_SIFT_kpts_from_laf(lafs, resps: Optional[torch.Tensor] = None):
    """Convert kornia LAFs to OpenCV keypoints using the SIFT `mrSize` convention.

    Args:
        lafs: LAFs of shape :math:`(1, N, 2, 3)`.
        resps: Optional keypoint responses of shape :math:`(1, N, 1)`.

    Returns:
        List of N OpenCV keypoints.
    """
    return opencv_kpts_from_laf(lafs, 6.0, resps);

def opencv_ORB_kpts_from_laf(lafs, resps: Optional[torch.Tensor] = None):
    """Convert kornia LAFs to OpenCV keypoints using the ORB `mrSize` convention.

    Args:
        lafs: LAFs of shape :math:`(1, N, 2, 3)`.
        resps: Optional keypoint responses of shape :math:`(1, N, 1)`.

    Returns:
        List of N OpenCV keypoints.
    """
    return opencv_kpts_from_laf(lafs, 1.0, resps);

def cv2_matches_from_kornia(match_dists: torch.Tensor, match_idxs: torch.Tensor) -> List[cv2.DMatch]:
    """Convert kornia match distances and indexes to a list of cv2.DMatch.

    Args:
        match_dists: Match distances of shape :math:`(N, 1)`.
        match_idxs: Match indexes (query, train) of shape :math:`(N, 2)`.

    Returns:
        List of N ``cv2.DMatch`` objects.

    Example:
        >>> dists, idxs = kornia.feature.match_nn(descs1, descs2)
        >>> cv2_matches = cv2_matches_from_kornia(dists, idxs)
    """
    return [cv2.DMatch(idx[0].item(), idx[1].item(), d.item()) for idx, d in zip (match_idxs, match_dists)]

def kornia_matches_from_cv2(cv2_matches, device=torch.device('cpu')):
    """Convert a list of cv2.DMatch to kornia match distance and index tensors.

    Args:
        cv2_matches: List of N ``cv2.DMatch`` objects.
        device: Device to place the output tensors on.

    Returns:
        Tuple of match distances of shape :math:`(N, 1)` (float) and match
        indexes (query, train) of shape :math:`(N, 2)` (long).

    Example:
        >>> cv2_matches = cv2_matches_from_kornia(match_dists, match_idxs)
        >>> match_dists_back, match_idxs_back = kornia_matches_from_cv2(cv2_matches)
    """
    num_matches = len(cv2_matches)
    match_dists = torch.zeros(num_matches, 1, device=device, dtype=torch.float)
    match_idxs = torch.zeros(num_matches, 2, device=device, dtype=torch.long)
    for i, m in enumerate(cv2_matches):
        match_dists[i, 0] = m.distance
        match_idxs[i, 0] = m.queryIdx
        match_idxs[i, 1] = m.trainIdx
    return match_dists, match_idxs

def make_keypoints_upright(kpts):
    """Deduplicate keypoints by response and zero their orientations.

    Keypoints are considered duplicates when they share the same response
    as the previous keypoint in the input list. Note that this mutates the
    input keypoints in place (setting ``angle = 0``).

    Args:
        kpts: List of OpenCV keypoints, assumed sorted by response.

    Returns:
        The deduplicated, upright keypoints sorted by response, descending.
    """
    unique_kp = []
    for i, kk in enumerate(kpts):
        if i > 0:
            if kk.response == kpts[i - 1].response:
                continue
        kk.angle = 0
        unique_kp.append(kk)
    top_resps = np.array([kk.response for kk in unique_kp])
    idxs = np.argsort(top_resps)[::-1]
    return [unique_kp[i] for i in idxs]


def _image_to_numpy_uint8(x: torch.Tensor) -> np.ndarray:
    # heuristic: tensors whose max value is below 2.0 are assumed to be
    # in [0, 1] range and are scaled by 255; genuinely dark [0, 255]
    # images with max < 2 would be misread — pre-scale those to [0, 1]
    if x.max() < 2.0:
        return (255 * tensor_to_image(x)).astype(np.uint8)
    return tensor_to_image(x).astype(np.uint8)


def _mask_to_numpy(mask: Optional[Union[torch.Tensor, np.ndarray]]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = tensor_to_image(mask)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (255 * (mask > 0)).astype(np.uint8)


class OpenCVDetectorKornia(nn.Module):
    """Wrap an OpenCV detector into a kornia-compatible detection module.

    The wrapped detector runs on the numpy image under the hood; the module
    accepts and returns torch tensors, so it can slot into kornia pipelines.

    Args:
        opencv_detector: Any OpenCV detector exposing ``detect`` (e.g.
            ``cv2.SIFT_create()``).
        mrSize: Measurement-region scale multiplier (6.0 for SIFT, 1.0 for ORB).
        make_upright: If True, deduplicate keypoints and zero their angles.
        max_kpts: Keep at most this many of the highest-response keypoints,
            applied whether or not ``make_upright`` is set. Non-positive
            values (the default ``-1``) keep all keypoints.

    Example:
        >>> detector = OpenCVDetectorKornia(cv2.SIFT_create(500))
        >>> timg = kornia.image_to_tensor(img, False).float() / 255.
        >>> lafs, resps = detector(timg)
    """
    def __init__(self, opencv_detector, mrSize: float = 6.0, make_upright = False, max_kpts = -1):
        super().__init__()
        self.features = opencv_detector
        self.mrSize = mrSize
        self.make_upright = make_upright
        self.max_kpts = max_kpts
        
    def forward(self, x:torch.Tensor, mask=None):
        """Detect keypoints on a batched image tensor.

        Args:
            x: Image tensor of shape :math:`(1, C, H, W)`, values in
                :math:`[0, 1]` float or :math:`[0, 255]`.
            mask: Optional detection mask, :math:`(1, 1, H, W)` tensor or
                :math:`(H, W)` array; keypoints are detected where nonzero.

        Returns:
            LAFs of shape :math:`(1, N, 2, 3)` and responses :math:`(1, N, 1)`.
        """
        img_np = _image_to_numpy_uint8(x)
        kpts = self.features.detect(img_np, _mask_to_numpy(mask))
        if self.make_upright:
            kpts = make_keypoints_upright(kpts)
        if self.max_kpts > 0:
            kpts = sorted(kpts, key=lambda k: k.response, reverse=True)[:self.max_kpts]
        lafs, resp = laf_from_opencv_kpts(kpts, mrSize=self.mrSize, with_resp=True, device=x.device)
        return lafs, resp

    
class OpenCVFeatureKornia(nn.Module):
    """Wrap an OpenCV detect-and-describe pipeline for kornia.

    The wrapped feature (e.g. ``cv2.SIFT_create()``) runs ``detectAndCompute``
    on the numpy image under the hood; the module accepts and returns torch
    tensors, so it can slot into kornia pipelines.

    Args:
        opencv_detector: Any OpenCV feature exposing ``detectAndCompute``.
        mrSize: Measurement-region scale multiplier (6.0 for SIFT, 1.0 for ORB).

    Example:
        >>> feature = OpenCVFeatureKornia(cv2.SIFT_create(500))
        >>> timg = kornia.image_to_tensor(img, False).float() / 255.
        >>> lafs, resps, descs = feature(timg)
    """
    def __init__(self, opencv_detector, mrSize: float = 6.0):
        super().__init__()
        self.features = opencv_detector
        self.mrSize = mrSize
        
    def forward(self, x:torch.Tensor, mask=None):
        """Detect and describe keypoints on a batched image tensor.

        Args:
            x: Image tensor of shape :math:`(1, C, H, W)`, values in
                :math:`[0, 1]` float or :math:`[0, 255]`.
            mask: Optional detection mask, :math:`(1, 1, H, W)` tensor or
                :math:`(H, W)` array; keypoints are detected where nonzero.

        Returns:
            LAFs of shape :math:`(1, N, 2, 3)`, responses of shape
            :math:`(1, N, 1)`, and descriptors of shape :math:`(1, N, D)`.
        """
        img_np = _image_to_numpy_uint8(x)
        kpts, descs = self.features.detectAndCompute(img_np, _mask_to_numpy(mask))
        if descs is None:
            descs = np.zeros((0, self.features.descriptorSize()), dtype=np.float32)
        lafs, resp = laf_from_opencv_kpts(kpts, mrSize=self.mrSize, with_resp=True, device=x.device)
        return lafs, resp, torch.from_numpy(descs).to(device=x.device)[None]


class OpenCVDetectorWithAffNetKornia(nn.Module):
    """Combine an OpenCV detector with kornia's AffNet affine-shape refinement.

    The wrapped detector locates keypoints on the numpy image, and kornia's
    ``LAFAffNetShapeEstimator`` then refines the elliptical shape of each
    LAF. Note that AffNet's pretrained weights are downloaded on first use,
    and the original keypoint orientation is preserved after refinement.

    Args:
        opencv_detector: Any OpenCV detector exposing ``detect``.
        make_upright: If True, deduplicate keypoints and zero their angles.
        mrSize: Measurement-region scale multiplier (6.0 for SIFT, 1.0 for ORB).
        max_kpts: Keep at most this many of the highest-response keypoints,
            applied whether or not ``make_upright`` is set. Non-positive
            values (the default ``-1``) keep all keypoints.

    Example:
        >>> detector = OpenCVDetectorWithAffNetKornia(cv2.SIFT_create(500))
        >>> timg = kornia.image_to_tensor(img, False).float() / 255.
        >>> lafs, resps = detector(timg)
    """
    def __init__(self, opencv_detector, make_upright = False, mrSize: float = 6.0, max_kpts = -1):
        super().__init__()
        self.features = opencv_detector
        self.mrSize = mrSize
        self.affnet = kornia.feature.LAFAffNetShapeEstimator(True).eval()
        self.make_upright = make_upright
        self.max_kpts = max_kpts
        
    def forward(self, x:torch.Tensor, mask=None):
        """Detect keypoints and refine their affine shape with AffNet.

        Args:
            x: Image tensor of shape :math:`(1, C, H, W)`, values in
                :math:`[0, 1]` float or :math:`[0, 255]`.
            mask: Optional detection mask, :math:`(1, 1, H, W)` tensor or
                :math:`(H, W)` array; keypoints are detected where nonzero.

        Returns:
            LAFs of shape :math:`(1, N, 2, 3)` and responses :math:`(1, N, 1)`.
        """
        self.affnet = self.affnet.to(x.device)
        img_np = _image_to_numpy_uint8(x)
        kpts = self.features.detect(img_np, _mask_to_numpy(mask))
        if self.make_upright:
            kpts = make_keypoints_upright(kpts)
        if self.max_kpts > 0:
            kpts = sorted(kpts, key=lambda k: k.response, reverse=True)[:self.max_kpts]
        lafs, resp = laf_from_opencv_kpts(kpts, mrSize=self.mrSize, with_resp=True, device=x.device)
        ori = kornia.feature.get_laf_orientation(lafs)
        lafs = self.affnet(lafs, x.mean(dim=1, keepdim=True))
        lafs = kornia.feature.set_laf_orientation(lafs, ori)
        return lafs, resp
