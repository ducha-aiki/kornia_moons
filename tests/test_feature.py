from pathlib import Path

import cv2
import kornia
import numpy as np
import pytest
import torch
from kornia.utils import image_to_tensor

from kornia_moons.feature import (
    OpenCVDetectorKornia,
    OpenCVDetectorWithAffNetKornia,
    OpenCVFeatureKornia,
    cv2_matches_from_kornia,
    kornia_matches_from_cv2,
    laf_from_opencv_ORB_kpts,
    laf_from_opencv_SIFT_kpts,
    laf_from_opencv_kpts,
    make_keypoints_upright,
    opencv_ORB_kpts_from_laf,
    opencv_SIFT_kpts_from_laf,
    opencv_kpts_from_laf,
    to_np,
    to_numpy_image,
    to_torch,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STRAHOV = str(DATA_DIR / "strahov.png")


@pytest.fixture(scope="module")
def img():
    return cv2.cvtColor(cv2.imread(STRAHOV), cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def orb_kpts(img):
    det = cv2.ORB_create(500)
    return det.detectAndCompute(img, None)


@pytest.fixture(scope="module")
def sift_kpts(img):
    det = cv2.SIFT_create(500)
    return det.detectAndCompute(img, None)


def test_to_numpy_image_from_path():
    assert isinstance(to_numpy_image(STRAHOV), np.ndarray)


def test_to_numpy_image_from_tensor(img):
    assert isinstance(to_numpy_image(image_to_tensor(img, False)), np.ndarray)


def test_to_numpy_image_rejects_bad_type():
    with pytest.raises(TypeError):
        to_numpy_image(42)


def test_to_torch_and_to_np():
    arr = np.zeros((3, 2), dtype=np.float32)
    t = to_torch(arr)
    assert isinstance(t, torch.Tensor)
    assert isinstance(to_np(t), np.ndarray)


def test_orb_laf_roundtrip(orb_kpts):
    kps, _ = orb_kpts
    lafs, resp = laf_from_opencv_kpts(kps, 1.0, with_resp=True)
    assert lafs.shape == (1, len(kps), 2, 3)
    assert resp.shape == (1, len(kps), 1)
    kps_back = opencv_kpts_from_laf(lafs, 1.0, resp)
    assert len(kps_back) == len(kps)
    for a, b in zip(kps, kps_back):
        assert a.pt[0] == pytest.approx(b.pt[0], abs=1e-3)
        assert a.pt[1] == pytest.approx(b.pt[1], abs=1e-3)
        assert a.size == pytest.approx(b.size, rel=1e-3)


def test_sift_laf_helpers(sift_kpts):
    kps, _ = sift_kpts
    lafs, resp = laf_from_opencv_SIFT_kpts(kps, with_resp=True)
    assert lafs.shape == (1, len(kps), 2, 3)
    assert len(opencv_SIFT_kpts_from_laf(lafs, resp)) == len(kps)


def test_orb_laf_helpers(orb_kpts):
    kps, _ = orb_kpts
    lafs = laf_from_opencv_ORB_kpts(kps)
    assert len(opencv_ORB_kpts_from_laf(lafs)) == len(kps)


def test_matches_roundtrip(sift_kpts):
    _, descs = sift_kpts
    d = torch.from_numpy(descs).float()
    match_dists, match_idxs = kornia.feature.match_nn(d, d)
    cv2_matches = cv2_matches_from_kornia(match_dists, match_idxs)
    dists_back, idxs_back = kornia_matches_from_cv2(cv2_matches)
    assert torch.allclose(dists_back, match_dists)
    assert torch.allclose(idxs_back, match_idxs)


def test_make_keypoints_upright(img):
    # fresh detection: make_keypoints_upright mutates the keypoints it receives
    kps, _ = cv2.SIFT_create(500).detectAndCompute(img, None)
    upright = make_keypoints_upright(kps)
    assert 0 < len(upright) <= len(kps)
    assert all(k.angle == 0 for k in upright)


def test_opencv_detector_kornia(img):
    timg = image_to_tensor(img, False).float() / 255.0
    lafs, resp = OpenCVDetectorKornia(cv2.SIFT_create(500))(timg)
    assert lafs.dim() == 4 and lafs.shape[-2:] == (2, 3)
    assert resp.shape[:2] == lafs.shape[:2]


def test_opencv_feature_kornia(img):
    timg = image_to_tensor(img, False).float() / 255.0
    lafs, resp, descs = OpenCVFeatureKornia(cv2.SIFT_create(500))(timg)
    assert isinstance(descs, torch.Tensor)
    assert lafs.shape[1] == descs.shape[1]


def test_opencv_detector_affnet(img):
    # downloads AffNet weights on first run
    timg = image_to_tensor(img, False).float() / 255.0
    det = OpenCVDetectorWithAffNetKornia(cv2.SIFT_create(500), make_upright=True)
    lafs, resp = det(timg)
    assert lafs.dim() == 4 and lafs.shape[-2:] == (2, 3)
