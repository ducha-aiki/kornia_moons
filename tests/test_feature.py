import warnings
from pathlib import Path

import cv2
import kornia
import numpy as np
import pytest
import torch
from kornia.image import image_to_tensor

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
    det = OpenCVDetectorWithAffNetKornia(cv2.SIFT_create(500), make_upright=True, max_kpts=50)
    lafs, resp = det(timg)
    assert lafs.dim() == 4 and lafs.shape[-2:] == (2, 3)
    assert lafs.shape[1] == 50


def test_to_torch_from_list():
    t = to_torch([1.0, 2.0, 3.0])
    assert isinstance(t, torch.Tensor)
    assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))


def test_to_torch_rejects_unsupported_type():
    with pytest.raises(TypeError):
        to_torch(42)


@pytest.mark.parametrize("max_kpts", [-1, 0])
def test_detector_max_kpts_default_keeps_all(img, max_kpts):
    # regression: make_upright=True with default max_kpts=-1 used to drop the last keypoint
    expected = len(make_keypoints_upright(list(cv2.SIFT_create(500).detect(img, None))))
    timg255 = image_to_tensor(img, False).float()  # 0-255 range: exact uint8 round-trip
    lafs, resp = OpenCVDetectorKornia(
        cv2.SIFT_create(500), make_upright=True, max_kpts=max_kpts)(timg255)
    assert lafs.shape[1] == expected


@pytest.mark.parametrize("upright", [False, True])
def test_detector_max_kpts_truncates(img, upright):
    timg = image_to_tensor(img, False).float() / 255.0
    lafs, resp = OpenCVDetectorKornia(
        cv2.SIFT_create(500), make_upright=upright, max_kpts=10)(timg)
    assert lafs.shape[1] == 10
    assert resp.shape[1] == 10


def test_detector_max_kpts_keeps_strongest(img):
    all_resps = sorted(
        (k.response for k in cv2.SIFT_create(500).detect(img, None)), reverse=True)
    timg255 = image_to_tensor(img, False).float()  # 0-255 range: exact uint8 round-trip
    _, resp = OpenCVDetectorKornia(cv2.SIFT_create(500), max_kpts=10)(timg255)
    got = sorted(resp.view(-1).tolist(), reverse=True)
    assert got == pytest.approx(all_resps[:10])


def _top_half_mask(timg):
    h, w = timg.shape[2], timg.shape[3]
    mask = torch.zeros(1, 1, h, w, dtype=torch.uint8)
    mask[..., : h // 2, :] = 255
    return mask


def test_detector_respects_mask(img):
    timg = image_to_tensor(img, False).float() / 255.0
    lafs, resp = OpenCVDetectorKornia(cv2.SIFT_create(500))(timg, _top_half_mask(timg))
    assert lafs.shape[1] > 0
    ys = lafs[0, :, 1, 2]
    assert (ys <= timg.shape[2] // 2 + 1).all()


def test_feature_respects_mask(img):
    timg = image_to_tensor(img, False).float() / 255.0
    lafs, resp, descs = OpenCVFeatureKornia(cv2.SIFT_create(500))(timg, _top_half_mask(timg))
    assert lafs.shape[1] > 0
    ys = lafs[0, :, 1, 2]
    assert (ys <= timg.shape[2] // 2 + 1).all()


def test_affnet_detector_respects_mask(img):
    timg = image_to_tensor(img, False).float() / 255.0
    det = OpenCVDetectorWithAffNetKornia(cv2.SIFT_create(200))
    lafs, resp = det(timg, _top_half_mask(timg))
    assert lafs.shape[1] > 0
    ys = lafs[0, :, 1, 2]
    assert (ys <= timg.shape[2] // 2 + 1).all()


def test_detector_no_detections_returns_empty():
    lafs, resp = OpenCVDetectorKornia(cv2.SIFT_create())(torch.zeros(1, 3, 64, 64))
    assert lafs.shape == (1, 0, 2, 3)
    assert resp.shape == (1, 0, 1)


def test_feature_no_detections_returns_empty():
    sift = cv2.SIFT_create()
    lafs, resp, descs = OpenCVFeatureKornia(sift)(torch.zeros(1, 3, 64, 64))
    assert lafs.shape == (1, 0, 2, 3)
    assert resp.shape == (1, 0, 1)
    assert descs.shape == (1, 0, sift.descriptorSize())


def test_to_numpy_image_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        to_numpy_image("/nonexistent/no_such_file.png")


def test_no_kornia_utils_deprecation_warning(img):
    # kornia 0.8.3 deprecated kornia.utils.{tensor_to_image,image_to_tensor}
    # in favor of kornia.image; the wrappers must not hit the deprecated path
    timg = image_to_tensor(img, False).float() / 255.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        OpenCVDetectorKornia(cv2.SIFT_create(50))(timg)
    deprecated = [w for w in caught if "kornia.image" in str(w.message)]
    assert not deprecated
