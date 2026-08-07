import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import cv2
import kornia
import numpy as np
import pytest
import torch
from kornia.image import image_to_tensor

from kornia_moons.feature import laf_from_opencv_kpts
from kornia_moons.viz import (
    draw_LAF_matches,
    draw_LAF_matches_from_result_dict,
    epilines_to_start_end_points,
    plot_images,
    plot_lines,
    visualize_LAF,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STRAHOV = str(DATA_DIR / "strahov.png")


@pytest.fixture(scope="module")
def img():
    return cv2.cvtColor(cv2.imread(STRAHOV), cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def lafs_descs(img):
    kps, descs = cv2.ORB_create(500).detectAndCompute(img, None)
    lafs, resp = laf_from_opencv_kpts(kps, 1.0, with_resp=True)
    return lafs, resp, descs


@pytest.fixture(autouse=True)
def close_figures():
    yield
    matplotlib.pyplot.close("all")


def test_visualize_laf_runs(img, lafs_descs):
    lafs, _, _ = lafs_descs
    fig, ax = visualize_LAF(
        image_to_tensor(img, False), lafs, 0, "y", return_fig_ax=True
    )
    assert fig is not None and ax is not None


def test_epilines_to_start_end_points():
    epi = torch.tensor([[0.5, -1.0, 10.0], [-0.25, 1.0, 5.0]])
    pts = epilines_to_start_end_points(epi, 100, 200)
    assert pts.shape == (2, 2, 2)


def test_draw_laf_matches(img, lafs_descs):
    lafs, _, descs = lafs_descs
    d = torch.from_numpy(descs).float()
    _, idxs = kornia.feature.match_nn(d, d)
    fig, ax = draw_LAF_matches(lafs, lafs, idxs, img, img, return_fig_ax=True)
    assert fig is not None and ax is not None


def test_plot_images_and_lines(img):
    plot_images([img, img], ["a", "b"])
    segs = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]])
    plot_lines([segs, segs], indices=(0, 1))


def _result_dict(lafs, descs, with_F=False):
    d = torch.from_numpy(descs).float()
    _, idxs = kornia.feature.match_nn(d, d)
    rd = {
        "feat_dict1": {"lafs": lafs},
        "feat_dict2": {"lafs": lafs},
        "tents_dict": {"idxs": idxs},
        "inlier_mask": [True] * len(idxs),
    }
    if with_F:
        # a valid rank-2 fundamental-like matrix is enough for drawing epilines
        rd["F"] = np.array(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [1.0, 1.0, 0.0]]
        )
    return rd


def test_draw_laf_matches_from_result_dict(img, lafs_descs):
    lafs, _, descs = lafs_descs
    draw_LAF_matches_from_result_dict(_result_dict(lafs, descs), img, img)


def test_draw_laf_matches_from_result_dict_with_F(img, lafs_descs):
    lafs, _, descs = lafs_descs
    draw_LAF_matches_from_result_dict(_result_dict(lafs, descs, with_F=True), img, img)


def test_draw_laf_matches_numpy_inputs(img, lafs_descs):
    lafs, _, descs = lafs_descs
    d = torch.from_numpy(descs).float()
    _, idxs = kornia.feature.match_nn(d, d)
    fig, ax = draw_LAF_matches(
        lafs.numpy(), lafs.numpy(), idxs.numpy(), img, img, return_fig_ax=True
    )
    assert fig is not None and ax is not None


@pytest.mark.parametrize("gray_first", [True, False])
def test_draw_laf_matches_mixed_gray_color(img, lafs_descs, gray_first):
    lafs, _, descs = lafs_descs
    d = torch.from_numpy(descs).float()
    _, idxs = kornia.feature.match_nn(d, d)
    gray = img[..., 0]
    img1, img2 = (gray, img) if gray_first else (img, gray)
    fig, ax = draw_LAF_matches(lafs, lafs, idxs, img1, img2, return_fig_ax=True)
    assert fig is not None and ax is not None


def test_epilines_keep_dtype_and_device():
    epi = torch.tensor([[0.5, -1.0, 10.0], [-0.25, 1.0, 5.0]], dtype=torch.float64)
    pts = epilines_to_start_end_points(epi, 100, 200)
    assert pts.dtype == torch.float64
    assert pts.device == epi.device


needs_gpu = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs a non-CPU device"
)


@needs_gpu
def test_epilines_on_gpu():
    epi = torch.tensor([[0.5, -1.0, 10.0], [-0.25, 1.0, 5.0]], device="mps")
    pts = epilines_to_start_end_points(epi, 100, 200)
    assert pts.device == epi.device


@needs_gpu
def test_draw_laf_matches_gpu_tensors(img, lafs_descs):
    lafs, _, descs = lafs_descs
    d = torch.from_numpy(descs).float()
    _, idxs = kornia.feature.match_nn(d, d)
    F = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [1.0, 1.0, 0.0]])
    fig, ax = draw_LAF_matches(
        lafs.to("mps"), lafs.to("mps"), idxs.to("mps"), img, img,
        [True] * len(idxs), Fm=F, return_fig_ax=True,
    )
    assert fig is not None and ax is not None
