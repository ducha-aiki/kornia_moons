# Follow-up Docs & Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `draw_LAF_matches_from_result_dict` NameError with a regression test, add kornia-style docstrings to the whole public API, and turn the README into a proper landing page.

**Architecture:** Three independent deliverables on one branch (`followup-docs-and-fixes`, already created off master). The bug fix is strict TDD. Docstrings are enforced by a new automated completeness test. README/index changes ship with a new showcase image extracted from the executed viz tutorial.

**Tech Stack:** pytest, mkdocs-material + mkdocstrings (already installed via `pip install -e ".[dev]"` in the active py311 conda env).

**Spec:** `docs/superpowers/specs/2026-08-07-followup-docs-and-fixes-design.md`

## Global Constraints

- The ONLY behavior change in this PR is the one-word fix `rbesult_dict` → `result_dict` at `kornia_moons/viz.py:231`. Docstring additions must not touch signatures, defaults, or code.
- `draw_LAF_inliers_perspective_repjojected` keeps its typo'd name (public API stability).
- Untracked files `kornia_moons/line.py` and `token` must never be committed, modified, or deleted. Never `git add -A` or `git add .` — always add explicit paths.
- Work from /Users/oldufo/dev/kornia_moons in the active py311 env.
- Local quirk: if `mkdocs build` or nbmake fails on a jupyter config error, prefix with `JUPYTER_CONFIG_DIR=<empty scratch dir>` (stray `~/.jupyter/jupyter_nbconvert_config.json` on this machine; CI unaffected).
- Commit messages end with:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RHsMk5EeKZHcZ2mHcdxeBM
```

---

### Task 1: Fix `rbesult_dict` NameError (TDD)

**Files:**
- Modify: `kornia_moons/viz.py:231` (one word)
- Test: `tests/test_viz.py` (append two tests)

**Interfaces:**
- Consumes: `draw_LAF_matches_from_result_dict(result_dict, img1, img2, draw_dict=...)` — expects `result_dict` with keys `feat_dict1` (`{'lafs': (1,N,2,3) tensor}`), `feat_dict2` (same), `tents_dict` (`{'idxs': (M,2) long tensor}`), `inlier_mask` (list/array of M bools), and optionally `'H'` (3x3 np.array) and/or `'F'`. Returns `None` (it forwards to `draw_LAF_matches` without `return_fig_ax`).
- Produces: `test_draw_laf_matches_from_result_dict` and `test_draw_laf_matches_from_result_dict_with_F` in `tests/test_viz.py`.

- [ ] **Step 1: Write the failing regression tests**

Append to `tests/test_viz.py` (it already imports `matplotlib` with Agg backend, `cv2`, `kornia`, `torch`, fixtures `img` and `lafs_descs`, and has an autouse `close_figures` fixture; extend the `kornia_moons.viz` import list with `draw_LAF_matches_from_result_dict`):

```python
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
```

`tests/test_viz.py` has no `numpy` import yet — add `import numpy as np` next to the existing imports.

- [ ] **Step 2: Run tests to verify they fail with NameError**

Run: `pytest tests/test_viz.py -k result_dict -v`
Expected: both FAIL with `NameError: name 'rbesult_dict' is not defined` (line 231 evaluates on every call, so both cases crash today).

- [ ] **Step 3: Apply the one-word fix**

In `kornia_moons/viz.py:231` change:

```python
    if 'F' in rbesult_dict:
```

to:

```python
    if 'F' in result_dict:
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests -v`
Expected: 18 passed (16 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add tests/test_viz.py kornia_moons/viz.py
git commit -m "fix: NameError in draw_LAF_matches_from_result_dict (rbesult_dict typo)"
```

---

### Task 2: Kornia-style docstrings for the whole public API

**Files:**
- Modify: `kornia_moons/feature.py` (docstrings only — every `__all__` member)
- Modify: `kornia_moons/viz.py` (docstrings only — every `__all__` member)
- Test: `tests/test_docstrings.py` (create)

**Interfaces:**
- Consumes: `kornia_moons.feature.__all__` (15 names), `kornia_moons.viz.__all__` (9 names).
- Produces: non-empty `__doc__` on every public member; `tests/test_docstrings.py::test_public_api_documented`.

- [ ] **Step 1: Write the failing completeness test**

Create `tests/test_docstrings.py`:

```python
import importlib

import pytest

MODULES = ["kornia_moons.feature", "kornia_moons.viz"]


@pytest.mark.parametrize("modname", MODULES)
def test_public_api_documented(modname):
    mod = importlib.import_module(modname)
    missing = []
    for name in mod.__all__:
        obj = getattr(mod, name)
        doc = getattr(obj, "__doc__", None)
        if not doc or not doc.strip():
            missing.append(name)
    assert not missing, f"{modname}: missing docstrings for {missing}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_docstrings.py -v`
Expected: `kornia_moons.feature` FAILS listing all 15 names; `kornia_moons.viz` FAILS listing at least 6 names (3 have one-liners already).

- [ ] **Step 3: Add docstrings to `kornia_moons/feature.py`**

Format rules (kornia house style, Google format — mkdocstrings renders these):
- One-line imperative summary, blank line, then `Args:` / `Returns:` (and `Raises:` where the code raises).
- Tensor shapes written explicitly: LAFs `(1, N, 2, 3)`, responses `(1, N, 1)`, match distances `(N, 1)`, match indexes `(N, 2)`.
- `Example:` blocks (indented code, no doctest `>>>` output-checking beyond what's shown) only for functions the tutorials demonstrate: `laf_from_opencv_kpts`, `opencv_kpts_from_laf`, `cv2_matches_from_kornia`, `kornia_matches_from_cv2`, `OpenCVDetectorKornia`, `OpenCVFeatureKornia`, `OpenCVDetectorWithAffNetKornia`, `visualize_LAF`, `draw_LAF_matches`.
- Docstrings ONLY — do not touch signatures, defaults, or bodies. Verify with `git diff` that every hunk is a pure docstring insertion.

Three complete exemplars to copy the style from (use these verbatim for these three members):

`laf_from_opencv_kpts`:
```python
def laf_from_opencv_kpts(kpts: List[cv2.KeyPoint],
                         mrSize: float = 6.0,
                         device: torch.device = torch.device('cpu'),
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
```

`to_numpy_image` (simple function, no example):
```python
def to_numpy_image(img: Union[str, np.array, torch.Tensor]):
    """Load or convert an image into an RGB numpy array.

    Args:
        img: Image file path, `(B, C, H, W)` / `(C, H, W)` tensor, or
            numpy array. Paths are read with OpenCV and converted to RGB.

    Returns:
        The image as an ``np.ndarray``.

    Raises:
        TypeError: If ``img`` is not a str, np.ndarray, or torch.Tensor.
    """
```

`OpenCVDetectorKornia` (class + forward):
```python
class OpenCVDetectorKornia(nn.Module):
    """Wrap an OpenCV detector into a kornia-compatible detection module.

    The wrapped detector runs on the numpy image under the hood; the module
    accepts and returns torch tensors, so it can slot into kornia pipelines.

    Args:
        opencv_detector: Any OpenCV detector exposing ``detect`` (e.g.
            ``cv2.SIFT_create()``).
        mrSize: Measurement-region scale multiplier (6.0 for SIFT, 1.0 for ORB).
        make_upright: If True, deduplicate keypoints and zero their angles.
        max_kpts: Keep at most this many keypoints when ``make_upright`` is
            set; -1 keeps the OpenCV output as is.

    Example:
        >>> detector = OpenCVDetectorKornia(cv2.SIFT_create(500))
        >>> timg = kornia.image_to_tensor(img, False).float() / 255.
        >>> lafs, resps = detector(timg)
    """
```
with `forward` getting:
```python
    def forward(self, x: torch.Tensor, mask=None):
        """Detect keypoints on a batched image tensor.

        Args:
            x: Image tensor of shape :math:`(1, C, H, W)`, values in
                :math:`[0, 1]` float or :math:`[0, 255]`.
            mask: Optional detection mask.

        Returns:
            LAFs of shape :math:`(1, N, 2, 3)` and responses :math:`(1, N, 1)`.
        """
```

Content table for the remaining feature.py members (expand each into the exemplar format):

| Member | Summary | Key Args/Returns notes |
|---|---|---|
| `to_torch` | Convert a list or numpy array to a torch tensor. | Raises TypeError on other types. |
| `to_np` | Convert a tensor or sequence to a numpy array. | Detaches and moves to CPU first. |
| `laf_from_opencv_ORB_kpts` | Convert ORB keypoints to LAFs (mrSize=1.0 convention). | Returns `(1, N, 2, 3)`; optional resp `(1, N, 1)`. |
| `laf_from_opencv_SIFT_kpts` | Convert SIFT keypoints to LAFs (mrSize=6.0 convention). | Same shapes. |
| `opencv_kpts_from_laf` | Convert LAFs back to OpenCV keypoints. | `mrSize` must match the one used to create the LAFs; optional `resps` `(1, N, 1)`. |
| `opencv_SIFT_kpts_from_laf` | LAFs to SIFT-convention keypoints (mrSize=6.0). | |
| `opencv_ORB_kpts_from_laf` | LAFs to ORB-convention keypoints (mrSize=1.0). | |
| `cv2_matches_from_kornia` | Convert kornia match distances/indexes to cv2.DMatch list. Example from tutorial (match_nn on SIFT descs). | dists `(N, 1)`, idxs `(N, 2)`. |
| `kornia_matches_from_cv2` | Convert cv2.DMatch list back to kornia distance/index tensors. | Returns dists `(N, 1)` float, idxs `(N, 2)` long. |
| `make_keypoints_upright` | Deduplicate keypoints by response and zero their orientations. | Note: mutates the input keypoints; returns them sorted by response, descending. |
| `OpenCVFeatureKornia` | Wrap an OpenCV detect-and-describe pipeline (detectAndCompute) for kornia. | forward returns lafs `(1, N, 2, 3)`, resps `(1, N, 1)`, descs `(1, N, D)`. Example from tutorial. |
| `OpenCVDetectorWithAffNetKornia` | OpenCV detector + kornia AffNet affine-shape refinement. | Notes: downloads AffNet weights on first use; keeps original orientation. Example from tutorial. |

- [ ] **Step 4: Add docstrings to `kornia_moons/viz.py`**

Same format rules. The three existing `'''…'''` one-liners (`draw_LAF_matches` line 88-89, `draw_LAF_inliers_perspective_repjojected` line 249, `draw_epipolar_errors_in_single_image` line 365) become the summary line of full Google-format docstrings — keep their wording, fix nothing else.

Content table:

| Member | Summary | Key Args/Returns notes |
|---|---|---|
| `visualize_LAF` | Draw local affine frames over an image. | `img` `(B, C, H, W)`; `LAF` `(B, N, 2, 3)`; returns `(fig, ax)` only when `return_fig_ax=True`. Example from tutorial. |
| `epilines_to_start_end_points` | Clip epipolar lines to image-boundary start/end points. | `epi` `(N, 3)` line coefficients; returns `(2, N, 2)` stacked start/end points for an `h` x `w` image. |
| `draw_LAF_matches` | (existing summary kept) draws LAFs, tentative matches, inlier epipolar lines (if F given), corner reprojection (if H given). | Full Args for the draw_dict keys (inlier_color, tentative_color, feature_color, vertical); Example from tutorial. |
| `draw_LAF_matches_from_result_dict` | Draw matches from a result dictionary (lafs, tentative idxs, inlier mask, optional H/F). | Document the exact dict schema from Task 1's Interfaces block. |
| `draw_LAF_inliers_perspective_repjojected` | (existing summary kept). | Note the name is kept for backward compatibility. |
| `draw_epipolar_errors_in_single_image` | (existing summary kept). | kp1/kp2 `(N, 2)` arrays, Fm `(3, 3)`. |
| `plot_images` | Plot a horizontal or vertical strip of images. | From SOLD2/LightGlue-style helper; establishes the figure `plot_lines` draws on. |
| `plot_lines` | Plot line segments (and endpoints) on the current figure's axes. | `lines`: list of `(N, 2, 2)` tensors, one per image; call after `plot_images`. |
| `plot_color_line_matches` | Plot matched line segments with one color per match. | Same input shape; used with SOLD2 matches in the tutorial. |

- [ ] **Step 5: Verify docstrings-only diff and green suite**

Run: `git diff kornia_moons/ | grep '^-' | grep -v '^---' | grep -v "'''"` — every removed line must be part of the three replaced `'''` one-liners; and `pytest tests -v` → 19 passed (18 + docstring test ×1... the parametrized test counts as 2, so 20 total).
Expected: 20 passed.

- [ ] **Step 6: Commit**

```bash
git add kornia_moons/feature.py kornia_moons/viz.py tests/test_docstrings.py
git commit -m "docs: add kornia-style docstrings to the whole public API"
```

---

### Task 3: README showcase, docs index, verification

**Files:**
- Create: `docs/assets/laf-matches.png` (extracted from executed `docs/viz.ipynb`)
- Modify: `README.md` (full replacement)
- Modify: `docs/index.md` (full replacement)

**Interfaces:**
- Consumes: the executed cell in `docs/viz.ipynb` whose source contains `cv2.findHomography` and `draw_LAF_matches` (SIFT + match_snn + MAGSAC homography demo, 1 image output).
- Produces: the final README/landing page.

- [ ] **Step 1: Extract the showcase image**

```bash
python - << 'EOF'
import base64, json
nb = json.load(open('docs/viz.ipynb'))
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'code' and 'findHomography' in src and 'draw_LAF_matches' in src:
        png = next(o['data']['image/png'] for o in c['outputs'] if 'image/png' in o.get('data', {}))
        open('docs/assets/laf-matches.png', 'wb').write(base64.b64decode(png))
        print('written', len(png))
        break
else:
    raise SystemExit('cell not found')
EOF
```

Expected: `written <N>`; `file docs/assets/laf-matches.png` reports PNG image data.

- [ ] **Step 2: Replace `README.md`**

````markdown
# kornia_moons

[![PyPI version](https://img.shields.io/pypi/v/kornia_moons.svg)](https://pypi.org/project/kornia_moons/)
[![CI](https://github.com/ducha-aiki/kornia_moons/actions/workflows/test.yaml/badge.svg)](https://github.com/ducha-aiki/kornia_moons/actions/workflows/test.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Conversions between [kornia](https://github.com/kornia/kornia) and other computer vision libraries formats, mainly OpenCV.

**Documentation:** https://ducha-aiki.github.io/kornia_moons/

## What's inside

- **Keypoint / LAF conversions** — OpenCV `cv2.KeyPoint` ↔ kornia local affine frames (LAFs), handling the differing scale and orientation conventions (`mrSize=6.0` for SIFT, `1.0` for ORB): `laf_from_opencv_SIFT_kpts`, `opencv_ORB_kpts_from_laf`, …
- **Match conversions** — kornia match tensors ↔ `cv2.DMatch` lists: `cv2_matches_from_kornia`, `kornia_matches_from_cv2`
- **OpenCV detectors as kornia modules** — `OpenCVDetectorKornia`, `OpenCVFeatureKornia`, `OpenCVDetectorWithAffNetKornia` (with kornia AffNet shape refinement)
- **Visualization** — LAFs (`visualize_LAF`), matches with inliers/epipolar lines/reprojected corners (`draw_LAF_matches`), epipolar errors, and SOLD2 line segments (`plot_lines`, `plot_color_line_matches`)

## Install

```bash
pip install kornia_moons
```

## Quick start: keypoint conversion

```python
import matplotlib.pyplot as plt
import cv2
import kornia as K

from kornia_moons.feature import laf_from_opencv_ORB_kpts, opencv_ORB_kpts_from_laf
from kornia_moons.viz import visualize_LAF

img = cv2.cvtColor(cv2.imread('data/strahov.png'), cv2.COLOR_BGR2RGB)

det = cv2.ORB_create(500)
kps, descs = det.detectAndCompute(img, None)

out_img = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(out_img)

lafs = laf_from_opencv_ORB_kpts(kps)
visualize_LAF(K.image_to_tensor(img, False), lafs, 0)

kps_back = opencv_ORB_kpts_from_laf(lafs)
out_img2 = cv2.drawKeypoints(img, kps_back, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(out_img2)
```

![ORB keypoints drawn by OpenCV](https://raw.githubusercontent.com/ducha-aiki/kornia_moons/master/docs/assets/orb-keypoints.png)

![The same keypoints as kornia LAFs](https://raw.githubusercontent.com/ducha-aiki/kornia_moons/master/docs/assets/laf-visualization.png)

![Keypoints converted back to OpenCV](https://raw.githubusercontent.com/ducha-aiki/kornia_moons/master/docs/assets/roundtrip-keypoints.png)

## Matching and visualization

```python
import numpy as np
import torch
import kornia

from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from kornia_moons.viz import draw_LAF_matches

det = cv2.SIFT_create(100)
img1 = cv2.cvtColor(cv2.imread('data/strahov.png'), cv2.COLOR_BGR2RGB)

Hgt = np.array([[0.5, 0.1, 10], [-0.1, 0.5, 10], [0, 0, 1]])
img2 = cv2.warpPerspective(img1, Hgt, img1.shape[:2][::-1], borderValue=(255, 255, 255))

kps1, descs1 = det.detectAndCompute(img1, None)
lafs1 = laf_from_opencv_SIFT_kpts(kps1)
kps2, descs2 = det.detectAndCompute(img2, None)
lafs2 = laf_from_opencv_SIFT_kpts(kps2)

match_dists, match_idxs = kornia.feature.match_snn(
    torch.from_numpy(descs1).float(), torch.from_numpy(descs2).float(), 0.98)

H, mask = cv2.findHomography(
    kornia.feature.get_laf_center(lafs1[:, match_idxs[:, 0]]).numpy().reshape(-1, 2),
    kornia.feature.get_laf_center(lafs2[:, match_idxs[:, 1]]).numpy().reshape(-1, 2),
    cv2.USAC_MAGSAC, 0.5)

draw_LAF_matches(lafs1, lafs2, match_idxs, img1, img2, mask,
                 draw_dict={"inlier_color": (0.2, 1, 0.2),
                            "tentative_color": (0.8, 0.8, 0),
                            "feature_color": None,
                            "vertical": False}, H=H)
```

![SIFT matches with MAGSAC inliers and reprojected corners](https://raw.githubusercontent.com/ducha-aiki/kornia_moons/master/docs/assets/laf-matches.png)

## Learn more

- [Local features tutorial](https://ducha-aiki.github.io/kornia_moons/feature/) — conversions in both directions, matches, detector wrappers
- [Visualization tutorial](https://ducha-aiki.github.io/kornia_moons/viz/) — matches, epipolar geometry, SOLD2 line segments
- [API reference](https://ducha-aiki.github.io/kornia_moons/api/)

## Development

```bash
pip install -e ".[dev]"
pytest tests                                # unit tests
pytest --nbmake docs/feature.ipynb docs/viz.ipynb   # run the docs notebooks
mkdocs serve                                # preview docs locally
```
````

- [ ] **Step 3: Replace `docs/index.md`**

Identical content to the new README with exactly these substitutions:
- Every `https://raw.githubusercontent.com/ducha-aiki/kornia_moons/master/docs/assets/<name>.png` → `assets/<name>.png`
- Tutorial links `https://ducha-aiki.github.io/kornia_moons/feature/` → `feature/`, `.../viz/` → `viz/`, `.../api/` → `api/`
- Drop the line `**Documentation:** https://ducha-aiki.github.io/kornia_moons/` (the reader is already on the site).

- [ ] **Step 4: Verify the site and suite**

```bash
mkdocs build 2>&1 | tail -3
pytest tests -q | tail -1
pytest --nbmake docs/feature.ipynb docs/viz.ipynb -q | tail -1
grep -c "raw.githubusercontent" README.md   # expect 4
grep -c "raw.githubusercontent" docs/index.md   # expect 0
```

Expected: clean build; 20 passed; nbmake 2 passed; counts 4 and 0. Spot-check `site/index.html` shows the new image and `site/api/index.html` now renders Args/Returns sections from Task 2's docstrings.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/index.md docs/assets/laf-matches.png
git commit -m "docs: README showcase with badges, feature overview, matching example"
```

- [ ] **Step 6: Local-only cleanup (nothing committed)**

Remove stale nbdev git hooks that error on every merge (`nbdev_trust: command not found`):

```bash
grep -l "nbdev" .git/hooks/* 2>/dev/null
```

Delete exactly the files that grep lists (expected: `post-merge`, possibly others). Do not touch `.git/hooks/*.sample`. Verify `git status` output is hook-error-free afterwards.

---

## Post-plan notes

- PR flows against master; PyPI badge shows the last released version (0.2.9) until the next release — expected.
- The nbmake gate is unchanged by this PR but runs in CI; no need to run locally unless modules' import behavior changed (it didn't — docstrings only).
