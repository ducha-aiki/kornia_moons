# nbdev → Standard Package Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the nbdev notebook-first workflow with a conventional Python package layout (pyproject.toml, tests/, MkDocs Material docs), keeping the documentation site with its executed visualizations.

**Architecture:** The `.py` modules under `kornia_moons/` become the source of truth (they already contain all the code — we only strip nbdev markers). Notebooks move to `docs/` as tutorials that *import* the package, rendered by mkdocs-jupyter from committed outputs. A pytest suite extracted from the notebooks' assertion cells is created FIRST as a safety net, before anything else changes.

**Tech Stack:** setuptools + pyproject.toml, pytest, nbmake, mkdocs-material, mkdocs-jupyter, mkdocstrings[python].

**Spec:** `docs/superpowers/specs/2026-08-07-nbdev-migration-design.md`

## Global Constraints

- **Zero semantic changes** to `kornia_moons/feature.py` and `kornia_moons/viz.py` — only remove nbdev headers/markers.
- `kornia_moons/line.py` is untracked WIP: **never** `git add` it, never modify it, never delete it. Same for the untracked `token` file (a credential — must never be committed).
- Version becomes `0.2.10.dev0` (PEP 440), single-sourced in `kornia_moons/__init__.py`.
- `requires-python = ">=3.9"`; runtime deps exactly: `kornia`, `torch`, `matplotlib`, `opencv-python`.
- Work happens in the active conda env `py311` (`/opt/homebrew/Caskroom/miniforge/base/envs/py311`). It already has torch + opencv; `pip install -e .` brings kornia.
- All commands run from the repo root `/Users/oldufo/dev/kornia_moons`.
- Commit after every task. Never use `git add -A` or `git add .` (would pick up `line.py`, `token`, junk dirs) — always add explicit paths.
- The first execution of the AffNet test and of `docs/viz.ipynb` downloads pretrained weights (AffNet, SOLD2) — expect a one-time delay; this matches what nbdev CI does today.

---

### Task 1: Pytest safety net against the current, unmodified package

The tests are extracted from the notebooks' demo/assertion cells and must PASS against the current nbdev-generated modules before we touch anything. This is a refactoring safety net — the "failing test first" cycle doesn't apply; instead we prove the tests pass *before* the migration so any later failure is a migration bug.

**Files:**
- Create: `tests/test_feature.py`
- Create: `tests/test_viz.py`

**Interfaces:**
- Consumes: current public API of `kornia_moons.feature` and `kornia_moons.viz` (unchanged throughout the migration).
- Produces: `pytest tests -v` as the regression gate every later task runs.

- [ ] **Step 1: Install the package and pytest into py311**

```bash
pip install -e . && pip install pytest
```

Expected: installs kornia + deps, `Successfully installed ... kornia_moons`.

- [ ] **Step 2: Write `tests/test_feature.py`**

```python
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
```

- [ ] **Step 3: Write `tests/test_viz.py`**

```python
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import cv2
import kornia
import pytest
import torch
from kornia.utils import image_to_tensor

from kornia_moons.feature import laf_from_opencv_kpts
from kornia_moons.viz import (
    draw_LAF_matches,
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
```

- [ ] **Step 4: Run the suite — must pass against the CURRENT code**

Run: `pytest tests -v`
Expected: all tests PASS (AffNet test downloads weights on first run). If a test fails, the test is wrong, not the code — fix the test to match actual current behavior.

- [ ] **Step 5: Commit**

```bash
git add tests/test_feature.py tests/test_viz.py
git commit -m "test: add pytest safety net extracted from notebook assertions"
```

---

### Task 2: Make the modules the source of truth (strip nbdev markers)

**Files:**
- Modify: `kornia_moons/feature.py` (header lines 1-6: `# AUTOGENERATED!` comment, `# %% auto 0` marker; and every `# %% ../feature.ipynb N` cell marker)
- Modify: `kornia_moons/viz.py` (same pattern)
- Modify: `kornia_moons/__init__.py` (version)
- Delete: `kornia_moons/_modidx.py`, `kornia_moons/_nbdev.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: same public API (`__all__` in both modules is kept verbatim); `kornia_moons.__version__ == "0.2.10.dev0"` (Task 3's pyproject reads it via setuptools dynamic metadata).

- [ ] **Step 1: Strip nbdev marker comments from both modules**

Remove only these line types from `kornia_moons/feature.py` and `kornia_moons/viz.py` (keep everything else byte-identical, including `__all__`):
- `# AUTOGENERATED! DO NOT EDIT: File to edit: ...` (line 1)
- `# %% auto 0` markers
- `# %% ../feature.ipynb N` / `# %% ../viz.ipynb N` cell markers

```bash
sed -i '' -e '/^# AUTOGENERATED! DO NOT EDIT/d' -e '/^# %%/d' kornia_moons/feature.py kornia_moons/viz.py
```

- [ ] **Step 2: Verify only comment lines changed**

Run: `git diff --stat kornia_moons/feature.py kornia_moons/viz.py && git diff kornia_moons/ | grep '^-' | grep -v '^---' | grep -cv '^-# \(%%\|AUTOGENERATED\)'`
Expected: the last count is `0` — every removed line is a `# %%` or AUTOGENERATED comment, nothing added except context.
Also run: `git diff kornia_moons/ | grep '^+[^+]' | head`
Expected: no output (no added lines at all).

- [ ] **Step 3: Delete nbdev machinery and bump version**

```bash
git rm kornia_moons/_modidx.py kornia_moons/_nbdev.py
```

Then edit `kornia_moons/__init__.py` to exactly:

```python
__version__ = "0.2.10.dev0"
```

- [ ] **Step 4: Run the safety net**

Run: `pytest tests -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add kornia_moons/feature.py kornia_moons/viz.py kornia_moons/__init__.py
git commit -m "refactor: strip nbdev markers, modules are now the source of truth"
```

---

### Task 3: pyproject.toml replaces settings.ini + setup.py

**Files:**
- Create: `pyproject.toml`
- Delete: `settings.ini`, `setup.py`, `MANIFEST.in`

**Interfaces:**
- Consumes: `kornia_moons.__version__` from Task 2.
- Produces: installable package with `[dev]` extra: `pytest`, `nbmake`, `mkdocs-material`, `mkdocs-jupyter`, `mkdocstrings[python]` — Tasks 4-6 rely on these exact extra names.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "kornia_moons"
dynamic = ["version"]
description = "Conversions between kornia and other computer vision libraries formats"
readme = "README.md"
requires-python = ">=3.9"
license = { text = "Apache-2.0" }
authors = [{ name = "Dmytro Mishkin", email = "ducha.aiki@gmail.com" }]
keywords = ["kornia", "python", "pytorch", "deep learning", "computer vision"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "kornia",
    "torch",
    "matplotlib",
    "opencv-python",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "nbmake",
    "mkdocs-material",
    "mkdocs-jupyter",
    "mkdocstrings[python]",
]

[project.urls]
Homepage = "https://github.com/ducha-aiki/kornia_moons"
Documentation = "https://ducha-aiki.github.io/kornia_moons/"
Repository = "https://github.com/ducha-aiki/kornia_moons"

[tool.setuptools]
packages = ["kornia_moons"]

[tool.setuptools.dynamic]
version = { attr = "kornia_moons.__version__" }

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Remove the nbdev packaging files**

```bash
git rm settings.ini setup.py MANIFEST.in
```

- [ ] **Step 3: Reinstall from pyproject and verify version**

```bash
pip install -e ".[dev]" && python -c "import kornia_moons; print(kornia_moons.__version__)"
```

Expected: install succeeds; prints `0.2.10.dev0`.

- [ ] **Step 4: Run the safety net**

Run: `pytest tests -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: replace settings.ini/setup.py with pyproject.toml"
```

---

### Task 4: Notebooks become docs tutorials in docs/

The notebooks currently *define* the library code in `#| export` cells. As docs tutorials they must instead **import** the installed package, so the demo cells exercise the real modules and nbmake becomes an integration test. Definition cells are removed (the API reference page in Task 5 covers signatures/docstrings).

**Files:**
- Create: `docs/feature.ipynb`, `docs/viz.ipynb` (moved + cleaned from root)
- Create: `scratchpad script` `clean_nb.py` (in the session scratchpad dir, NOT committed)
- Delete (from git, in this task): root `feature.ipynb`, `viz.ipynb`, `index.ipynb`

**Interfaces:**
- Consumes: installed `kornia_moons` package (Tasks 2-3).
- Produces: `docs/feature.ipynb`, `docs/viz.ipynb` with committed executed outputs — Task 5 renders them (`execute: false`), Task 6 CI runs them via `pytest --nbmake`.

- [ ] **Step 1: Write the cleaning script** to `/private/tmp/claude-501/-Users-oldufo-dev-kornia-moons/21eec326-0aa0-48ab-bb5e-56183c47de19/scratchpad/clean_nb.py`

```python
"""Convert an nbdev notebook into a docs tutorial: drop nbdev cells, fix paths."""
import json
import sys

IMPORT_CELL = """import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.utils import image_to_tensor, tensor_to_image

from kornia_moons.feature import *
from kornia_moons.viz import *"""


def keep(cell):
    if cell["cell_type"] != "code":
        return True
    src = "".join(cell["source"]).strip()
    if not src:
        return False
    # nbdev directive cells and exported definitions are dropped:
    # the package is imported instead of defined in the notebook
    for prefix in ("#| default_exp", "#| hide", "#| export", "#|export"):
        if src.startswith(prefix):
            return False
    return True


def fix_paths(cell):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        # notebooks move from repo root to docs/ -> data lives one level up
        src = src.replace("'data/", "'../data/").replace('"data/', '"../data/')
        cell["source"] = src.splitlines(keepends=True)
    return cell


def main(path_in, path_out):
    with open(path_in) as f:
        nb = json.load(f)
    cells = [fix_paths(c) for c in nb["cells"] if keep(c)]
    first = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": IMPORT_CELL.splitlines(keepends=True),
    }
    nb["cells"] = [first] + cells
    with open(path_out, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Produce the cleaned notebooks in docs/**

```bash
mkdir -p docs
python /private/tmp/claude-501/-Users-oldufo-dev-kornia-moons/21eec326-0aa0-48ab-bb5e-56183c47de19/scratchpad/clean_nb.py feature.ipynb docs/feature.ipynb
python /private/tmp/claude-501/-Users-oldufo-dev-kornia-moons/21eec326-0aa0-48ab-bb5e-56183c47de19/scratchpad/clean_nb.py viz.ipynb docs/viz.ipynb
```

- [ ] **Step 3: Execute both notebooks in place to refresh outputs**

```bash
cd docs && jupyter nbconvert --to notebook --execute --inplace feature.ipynb viz.ipynb; cd ..
```

Expected: both execute cleanly (viz downloads SOLD2 weights on first run; takes a few minutes). If `jupyter` is missing: `pip install jupyter`. If a cell fails on an undefined name, the import cell in `clean_nb.py` is missing that name — add the specific import and re-run from Step 2.

- [ ] **Step 4: Smoke-run via nbmake (what CI will do)**

Run: `pytest --nbmake docs/feature.ipynb docs/viz.ipynb`
Expected: 2 passed.

- [ ] **Step 5: Remove root notebooks from git and commit**

```bash
git rm feature.ipynb viz.ipynb index.ipynb
git add docs/feature.ipynb docs/viz.ipynb
git commit -m "docs: convert notebooks to package-importing tutorials under docs/"
```

Note: `index.ipynb`'s content becomes `docs/index.md` + README in Task 5; its rendered figures already exist in `index_files/` (untracked) and are salvaged there.

---

### Task 5: MkDocs Material site + hand-maintained README

**Files:**
- Create: `mkdocs.yml`, `docs/index.md`, `docs/api.md`, `docs/assets/orb-keypoints.png`, `docs/assets/laf-visualization.png`, `docs/assets/roundtrip-keypoints.png`
- Modify: `README.md`

**Interfaces:**
- Consumes: `docs/feature.ipynb`, `docs/viz.ipynb` from Task 4; docstrings in the modules.
- Produces: `mkdocs build` as the docs gate; `mkdocs gh-deploy --force` used by Task 6's deploy workflow.

- [ ] **Step 1: Salvage the README figures into docs/assets/**

The three figures referenced by the current README exist only in the untracked `index_files/` dir (they are broken on GitHub today — this fixes that):

```bash
mkdir -p docs/assets
cp index_files/figure-commonmark/cell-3-output-2.png docs/assets/orb-keypoints.png
cp index_files/figure-commonmark/cell-3-output-3.png docs/assets/laf-visualization.png
cp index_files/figure-commonmark/cell-3-output-5.png docs/assets/roundtrip-keypoints.png
```

- [ ] **Step 2: Rewrite `README.md`**

Keep the current content/flow (install, ORB example, links) but hand-maintained. Exact content:

````markdown
# kornia_moons

Conversions between [kornia](https://github.com/kornia/kornia) and other computer vision libraries formats: OpenCV keypoints ↔ kornia LAFs, matches, and visualization utilities.

**Documentation:** https://ducha-aiki.github.io/kornia_moons/

## Install

```bash
pip install kornia_moons
```

## How to use

Convert OpenCV ORB keypoints to kornia LAFs and back:

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

![ORB keypoints drawn by OpenCV](docs/assets/orb-keypoints.png)

![The same keypoints as kornia LAFs](docs/assets/laf-visualization.png)

![Keypoints converted back to OpenCV](docs/assets/roundtrip-keypoints.png)

See the [feature tutorial](https://ducha-aiki.github.io/kornia_moons/feature/) and [visualization tutorial](https://ducha-aiki.github.io/kornia_moons/viz/) for more.

## Development

```bash
pip install -e ".[dev]"
pytest tests                                # unit tests
pytest --nbmake docs/feature.ipynb docs/viz.ipynb   # run the docs notebooks
mkdocs serve                                # preview docs locally
```
````

- [ ] **Step 3: Write `docs/index.md`**

Same content as the README, with two path adjustments for the docs site: image links become `assets/orb-keypoints.png` (etc., no `docs/` prefix) and tutorial links become relative (`feature/`, `viz/`).

- [ ] **Step 4: Write `docs/api.md`**

```markdown
# API reference

## kornia_moons.feature

::: kornia_moons.feature

## kornia_moons.viz

::: kornia_moons.viz
```

- [ ] **Step 5: Write `mkdocs.yml`**

```yaml
site_name: kornia_moons
site_url: https://ducha-aiki.github.io/kornia_moons/
site_description: Conversions between kornia and other computer vision libraries formats
repo_url: https://github.com/ducha-aiki/kornia_moons
repo_name: ducha-aiki/kornia_moons

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

plugins:
  - search
  - mkdocs-jupyter:
      execute: false
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            members_order: source

nav:
  - Home: index.md
  - Local features tutorial: feature.ipynb
  - Visualization tutorial: viz.ipynb
  - API reference: api.md

exclude_docs: |
  superpowers/
```

(`exclude_docs` keeps the design spec and plan out of the published site.)

- [ ] **Step 6: Build the site locally**

Run: `mkdocs build`
Expected: `site/` builds without errors; warnings about `docs/superpowers` must NOT appear (they're excluded). Spot-check: `ls site/feature/index.html site/api/index.html` and open `site/index.html` to confirm the three images render.

- [ ] **Step 7: Commit**

```bash
git add mkdocs.yml docs/index.md docs/api.md docs/assets README.md
git commit -m "docs: add MkDocs Material site, hand-maintained README"
```

---

### Task 6: Rewrite CI workflows

**Files:**
- Modify: `.github/workflows/test.yaml` (full replacement)
- Modify: `.github/workflows/deploy.yaml` (full replacement)

**Interfaces:**
- Consumes: `[dev]` extra (Task 3), `tests/` (Task 1), `docs/*.ipynb` (Task 4), `mkdocs.yml` (Task 5).
- Produces: CI green on push/PR; docs deployed to gh-pages on master pushes.

- [ ] **Step 1: Replace `.github/workflows/test.yaml`**

```yaml
name: CI
on: [workflow_dispatch, pull_request, push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: pytest tests -v
      - run: pytest --nbmake docs/feature.ipynb docs/viz.ipynb
```

- [ ] **Step 2: Replace `.github/workflows/deploy.yaml`**

```yaml
name: Deploy docs
permissions:
  contents: write
on:
  push:
    branches: [master, main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: mkdocs gh-deploy --force
```

- [ ] **Step 3: Validate YAML parses**

Run: `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yaml')]; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/test.yaml .github/workflows/deploy.yaml
git commit -m "ci: replace nbdev workflows with pytest + nbmake + mkdocs deploy"
```

---

### Task 7: Cleanup, .gitignore, contributor docs

**Files:**
- Delete (tracked): `nbdev.yml`, `_quarto.yml`, `sidebar.yml`, `styles.css`, `docker-compose.yml`, `.devcontainer.json`
- Modify: `.gitignore`, `CONTRIBUTING.md`, `CLAUDE.md`
- Remove (untracked local junk): `_proc/`, `build/`, `dist/`, `kornia_moons.egg-info/`, `index_files/`, `CHANGELOG.bak`, root `kn_church-2.jpg`, `kn_church-8.jpg`

**Interfaces:**
- Consumes: everything before.
- Produces: final clean tree; full verification gate.

- [ ] **Step 1: Delete tracked nbdev/fastai artifacts**

```bash
git rm nbdev.yml _quarto.yml sidebar.yml styles.css docker-compose.yml .devcontainer.json
```

- [ ] **Step 2: Replace `.gitignore`**

```gitignore
# credentials — never commit
token

# build artifacts
build/
dist/
*.egg-info/
__pycache__/
*.py[cod]

# docs build
site/
_proc/
_docs/

# notebooks
.ipynb_checkpoints/

# misc
*.bak
*.log
*~
~*
_tmp*
tmp*
.last_checked
.gitattributes
.gitconfig
.DS_Store
```

- [ ] **Step 3: Remove untracked junk (verify each is junk before deleting)**

`kn_church-2.jpg` / `kn_church-8.jpg` at the root are referenced by nothing (verify: `grep -r "kn_church" --include="*.py" --include="*.ipynb" --include="*.md" . --exclude-dir=site` returns only matches in `_proc/` or nothing). `_proc/`, `build/`, `dist/`, `kornia_moons.egg-info/` are nbdev/setuptools build output. `index_files/` figures were salvaged into `docs/assets/` in Task 5.

```bash
rm -rf _proc build dist kornia_moons.egg-info index_files CHANGELOG.bak kn_church-2.jpg kn_church-8.jpg
```

Do NOT touch: `token`, `kornia_moons/line.py`.

- [ ] **Step 4: Rewrite `CONTRIBUTING.md`**

```markdown
# How to contribute

## Setup

```bash
git clone https://github.com/ducha-aiki/kornia_moons
cd kornia_moons
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests                                        # unit tests
pytest --nbmake docs/feature.ipynb docs/viz.ipynb   # docs notebooks
```

## Documentation

Docs live in `docs/` and are built with MkDocs Material. The tutorial
notebooks are rendered from their committed outputs; if you change one,
re-execute it before committing:

```bash
jupyter nbconvert --to notebook --execute --inplace docs/<name>.ipynb
mkdocs serve   # preview at http://127.0.0.1:8000
```

## Did you find a bug?

* Ensure the bug was not already reported by searching on GitHub under Issues.
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages.

#### Did you write a patch that fixes a bug?

* Open a new GitHub pull request with the patch.
* Ensure that your PR includes a test that fails without your patch, and pass with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

## PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.
* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.
* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.
```

- [ ] **Step 5: Rewrite `CLAUDE.md`**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

kornia_moons is a small Python library providing conversions between kornia and other computer vision library formats (mainly OpenCV): local features (LAFs ↔ cv2.KeyPoint), matches, and visualization helpers. Published to PyPI as `kornia_moons`.

## Structure

Standard flat layout: source in `kornia_moons/` (`feature.py`, `viz.py`), tests in `tests/`, docs in `docs/` (MkDocs Material + tutorial notebooks with committed outputs). Packaging via `pyproject.toml`; version single-sourced in `kornia_moons/__init__.py`.

`kornia_moons/line.py` is untracked work-in-progress — do not commit, modify, or delete it. The untracked `token` file is a credential — never commit it.

## Commands

```bash
pip install -e ".[dev]"                             # editable install with dev tools
pytest tests                                        # run unit tests
pytest tests/test_feature.py::test_orb_laf_roundtrip -v   # single test
pytest --nbmake docs/feature.ipynb docs/viz.ipynb   # execute docs notebooks (CI does this)
mkdocs serve                                        # live docs preview
mkdocs build                                        # build docs site
```

Release: bump `__version__`, then `python -m build` + `twine upload`.

## Conventions

- Docs tutorials (`docs/*.ipynb`) import the installed package; they contain no source definitions. If you change one, re-execute it in place (`jupyter nbconvert --to notebook --execute --inplace docs/<name>.ipynb`) so committed outputs stay fresh — mkdocs renders stored outputs without executing.
- The API reference (`docs/api.md`) is generated from docstrings by mkdocstrings.
- Function naming follows the direction of conversion, e.g. `laf_from_opencv_SIFT_kpts` / `opencv_SIFT_kpts_from_laf`, `cv2_matches_from_kornia` / `kornia_matches_from_cv2`.
- Keep PRs focused; don't mix style-only changes with functional changes (see CONTRIBUTING.md).
```

- [ ] **Step 6: Full verification gate**

```bash
pytest tests -v && pytest --nbmake docs/feature.ipynb docs/viz.ipynb && mkdocs build && python -c "import kornia_moons; print(kornia_moons.__version__)"
```

Expected: all pass, prints `0.2.10.dev0`.
Also: `git status` must still show `kornia_moons/line.py` and `token` as untracked (untouched).

- [ ] **Step 7: Commit**

```bash
git add .gitignore CONTRIBUTING.md CLAUDE.md
git commit -m "chore: remove nbdev artifacts, update contributor docs"
```

---

## Post-plan notes

- After merge to master, the deploy workflow publishes the MkDocs site to gh-pages, replacing the Quarto site at the same URL.
- PyPI release flow is now: bump `__version__` → `python -m build` → `twine upload dist/*` (no nbdev_pypi).
