# to_torch & max_kpts Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `to_torch` so plain Python lists convert, and give `max_kpts` sane semantics (non-positive keeps all; positive keeps the highest-response keypoints regardless of `make_upright`) — both with TDD regression tests.

**Architecture:** Two independent fixes in `kornia_moons/feature.py`, each its own task with red-green cycle. The `max_kpts` truncation moves out of the `make_upright` branch into a unified response-sorted top-k step shared by both detector classes.

**Tech Stack:** pytest (active py311 conda env, `pip install -e ".[dev]"` already done).

**Spec:** `docs/superpowers/specs/2026-08-07-to-torch-max-kpts-fixes-design.md`

## Global Constraints

- Branch `fix-to-torch-and-max-kpts` (already created off master).
- Only `kornia_moons/feature.py` and `tests/test_feature.py` change. No signature changes — only the bodies and docstrings shown below.
- Untracked `kornia_moons/line.py` and `token` must never be committed, modified, or deleted. Never `git add -A` or `git add .`.
- Version stays `0.2.10.dev0`.
- Work from /Users/oldufo/dev/kornia_moons.
- Commit messages end with:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RHsMk5EeKZHcZ2mHcdxeBM
```

---

### Task 1: `to_torch` accepts plain Python lists (TDD)

**Files:**
- Modify: `kornia_moons/feature.py:46-74` (the `to_torch` function)
- Test: `tests/test_feature.py` (append two tests)

**Interfaces:**
- Consumes: nothing new.
- Produces: `to_torch(list) -> torch.Tensor`; `to_torch(<unsupported>)` still raises `TypeError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_feature.py` (all needed imports already exist there):

```python
def test_to_torch_from_list():
    t = to_torch([1.0, 2.0, 3.0])
    assert isinstance(t, torch.Tensor)
    assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))


def test_to_torch_rejects_unsupported_type():
    with pytest.raises(TypeError):
        to_torch(42)
```

- [ ] **Step 2: Run to verify red**

Run: `pytest tests/test_feature.py -k to_torch -v`
Expected: `test_to_torch_from_list` FAILS with `TypeError: img should be List, np.array or torch.Tensor`; the other two to_torch tests pass.

- [ ] **Step 3: Fix the check and the docstring**

In `kornia_moons/feature.py`, change the body line:

```python
    if type(x) is List:
```

to:

```python
    if isinstance(x, list):
```

and replace the entire docstring of `to_torch` with:

```python
    """Convert a Python list or numpy array to a torch tensor, passing tensors through.

    Args:
        x: List, numpy array, or torch tensor to convert.

    Returns:
        ``x`` as a ``torch.Tensor``.

    Raises:
        TypeError: If ``x`` is not a list, np.ndarray, or torch.Tensor.
    """
```

(The previous `.. note::` block documenting the dead `typing.List` check is deleted — the quirk no longer exists.)

- [ ] **Step 4: Run the full suite**

Run: `pytest tests -v`
Expected: 22 passed (20 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add kornia_moons/feature.py tests/test_feature.py
git commit -m "fix: to_torch accepts plain Python lists (dead typing.List check)"
```

---

### Task 2: `max_kpts` — non-positive keeps all, positive keeps top-k by response, always applied (TDD)

**Files:**
- Modify: `kornia_moons/feature.py` — `OpenCVDetectorKornia` (class at ~line 276: docstring Args + `forward`) and `OpenCVDetectorWithAffNetKornia` (class at ~line 376: same)
- Test: `tests/test_feature.py` (three new tests + one modified)

**Interfaces:**
- Consumes: `make_keypoints_upright` (returns response-descending, deduped, mutated keypoints — already imported in the test file).
- Produces: for both classes: `max_kpts <= 0` → all keypoints kept; `max_kpts > 0` → exactly `max_kpts` highest-response keypoints, in both upright and non-upright modes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_feature.py`:

Note on tensor ranges: two of these tests compare the class's output against a direct `detect(img, ...)` call, so the image the class sees internally must be bit-identical to `img`. The class's low-range path (`255 * tensor_to_image(x)` for inputs scaled to [0,1]) does NOT round-trip uint8 exactly (float32 `255*(v/255)` can land just below the integer and truncate), which would make such comparisons flaky. Passing the tensor in the 0-255 range takes the exact `astype(np.uint8)` path instead — bit-identical. Use `image_to_tensor(img, False).float()` (no `/ 255.0`) in exactly these tests, as written:

```python
def test_detector_max_kpts_default_keeps_all(img):
    # regression: make_upright=True with default max_kpts=-1 used to drop the last keypoint
    expected = len(make_keypoints_upright(list(cv2.SIFT_create(500).detect(img, None))))
    timg255 = image_to_tensor(img, False).float()  # 0-255 range: exact uint8 round-trip
    lafs, resp = OpenCVDetectorKornia(cv2.SIFT_create(500), make_upright=True)(timg255)
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
```

And modify the existing `test_opencv_detector_affnet` to also pin the AffNet class's new semantics — change its detector construction line from:

```python
    det = OpenCVDetectorWithAffNetKornia(cv2.SIFT_create(500), make_upright=True)
```

to:

```python
    det = OpenCVDetectorWithAffNetKornia(cv2.SIFT_create(500), make_upright=True, max_kpts=50)
```

and add after its existing assertions:

```python
    assert lafs.shape[1] == 50
```

- [ ] **Step 2: Run to verify red**

Run: `pytest tests/test_feature.py -k "max_kpts or affnet" -v`
Expected failures on current code:
- `test_detector_max_kpts_default_keeps_all` — off by one (`expected - 1` returned).
- `test_detector_max_kpts_truncates[False]` — `max_kpts` currently ignored without `make_upright` (≈500 returned, not 10).
- `test_detector_max_kpts_keeps_strongest` — same reason.
- `test_opencv_detector_affnet` — passes or fails depending on ordering; with `max_kpts=50` on current code it returns 50 (`kpts[:min(len,50)]`) so it may PASS — that's fine; the red gate is the other three.
`test_detector_max_kpts_truncates[True]` also passes on current code (slice `[:10]` works); expected.

- [ ] **Step 3: Implement the unified truncation in BOTH classes**

In `OpenCVDetectorKornia.forward`, replace:

```python
        kpts = self.features.detect(img_np, mask)
        # Compute descriptors
        if self.make_upright:
            kpts = make_keypoints_upright(kpts)
            kpts = kpts[:min(len(kpts), self.max_kpts)]
```

with:

```python
        kpts = self.features.detect(img_np, mask)
        if self.make_upright:
            kpts = make_keypoints_upright(kpts)
        if self.max_kpts > 0:
            kpts = sorted(kpts, key=lambda k: k.response, reverse=True)[:self.max_kpts]
```

In `OpenCVDetectorWithAffNetKornia.forward`, replace the identical block (it also has the stale `# Compute descriptors` comment) with the identical new code.

Update the `max_kpts` Args entry in BOTH class docstrings to exactly:

```
        max_kpts: Keep at most this many of the highest-response keypoints,
            applied whether or not ``make_upright`` is set. Non-positive
            values (the default ``-1``) keep all keypoints.
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests -v`
Expected: 26 passed (22 from Task 1 + 4 new test items: default_keeps_all, truncates[False], truncates[True], keeps_strongest; affnet modified in place).

- [ ] **Step 5: Verify docs and notebooks still green**

```bash
mkdocs build 2>&1 | tail -1
pytest --nbmake docs/feature.ipynb docs/viz.ipynb -q | tail -1
```

(Local quirk: prefix both with `JUPYTER_CONFIG_DIR=<empty scratch dir>` if they fail on a jupyter config error; CI unaffected.)
Expected: clean build; 2 passed.

- [ ] **Step 6: Commit**

```bash
git add kornia_moons/feature.py tests/test_feature.py
git commit -m "fix: max_kpts keeps all by default and top-k by response always"
```

---

## Post-plan notes

- PR against master; PR body should call out the intentional behavior change: `max_kpts > 0` now truncates even when `make_upright=False` (previously silently ignored).
- `OpenCVFeatureKornia` has no `max_kpts` — out of scope per spec.
