# Fix to_torch list handling and max_kpts semantics

**Date:** 2026-08-07
**Status:** Approved
**Branch:** `fix-to-torch-and-max-kpts` off master (post-#21 merge)

## Goal

Fix the two pre-existing bugs documented (but deliberately not fixed) by
PR #21, with TDD regression tests:

1. `to_torch` cannot accept plain Python lists (dead `type(x) is List`
   check against `typing.List`).
2. `max_kpts=-1` (default) with `make_upright=True` silently drops the
   last keypoint (`kpts[:min(len(kpts), -1)]` → `kpts[:-1]`).

## 1. `to_torch` (kornia_moons/feature.py)

- Replace `if type(x) is List:` with `if isinstance(x, list):`.
- Lists convert via the existing `torch.tensor(x)` branch; np.ndarray and
  torch.Tensor paths unchanged; other types still raise `TypeError`.
- Docstring updated: remove the `.. note::` quirk block; `Args`/`Raises`
  state lists are accepted.
- Tests (red first): a Python list converts with correct values; `42`
  still raises `TypeError`.

## 2. `max_kpts` (both `OpenCVDetectorKornia` and `OpenCVDetectorWithAffNetKornia`)

New semantics, identical in both classes (user decision: applies always,
not only under `make_upright`):

- `max_kpts <= 0` (default `-1`): keep ALL keypoints.
- `max_kpts > 0`: keep the `max_kpts` highest-response keypoints,
  regardless of `make_upright`:
  - upright path: `make_keypoints_upright` already returns
    response-descending order — truncate after it as today (minus the -1
    bug).
  - non-upright path: sort keypoints by response descending, then
    truncate. (These `forward`s handle keypoints only — no descriptor
    arrays to desync.)
- Docstrings in both classes rewritten for the new behavior; the
  "drops the final keypoint" warning is removed.
- Tests (red first):
  - `make_upright=True, max_kpts=-1`: keypoint count equals the full
    deduped set (regression for the drop).
  - `max_kpts=10` yields exactly 10 LAFs in both upright and non-upright
    modes.
  - Non-upright survivors are the top-10 by response.
  - The existing AffNet test gains a `max_kpts` assertion (no new
    heavyweight test).

## Ripples

- `tests/test_docstrings.py` unaffected (docstrings shrink, don't vanish).
- Tutorial notebooks don't set `max_kpts`; committed outputs stay valid.
  nbmake still gates in CI.
- Version stays `0.2.10.dev0`; ships with the next release.

## Verification gate

- `pytest tests -v` — all pass (20 existing + new).
- `pytest --nbmake docs/feature.ipynb docs/viz.ipynb` — 2/2.
- `mkdocs build` — clean.

## Out of scope

- `OpenCVFeatureKornia` (has no `max_kpts` parameter).
- `kornia_moons/line.py` (untracked WIP), any renames, tuple support in
  `to_torch`.
