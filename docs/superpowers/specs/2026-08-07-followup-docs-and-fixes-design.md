# Follow-up: bug fix, kornia-style docstrings, README showcase

**Date:** 2026-08-07
**Status:** Approved
**Branch:** `followup-docs-and-fixes` off master (post-#20 merge)

## Goal

One follow-up PR delivering the items parked by the nbdev-migration review:
fix the `draw_LAF_matches_from_result_dict` NameError, populate the API
reference by adding docstrings to the whole public API, and turn the README
into a proper landing page.

## 1. Bug fix (TDD — the only behavior change)

- `kornia_moons/viz.py:231`: `if 'F' in rbesult_dict:` → `if 'F' in result_dict:`.
- Regression test written FIRST in `tests/test_viz.py` (red: today the
  function crashes with `NameError` on every call): build a minimal
  `result_dict` from a real ORB detection + `match_nn` on
  `data/strahov.png`; two cases — with `'F'` key present and absent —
  assert the call completes (and returns fig/ax with `return_fig_ax=True`
  if the function supports it; the implementer reads the function body to
  construct the dict it actually expects).

## 2. Docstrings — kornia-style, whole public API

- Every member of `__all__` in `kornia_moons/feature.py` (15 entries,
  currently zero docstrings) and `kornia_moons/viz.py` (9 entries, three
  have `'''…'''` one-liners) gets a Google-format docstring in kornia's
  style: one-line summary, `Args:` with names/types/defaults, `Returns:`
  with tensor shapes spelled out (LAFs `(1, N, 2, 3)`, responses
  `(1, N, 1)`, etc.).
- The three existing one-liners keep their wording as the summary line,
  upgraded to the full format.
- Where the tutorials (`docs/feature.ipynb`, `docs/viz.ipynb`) demonstrate
  a function, a short `Example:` block in the docstring carries that usage
  (matching the old nbdev docs, which showed usage next to each function).
- **No signature or behavior changes.** Classes get a class-level docstring
  plus `forward` docs. `draw_LAF_inliers_perspective_repjojected` keeps its
  typo'd name — public-API stability; renaming is explicitly out of scope.

## 3. README showcase

- Badges: PyPI version, CI workflow status, Apache-2.0 license.
- "What's inside" overview: keypoint/LAF conversions (ORB/SIFT
  conventions), match conversions, OpenCV-detector wrappers
  (`OpenCVDetectorKornia` etc.), visualization (LAFs, matches, epipolar
  lines, SOLD2 line segments).
- Existing ORB example stays with its three images.
- New second example: SIFT + `kornia.feature.match_nn` +
  `draw_LAF_matches`, with its output image extracted from the executed
  `docs/viz.ipynb` into `docs/assets/laf-matches.png`. README references it
  by absolute raw.githubusercontent URL (renders on PyPI); the snippet is
  consistent with the tutorial code.
- Closing links: tutorials (feature, viz) and API reference on the docs
  site.
- `docs/index.md` mirrors the README with relative `assets/` image links
  and relative tutorial links (`feature/`, `viz/`), as established.

## 4. Local-only cleanup (not part of the PR)

Delete stale nbdev git hooks in `.git/hooks/` (e.g. `post-merge` calling
the nonexistent `nbdev_trust`). Local machine only; nothing committed.

## Verification gate

- `pytest tests -v` — all pass (16 existing + new regression tests).
- `pytest --nbmake docs/feature.ipynb docs/viz.ipynb` — 2/2 (notebooks
  untouched, but they import the modules being edited).
- `mkdocs build` — clean; spot-check the built API page renders the new
  docstrings (Args/Returns/Example sections visible).
- Automated docstring completeness check: every `__all__` member of both
  modules has a non-empty `__doc__`.

## Out of scope

- Renaming `draw_LAF_inliers_perspective_repjojected`.
- `kornia_moons/line.py` (still untracked WIP).
- Any changes to conversion/visualization behavior beyond the one-word fix.
