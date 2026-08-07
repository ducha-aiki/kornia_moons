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
