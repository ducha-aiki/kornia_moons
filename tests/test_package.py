import importlib

import pytest

MODULES = ["kornia_moons", "kornia_moons.feature", "kornia_moons.viz"]


@pytest.mark.parametrize("modname", MODULES)
def test_module_has_docstring(modname):
    mod = importlib.import_module(modname)
    assert mod.__doc__ and mod.__doc__.strip()


def test_top_level_reexports():
    import kornia_moons

    assert kornia_moons.laf_from_opencv_kpts is kornia_moons.feature.laf_from_opencv_kpts
    assert kornia_moons.draw_LAF_matches is kornia_moons.viz.draw_LAF_matches
    assert kornia_moons.draw_point_matches is kornia_moons.viz.draw_point_matches


def test_version_attribute():
    import kornia_moons

    assert isinstance(kornia_moons.__version__, str)
