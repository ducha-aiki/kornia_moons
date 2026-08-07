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
