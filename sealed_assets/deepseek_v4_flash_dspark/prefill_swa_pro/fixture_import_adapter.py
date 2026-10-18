"""Reviewed PRO Fixture/CPU import support; never an NPU implementation."""
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

SELECTED_HC_PRE_SHA256 = "b5e347ed55eae5e7678c40bce3b9212c1b715419589a09b7edbcdf2a922755ee"


@contextmanager
def pro_fixture_import(official_root):
    helper = Path(official_root).resolve() / "hc_pre.py"
    source = helper.read_bytes()
    if hashlib.sha256(source).hexdigest() != SELECTED_HC_PRE_SHA256:
        raise ValueError("Unrecognized historical HC helper; refuse import adaptation")
    old = b"RMS_OK = 16\n"
    if source.count(old) != 1:
        raise ValueError("Historical RMS split binding is not unique")
    adapted = source.replace(old, b"RMS_OK = 8\n")

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            exec(compile(adapted, str(helper), "exec"), module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "hc_pre":
                return importlib.util.spec_from_file_location(fullname, helper, loader=Loader())
            return None

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
