"""Resolve the ProfileDB source shipped with this SLUICE implementation."""
from pathlib import Path
import sys


def package_root():
    mode = Path(__file__).resolve().parents[1]
    bundled = mode / 'profiling/vendor'
    if (bundled / 'profile_db/api.py').is_file():
        return bundled
    # Prepared participants place the same package at the snapshot import root.
    frozen = mode.parents[1]
    if (frozen / 'profile_db/api.py').is_file():
        return frozen
    raise ImportError('The SLUICE profiling core bundle is missing')


def profile_db_class():
    root = package_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import profile_db
    if Path(profile_db.__file__).resolve().parent != (root / "profile_db").resolve():
        raise ImportError("An unrelated ProfileDB package is already imported")
    from profile_db.api import ProfileDB
    return ProfileDB
