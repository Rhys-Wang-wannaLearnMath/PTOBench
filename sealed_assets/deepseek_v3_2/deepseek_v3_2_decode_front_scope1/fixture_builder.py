"""Generate scope1 inputs with the frozen official initializer."""
import common


def build(case_dir, sealed_dir, case, seed):
    return common.load_official_fixture_specs(
        sealed_dir / "official/original.py", "scope1_official", seed
    )
