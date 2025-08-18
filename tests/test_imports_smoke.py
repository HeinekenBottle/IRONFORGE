import pytest


@pytest.mark.smoke
def test_imports_smoke() -> None:
    # Ensure packages import without side-effects
    import ironforge  # noqa: F401
    import ironforge.analysis  # noqa: F401
    import ironforge.integration  # noqa: F401
    import ironforge.learning  # noqa: F401
    import ironforge.synthesis  # noqa: F401
    import ironforge.utilities  # noqa: F401
    import ironforge.validation  # noqa: F401
