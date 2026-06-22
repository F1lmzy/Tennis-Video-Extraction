"""Smoke test for the tennis_tracker package."""

from tennis_tracker import __version__


def test_version() -> None:
    """Package exposes a version string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_cli_imports() -> None:
    """CLI module imports without error."""
    from tennis_tracker.cli import main  # noqa: F811

    assert main is not None
