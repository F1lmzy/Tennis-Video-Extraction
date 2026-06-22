"""Ball detection training entry point (top-level module).

Shim that delegates to ``tennis_tracker.training.ball``.

Usage:
    uv run python -m tennis_tracker.train_ball --help
"""

from tennis_tracker.training.ball import main

if __name__ == "__main__":
    main()
