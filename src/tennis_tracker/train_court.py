"""Court keypoint detection training entry point (top-level module).

Shim that delegates to ``tennis_tracker.training.court``.

Usage:
    uv run python -m tennis_tracker.train_court --help
"""

from tennis_tracker.training.court import main

if __name__ == "__main__":
    main()
