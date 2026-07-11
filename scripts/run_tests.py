"""Run the lightweight test suite without requiring pytest on the GPU host."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    count = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        namespace = runpy.run_path(str(path))
        tests = [
            (name, function)
            for name, function in namespace.items()
            if name.startswith("test_") and callable(function)
        ]
        for _, function in tests:
            function()
            count += 1
        print(f"{path.name}: {len(tests)} passed")
    print(f"{count} tests passed")


if __name__ == "__main__":
    main()
