#!/usr/bin/env python
"""Print the compute-device report (CPU / CUDA / MPS) for this machine.

    uv run python code/check_devices.py

Use ``--device auto`` (the default) to let every runner pick the best available
device, or ``--device cpu`` / ``--device cuda`` / ``--device cuda:1`` to pin one.
"""
from __future__ import annotations

from ids_defense_selection import device_report


def main() -> None:
    print(device_report())


if __name__ == "__main__":
    main()
