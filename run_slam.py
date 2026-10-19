"""Run the generic stereo SLAM pipeline.

Example:
    python run_slam.py --dataset-root dataset --sequence 00 --output-root outputs/seq00

Equivalent to ``python -m generic_slam``; see ``--help`` for all options.
"""

from generic_slam.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
