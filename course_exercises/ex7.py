"""Safe command-line dispatch for the clean Exercise 7 lane.

The Loop Closure algorithm is not implemented here.  Its scientific stages
remain in their validated modules: candidate detection -> four-view consensus
matching -> two-pose Bundle measurement -> cumulative Pose Graph update ->
sequential sealing.  This entrypoint gives those stages one provenance-aware
command surface:

* ``preflight`` checks selected E5/E6 inputs and the E7 output lane without
  writing or running Exercise 7 science;
* explicitly confirmed ``run`` delegates to the sequential scientific runner;
* ``validate`` independently replays already sealed scientific evidence;
* ``report`` reads sealed evidence to produce presentation artifacts; and
* ``submission`` materializes the curated package from the report freeze.

``all`` performs validation, report generation, and submission in that
order; it does not include ``run``.  Every delegated command receives the same
selected E5, E6, and E7 roots so clean and historical provenance cannot be
mixed accidentally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_E5_ROOT = REPOSITORY_ROOT / "diagnostics/ex5_clean_final"
DEFAULT_E6_ROOT = REPOSITORY_ROOT / "diagnostics/ex6_clean_final"
DEFAULT_SCIENTIFIC_ROOT = REPOSITORY_ROOT / "diagnostics/ex7_clean_final"
DEFAULT_REPORT_ROOT = DEFAULT_SCIENTIFIC_ROOT / "report_freeze"
DEFAULT_SUBMISSION_ROOT = (
    REPOSITORY_ROOT / "project_submission/loop_closure_clean_final"
)
CANONICAL_RUN_CONFIRMATION = "I_UNDERSTAND_THIS_RUNS_E7_CLEAN_FINAL"


def preflight_exercise7(
    *,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
    scientific_root: Path = DEFAULT_SCIENTIFIC_ROOT,
) -> Mapping[str, object]:
    """Delegate the write-free input/output-lane gate without loading eagerly.

    This checks whether the selected E5/E6 dependency pair and E7 destination
    form a coherent future-run contract.  It performs no matching, Bundle,
    Pose Graph optimization, Marginals construction, or artifact write.
    """

    from exercise7_sequential_loop_closure import preflight_exercise7

    return preflight_exercise7(
        e5_root=e5_root,
        e6_root=e6_root,
        output_root=scientific_root,
    )


def generate_report_freeze(
    *,
    output_root: Path,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
    scientific_root: Path = DEFAULT_SCIENTIFIC_ROOT,
) -> Mapping[str, object]:
    """Delegate report-only generation from already sealed E5/E6/E7 evidence."""

    from exercise7_report_freeze import generate_report_freeze as generate

    return generate(
        output_root=output_root,
        e5_root=e5_root,
        e6_root=e6_root,
        scientific_root=scientific_root,
    )


def materialize_loop_closure_submission(
    *,
    report_root: Path,
    output_root: Path,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
    scientific_root: Path = DEFAULT_SCIENTIFIC_ROOT,
) -> Mapping[str, object]:
    """Delegate curated submission publication from a validated report freeze."""

    from exercise7_report_freeze import (
        materialize_loop_closure_submission as materialize,
    )

    return materialize(
        report_root=report_root,
        output_root=output_root,
        e5_root=e5_root,
        e6_root=e6_root,
        scientific_root=scientific_root,
    )


def validate_canonical_exercise7(
    *,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
    scientific_root: Path = DEFAULT_SCIENTIFIC_ROOT,
) -> Mapping[str, object]:
    """Delegate read-only scientific validation to the sealed-run replayer."""

    from exercise7_sequential_loop_closure import validate_complete_run

    return validate_complete_run(
        output_root=scientific_root,
        e5_root=e5_root,
        e6_root=e6_root,
    )


def _selected_roots(
    arguments: argparse.Namespace,
) -> Tuple[Path, Path, Path]:
    """Read root arguments while preserving older direct Namespace callers."""

    return (
        Path(getattr(arguments, "e5_root", DEFAULT_E5_ROOT)),
        Path(getattr(arguments, "e6_root", DEFAULT_E6_ROOT)),
        Path(
            getattr(
                arguments, "scientific_root", DEFAULT_SCIENTIFIC_ROOT
            )
        ),
    )


def dispatch_command(arguments: argparse.Namespace) -> Mapping[str, object]:
    """Dispatch one parsed command without embedding stage algorithms.

    The branches below define orchestration responsibility only.  In
    particular, ``run`` is the sole scientific command and is guarded by the
    exact confirmation token; reporting and submission operate on sealed
    results and do not silently trigger a scientific rerun.
    """

    e5_root, e6_root, scientific_root = _selected_roots(arguments)
    if arguments.command == "preflight":
        return preflight_exercise7(
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
        )
    if arguments.command == "validate":
        return validate_canonical_exercise7(
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
        )
    if arguments.command == "report":
        return generate_report_freeze(
            output_root=arguments.report_root,
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
        )
    if arguments.command == "submission":
        generate_report_freeze(
            output_root=arguments.report_root,
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
        )
        return materialize_loop_closure_submission(
            report_root=arguments.report_root,
            output_root=arguments.submission_root,
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
        )
    if arguments.command == "all":
        results = {}
        results["validate"] = validate_canonical_exercise7(
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
        )
        results["report"] = generate_report_freeze(
            output_root=arguments.report_root,
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
        )
        results["submission"] = materialize_loop_closure_submission(
            report_root=arguments.report_root,
            output_root=arguments.submission_root,
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
        )
        return {"command": "all", "status": "pass", "results": results}
    if arguments.command == "run":
        if arguments.confirm != CANONICAL_RUN_CONFIRMATION:
            raise RuntimeError(
                "run refused: pass --confirm {}".format(
                    CANONICAL_RUN_CONFIRMATION
                )
            )
        from exercise7_sequential_loop_closure import (
            run_sequential_loop_closure,
        )

        return run_sequential_loop_closure(
            output_root=scientific_root,
            e5_root=e5_root,
            e6_root=e6_root,
        )
    raise RuntimeError("unsupported Exercise 7 command")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command surface while leaving all stage logic in its owner."""

    parser = argparse.ArgumentParser(
        description=(
            "Exercise 7: validate the sealed run, then generate "
            "report-ready and curated submission evidence"
        )
    )
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "validate",
            "report",
            "submission",
            "all",
            "run",
        ),
        help=(
            "preflight: write-free clean-run gate; "
            "validate: independent sealed-run replay; "
            "report: non-destructive report freeze; submission: curated "
            "package; all: validation then report then submission; "
            "run: explicitly confirmed delegation to the canonical runner"
        ),
    )
    parser.add_argument("--e5-root", type=Path, default=DEFAULT_E5_ROOT)
    parser.add_argument("--e6-root", type=Path, default=DEFAULT_E6_ROOT)
    parser.add_argument(
        "--scientific-root", type=Path, default=DEFAULT_SCIENTIFIC_ROOT
    )
    parser.add_argument(
        "--report-root", type=Path, default=DEFAULT_REPORT_ROOT
    )
    parser.add_argument(
        "--submission-root", type=Path, default=DEFAULT_SUBMISSION_ROOT
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=(
            "required only for run; exact token: "
            + CANONICAL_RUN_CONFIRMATION
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, safely dispatch, and print one JSON result."""

    arguments = build_argument_parser().parse_args(argv)
    result = dispatch_command(arguments)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
