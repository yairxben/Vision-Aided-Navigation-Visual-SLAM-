"""Explicit command-line boundary for the Exercise 5 Bundle workflow.

The CLI contains no Bundle science.  It dispatches read-only validation, post-run aggregation/reporting, curated submission materialization,
and separately confirmed scientific computations to their owning modules.
The track-survival commands are retained only for the completed rejected
experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exercise5_aggregation import aggregate_all
from exercise5_full_run import (
    ArtifactValidationError,
    CANONICAL_CONFIRMATION,
    DEFAULT_CONFIG_PATH,
    ROOT,
    run_streamed,
    validate_run_root,
)
from exercise5_reporting import generate_report_figures
from exercise5_submission import materialize_bundle_package
from exercise5_track_survival import (
    EXPERIMENT_CONFIG_PATH,
    EXPERIMENT_CONFIRMATION,
    EXPERIMENT_OUTPUT_ROOT,
    EXPERIMENT_RULE_ID,
    derive_and_persist_experiment_configuration,
    finalize_experimental_run,
    generate_policy_comparison,
)


def _parser() -> argparse.ArgumentParser:
    """Build the explicit canonical and historical command surface."""

    parser = argparse.ArgumentParser(
        description="Canonical, explicit Exercise 5 Bundle workflows"
    )
    subparsers = parser.add_subparsers(dest="command")

    probe = subparsers.add_parser(
        "probe", help="solve exactly the frozen first window under /tmp"
    )
    probe.add_argument("--output-root", type=Path, required=True)

    canonical = subparsers.add_parser(
        "run", help="resume the separately authorized canonical 355-window run"
    )
    canonical.add_argument("--output-root", type=Path, required=True)
    canonical.add_argument(
        "--confirm",
        required=True,
        help="must equal {}".format(CANONICAL_CONFIRMATION),
    )

    derive_experiment = subparsers.add_parser(
        "derive-track-survival-082",
        help="derive and freeze the sole E5-I03C keyframe experiment",
    )
    derive_experiment.set_defaults(
        configuration_path=EXPERIMENT_CONFIG_PATH
    )

    run_experiment = subparsers.add_parser(
        "run-track-survival-082",
        help="resume the frozen E5-I03C experiment under its protected root",
    )
    run_experiment.add_argument(
        "--confirm",
        required=True,
        help="must equal {}".format(EXPERIMENT_CONFIRMATION),
    )

    subparsers.add_parser(
        "finalize-track-survival-082",
        help="aggregate, report, compare, checksum, and strictly validate E5-I03C",
    )

    validate = subparsers.add_parser(
        "validate", help="reload and validate a run root"
    )
    validate.add_argument("--source-root", type=Path, required=True)
    validate.add_argument("--allow-partial", action="store_true")

    aggregate = subparsers.add_parser(
        "aggregate", help="generate frozen aggregate tables"
    )
    aggregate.add_argument("--source-root", type=Path, required=True)
    aggregate.add_argument("--allow-partial", action="store_true")

    report = subparsers.add_parser(
        "report", help="generate figures from aggregate machine-readable tables"
    )
    report.add_argument("--source-root", type=Path, required=True)

    submission = subparsers.add_parser(
        "submission",
        help="materialize the curated layout from a complete sealed source",
    )
    submission.add_argument("--source-root", type=Path, required=True)
    submission.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    """Parse one command, execute its owning workflow, and print JSON output."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "probe":
        # The probe is still a real one-window solve; its /tmp gate is enforced
        # by run_streamed rather than duplicated in this thin dispatcher.
        result = run_streamed(
            output_root=args.output_root,
            maximum_windows=1,
            bounded_probe=True,
        )
        tables = aggregate_all(
            args.output_root,
            repository_root=ROOT,
            allow_partial=True,
        )
        figures = generate_report_figures(args.output_root)
        try:
            materialize_bundle_package(
                args.output_root,
                args.output_root / "rejected_partial_submission",
            )
        except ArtifactValidationError:
            submission_gate = "partial-source-rejected"
        else:
            raise AssertionError("partial probe source was materialized")
        result["aggregate_tables"] = [str(path) for path in tables]
        result["report_figures"] = [str(path) for path in figures]
        result["submission_materializer_gate"] = submission_gate
    elif args.command == "run":
        # The full scientific computation requires the exact confirmation
        # token checked by run_streamed before any output root is created.
        result = run_streamed(
            output_root=args.output_root,
            canonical_confirmation=args.confirm,
        )
    elif args.command == "derive-track-survival-082":
        # These three commands preserve the completed rejected experiment's
        # explicit interface; they are not alternate production workflows.
        result = derive_and_persist_experiment_configuration(
            configuration_path=args.configuration_path
        )
    elif args.command == "run-track-survival-082":
        result = run_streamed(
            output_root=EXPERIMENT_OUTPUT_ROOT,
            configuration_path=EXPERIMENT_CONFIG_PATH,
            canonical_confirmation=args.confirm,
            experimental_policy=EXPERIMENT_RULE_ID,
        )
    elif args.command == "finalize-track-survival-082":
        manifest = json.loads(
            (EXPERIMENT_OUTPUT_ROOT / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if manifest["status"] == "complete-and-sealed":
            result = finalize_experimental_run()
            result["aggregate_tables"] = "already-finalized"
            result["report_figures"] = "already-finalized"
            result["comparison"] = "already-finalized"
        else:
            tables = aggregate_all(
                EXPERIMENT_OUTPUT_ROOT,
                repository_root=ROOT,
                configuration_path=EXPERIMENT_CONFIG_PATH,
                allow_partial=True,
            )
            figures = generate_report_figures(EXPERIMENT_OUTPUT_ROOT)
            comparison = generate_policy_comparison()
            result = finalize_experimental_run()
            result["aggregate_tables"] = [str(path) for path in tables]
            result["report_figures"] = [str(path) for path in figures]
            result["comparison"] = comparison
    elif args.command == "validate":
        result = validate_run_root(
            args.source_root, allow_partial=args.allow_partial
        )
    elif args.command == "aggregate":
        result = {
            "outputs": [
                str(path)
                for path in aggregate_all(
                    args.source_root,
                    repository_root=ROOT,
                    allow_partial=args.allow_partial,
                )
            ]
        }
    elif args.command == "report":
        result = {
            "outputs": [
                str(path) for path in generate_report_figures(args.source_root)
            ]
        }
    elif args.command == "submission":
        result = {
            "destination": str(
                materialize_bundle_package(
                    args.source_root, args.destination
                )
            )
        }
    else:
        parser.error("unknown command")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
