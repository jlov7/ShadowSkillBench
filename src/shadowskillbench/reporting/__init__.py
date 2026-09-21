"""Evidence-bound report helpers."""

# The lazy hash shim must precede report imports: analysis.sealed imports this
# package while the report module is still being initialised.
# ruff: noqa: E402


def hash_analysis_dataset(dataset: object) -> str:
    """Resolve the report hash lazily while analysis package initialises."""
    from shadowskillbench.reporting.report import hash_analysis_dataset as _hash

    return _hash(dataset)  # type: ignore[arg-type]


from shadowskillbench.reporting.report import (
    ProtocolEvidence,
    ReportEvidence,
    ReportInputError,
    ReportStatus,
    RepresentativeTrace,
    dataset_projection,
    render_report,
    render_report_artifacts,
    report_status,
    sealed_analysis_is_release_complete,
    select_representative_traces,
    write_report_artifacts,
)
from shadowskillbench.reporting.selection import (
    SelectionCandidate,
    SelectionInputError,
    SelectionResult,
    SelectionStatus,
    select_stage_a,
    select_stage_b,
)
from shadowskillbench.reporting.workbench_export import (
    WORKBENCH_SCHEMA,
    WORKBENCH_SCHEMA_VERSION,
    PublicEpisodeTrace,
    PublicTraceEvent,
    WorkbenchDataset,
    WorkbenchExport,
    WorkbenchExportError,
    build_workbench_export,
    render_workbench_json,
    validate_workbench_export,
    write_workbench_export,
)

__all__ = [
    "SelectionCandidate",
    "SelectionInputError",
    "SelectionResult",
    "SelectionStatus",
    "WORKBENCH_SCHEMA",
    "WORKBENCH_SCHEMA_VERSION",
    "PublicEpisodeTrace",
    "PublicTraceEvent",
    "ProtocolEvidence",
    "ReportEvidence",
    "ReportInputError",
    "ReportStatus",
    "RepresentativeTrace",
    "WorkbenchDataset",
    "WorkbenchExport",
    "WorkbenchExportError",
    "build_workbench_export",
    "dataset_projection",
    "hash_analysis_dataset",
    "render_report",
    "render_report_artifacts",
    "render_workbench_json",
    "report_status",
    "sealed_analysis_is_release_complete",
    "select_stage_a",
    "select_stage_b",
    "select_representative_traces",
    "validate_workbench_export",
    "write_workbench_export",
    "write_report_artifacts",
]
