"""Evidence bundle export and verification service (Person 3).

Builds the complete export directory and zip for an approved run:

- ``report.html`` / ``report.json`` rendered by :class:`ReportRenderer`
- lossless target/donor/result stills and the authoritative ``provenance.npz``
- ``policy_decisions.json`` (ordered audit log) and ``trace_summary.json`` (sanitized spans)
- ``manifest.json`` listing the SHA-256 of every file, so the bundle can be re-verified

Export is refused unless :func:`probity.reports.gate.export_blockers` is empty.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from probity.domain.enums import InferenceMode
from probity.domain.models import (
    CaseWorkspace,
    EvidenceReport,
    HumanReview,
    PolicyDecision,
    ReconstructionRun,
    SourceVideo,
)
from probity.ports import ReportNarrative
from probity.reports.gate import export_blockers
from probity.reports.render import ReportRenderer

MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    role: str
    sha256: str
    recorded_sha256: str | None = None


@dataclass
class ExportBundle:
    report: EvidenceReport
    directory: Path
    html_path: Path
    json_path: Path
    zip_path: Path
    manifest: list[ManifestEntry] = field(default_factory=list)


class ExportBlockedError(ValueError):
    """Raised when the review/export gate refuses an export."""


def export_evidence_bundle(
    *,
    case: CaseWorkspace,
    video: SourceVideo,
    run: ReconstructionRun,
    review: HumanReview | None,
    decisions: Sequence[PolicyDecision],
    source_path: Path,
    artifacts: dict[str, tuple[Path, str]],
    trace_spans: Sequence[dict[str, Any]],
    narrative: ReportNarrative,
    narrative_fell_back: bool,
    output_dir: Path,
    source_tampered: bool = False,
    mode: InferenceMode = InferenceMode.FIXTURE,
) -> ExportBundle:
    """Re-verify the source, enforce the gate, and write a verifiable evidence bundle.

    ``artifacts`` maps a bundle filename to ``(source_path, role)``. Roles ``result`` and
    ``provenance`` are checked against the run's recorded hashes before anything is written.
    """
    observed_source = sha256_file(source_path) if source_path.exists() else "0" * 64
    if source_tampered:
        observed_source = hashlib.sha256(observed_source.encode() + b"tampered").hexdigest()
    source_verified = observed_source == video.sha256

    blockers = export_blockers(run, review, source_verified=source_verified)
    if blockers or review is None:
        raise ExportBlockedError("Export blocked: " + "; ".join(b.message for b in blockers))

    recorded = {"result": run.result_png_sha256, "provenance": run.provenance_sha256}
    for name, (path, role) in artifacts.items():
        expected = recorded.get(role)
        if expected is not None and sha256_file(path) != expected:
            raise ExportBlockedError(
                f"Export blocked: {name} on disk does not match the recorded {role} hash."
            )

    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest: list[ManifestEntry] = [
        ManifestEntry(
            path=f"(source) {video.original_name}",
            role="source_video",
            sha256=observed_source,
            recorded_sha256=video.sha256,
        )
    ]
    for name, (path, role) in artifacts.items():
        shutil.copyfile(path, out / name)
        manifest.append(
            ManifestEntry(
                path=name,
                role=role,
                sha256=sha256_file(out / name),
                recorded_sha256=recorded.get(role),
            )
        )

    decisions_path = out / "policy_decisions.json"
    decisions_path.write_text(
        json.dumps([d.model_dump(mode="json") for d in decisions], indent=2), encoding="utf-8"
    )
    manifest.append(
        ManifestEntry("policy_decisions.json", "policy_log", sha256_file(decisions_path))
    )

    trace_path = out / "trace_summary.json"
    trace_path.write_text(json.dumps(list(trace_spans), indent=2, default=str), encoding="utf-8")
    manifest.append(ManifestEntry("trace_summary.json", "trace_summary", sha256_file(trace_path)))

    report, html_path, json_path = ReportRenderer().render_bundle(
        case=case,
        video=video,
        run=run,
        review=review,
        output_dir=out,
        narrative=narrative,
        source_path=None if source_tampered else source_path,
        mode=mode,
        extra_context={
            "decisions": list(decisions),
            "manifest": manifest,
            "narrative_fell_back": narrative_fell_back,
        },
    )
    manifest.append(ManifestEntry("report.html", "report_html", sha256_file(html_path)))
    manifest.append(ManifestEntry("report.json", "report_json", sha256_file(json_path)))

    (out / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "report_id": report.report_id,
                "run_id": run.run_id,
                "review_id": review.review_id,
                "bundle_sha256": report.bundle_sha256,
                "source_sha256_verified_at_export": observed_source,
                "files": [e.__dict__ for e in manifest],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.iterdir()):
            zf.write(f, arcname=f.name)

    return ExportBundle(
        report=report,
        directory=out,
        html_path=html_path,
        json_path=json_path,
        zip_path=zip_path,
        manifest=manifest,
    )


def verify_bundle(directory: Path | str) -> list[str]:
    """Re-hash every file listed in the bundle manifest; returns a list of problems."""
    out = Path(directory)
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    problems: list[str] = []
    for entry in manifest["files"]:
        if entry["path"].startswith("(source)"):
            if entry["sha256"] != entry["recorded_sha256"]:
                problems.append("source hash at export differs from ingest hash")
            continue
        f = out / entry["path"]
        if not f.exists():
            problems.append(f"missing {entry['path']}")
            continue
        actual = sha256_file(f)
        if actual != entry["sha256"]:
            problems.append(f"{entry['path']} hash changed since export")
        if entry.get("recorded_sha256") and actual != entry["recorded_sha256"]:
            problems.append(f"{entry['path']} does not match the reviewed {entry['role']} hash")
    report = json.loads((out / "report.json").read_text())
    if report.get("bundle_sha256") != manifest["bundle_sha256"]:
        problems.append("report.json bundle hash differs from manifest")
    return problems
