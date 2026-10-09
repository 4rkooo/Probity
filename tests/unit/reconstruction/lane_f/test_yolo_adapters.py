"""Lane F unit tests: fixture replay, CoreWeave refusal, and no-network / no-download guards."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from probity.adapters.fixture.yolo import FixtureYoloAdapter
from probity.adapters.live import yolo as live_yolo
from probity.adapters.live.yolo import LiveYoloAdapter, NotConfigured, ultralytics_available
from probity.domain.enums import AdapterMode, HealthStatus, SubjectType
from probity.domain.errors import FixtureNotFound, ValidationFailed
from probity.domain.ids import parse_frame_id
from probity.domain.models import Track
from probity.eval.window import load_window
from probity.ports import DetectorTracker, TrackRequest

REPO = Path(__file__).resolve().parents[4]
SYNTH = REPO / "fixtures" / "synthetic" / "plate_translate_v1"
CONTRACTS = REPO / "fixtures" / "contracts"


def _synth_request():
    win = load_window(SYNTH)
    seed = next(det for det in win.detections if det.frame_id == win.target_frame_id)
    request = TrackRequest(
        track_id=win.track_id,
        case_id=win.case_id,
        video_id=win.video_id,
        source_sha256=win.source_sha256,
        subject_type=SubjectType.LICENSE_PLATE,
        seed_frame_id=win.target_frame_id,
        seed_bbox_px=win.target_bbox_px,
        seed_detection_id=seed.detection_id,
    )
    return win, request


def test_fixture_adapter_is_runtime_checkable() -> None:
    assert isinstance(FixtureYoloAdapter(SYNTH), DetectorTracker)


def test_fixture_replays_synth_detections_and_track() -> None:
    win, request = _synth_request()
    adapter = FixtureYoloAdapter(SYNTH)
    assert asyncio.run(adapter.detect(win.frames, {"license_plate"})) == win.detections
    assert asyncio.run(adapter.track(request)) == win.track
    assert adapter.mode is AdapterMode.FIXTURE
    health = asyncio.run(adapter.health())
    assert health.status is HealthStatus.OK


def test_fixture_output_is_deterministic() -> None:
    win, request = _synth_request()
    adapter = FixtureYoloAdapter(SYNTH)
    first = asyncio.run(adapter.detect(win.frames, {"license_plate"}))
    second = asyncio.run(adapter.detect(win.frames, {"license_plate"}))
    assert first == second
    assert asyncio.run(adapter.track(request)) == asyncio.run(adapter.track(request))


def test_fixture_reads_contract_tracks_by_id() -> None:
    confirmed = Track.model_validate(json.loads(
        (CONTRACTS / "track_confirmed.json").read_text(encoding="utf-8")
    ))
    adapter = FixtureYoloAdapter(CONTRACTS)
    request = TrackRequest(
        track_id=confirmed.track_id,
        case_id=confirmed.case_id,
        video_id=confirmed.video_id,
        source_sha256="d31ad69e9591d11df9dd7a1a4d7301d9092790c3868a547430879f85f5addf05",
        subject_type=confirmed.subject_type,
        seed_frame_id=confirmed.seed_frame_id,
        seed_bbox_px=confirmed.seed_bbox_px,
        seed_detection_id=confirmed.seed_detection_id,
    )
    assert asyncio.run(adapter.track(request)) == confirmed


def test_fixture_missing_root_is_unavailable(tmp_path: Path) -> None:
    adapter = FixtureYoloAdapter(tmp_path / "missing")
    health = asyncio.run(adapter.health())
    assert health.status is HealthStatus.UNAVAILABLE
    with pytest.raises(FixtureNotFound):
        asyncio.run(adapter.detect((), {"license_plate"}))
    with pytest.raises(FixtureNotFound):
        asyncio.run(adapter.track(TrackRequest(
            track_id="01a1219b-7a81-7637-9c64-97d31e3071ff",
            case_id="01a1219b-7a80-71a9-89ee-ba5341bea516",
            video_id="01a1219b-7a80-7bd1-b62c-670da892173e",
            source_sha256="d31ad69e9591d11df9dd7a1a4d7301d9092790c3868a547430879f85f5addf05",
            subject_type=SubjectType.LICENSE_PLATE,
            seed_frame_id="01a1219b-7a80-7bd1-b62c-670da892173e:f37",
            seed_bbox_px=(86, 111, 306, 183),
        )))


def test_fixture_detect_filters_frames_and_empty_classes() -> None:
    win, _request = _synth_request()
    adapter = FixtureYoloAdapter(SYNTH)
    one = win.frames[:1]
    found = asyncio.run(adapter.detect(one, {"license_plate"}))
    assert found
    assert all(det.frame_id == one[0].frame_id for det in found)
    assert asyncio.run(adapter.detect(win.frames, set())) == ()


def test_fixture_track_matches_seed_when_id_differs() -> None:
    win, request = _synth_request()
    adapter = FixtureYoloAdapter(SYNTH)
    alt = request.model_copy(update={"track_id": "01a1219b-7a81-7637-9c64-97d31e307100"})
    assert asyncio.run(adapter.track(alt)) == win.track
    missing = alt.model_copy(update={"seed_bbox_px": (1, 2, 3, 4)})
    with pytest.raises(FixtureNotFound):
        asyncio.run(adapter.track(missing))


def test_fixture_reads_tracks_subdirectory(tmp_path: Path) -> None:
    win, request = _synth_request()
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "one.json").write_text(
        (SYNTH / "track.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    adapter = FixtureYoloAdapter(tmp_path)
    assert asyncio.run(adapter.track(request)) == win.track


def test_fixture_never_reads_ground_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    win, _request = _synth_request()
    forbidden = SYNTH / "ground_truth"

    original = Path.read_text

    def guarded(self: Path, *args: object, **kwargs: object) -> str:
        if forbidden in self.parents or self == forbidden:
            raise AssertionError(f"fixture adapter read evaluation-only path {self}")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    adapter = FixtureYoloAdapter(SYNTH)
    asyncio.run(adapter.detect(win.frames, {"license_plate"}))


def test_live_health_never_raises_without_ultralytics() -> None:
    adapter = LiveYoloAdapter()
    health = asyncio.run(adapter.health())
    assert health.status is HealthStatus.DISABLED
    assert health.mode is AdapterMode.DISABLED
    assert isinstance(adapter, DetectorTracker)


def test_live_coreweave_raises_not_configured() -> None:
    adapter = LiveYoloAdapter(coreweave=True)
    with pytest.raises(NotConfigured, match="CoreWeave"):
        adapter.run_on_coreweave()
    with pytest.raises(NotConfigured, match="CoreWeave"):
        asyncio.run(adapter.detect((), set()))


def test_live_refuses_hub_name_without_local_file() -> None:
    adapter = LiveYoloAdapter(weights_path="yolov8n.pt")
    with pytest.raises(NotConfigured, match="checkpoint|download"):
        asyncio.run(adapter.detect((), {"license_plate"}))
    assert not Path("yolov8n.pt").exists()


def test_live_module_does_not_import_ultralytics() -> None:
    tree = ast.parse(Path(live_yolo.__file__).read_text(encoding="utf-8"))
    top_level: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module.split(".", 1)[0])
    assert "ultralytics" not in top_level
    sys.modules.pop("ultralytics", None)
    assert ultralytics_available() is (importlib.util.find_spec("ultralytics") is not None)
    assert "ultralytics" not in sys.modules


def test_live_track_requires_bound_frames_and_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "local.pt"
    weights.write_bytes(b"not-a-real-checkpoint")
    monkeypatch.setattr(live_yolo, "ultralytics_available", lambda: True)
    win, request = _synth_request()
    bare = LiveYoloAdapter(weights_path=weights)
    with pytest.raises(ValidationFailed, match="frames"):
        asyncio.run(bare.track(request))
    no_loader = LiveYoloAdapter(weights_path=weights, frames=win.frames)
    with pytest.raises(ValidationFailed, match="frame loader"):
        asyncio.run(no_loader.track(request))
    with pytest.raises(ValidationFailed, match="frame loader"):
        asyncio.run(no_loader.detect(win.frames[:1], {"license_plate"}))


def test_live_detect_sorts_by_frame_number(monkeypatch: pytest.MonkeyPatch) -> None:
    win, _request = _synth_request()
    by_id = {det.frame_id: det for det in win.detections}
    numbered = sorted(
        (parse_frame_id(frame_id)[1], det) for frame_id, det in by_id.items()
    )
    earlier = numbered[0][1]
    later = next(det for number, det in numbered if number >= 10)
    assert parse_frame_id(later.frame_id)[1] > parse_frame_id(earlier.frame_id)[1]

    def fake_predict(
        self: LiveYoloAdapter, ref: object, *, track: bool
    ) -> list[tuple[object, None]]:
        del self, track
        det = by_id.get(getattr(ref, "frame_id"))
        return [(det, None)] if det is not None else []

    monkeypatch.setattr(LiveYoloAdapter, "_require_local", lambda self: None)
    monkeypatch.setattr(LiveYoloAdapter, "_predict_boxes", fake_predict)
    frames = [ref for ref in win.frames if ref.frame_id in {earlier.frame_id, later.frame_id}]
    frames.sort(key=lambda ref: ref.frame_number, reverse=True)
    found = asyncio.run(LiveYoloAdapter().detect(frames, {"license_plate"}))
    assert [det.detection_id for det in found] == [earlier.detection_id, later.detection_id]
