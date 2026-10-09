"""Same DetectorTracker contract for the fixture adapter and the live YOLO shell.

Live inference that needs ultralytics and a local checkpoint is skippable and never downloads
weights. CoreWeave is tested as NotConfigured in the companion live-shell cases.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from probity.adapters.fixture.yolo import FixtureYoloAdapter
from probity.adapters.live.yolo import LiveYoloAdapter, NotConfigured, ultralytics_available
from probity.domain.enums import AdapterMode, HealthStatus, ObservationSource, SubjectType
from probity.domain.models import AdapterHealth, Detection, Track
from probity.eval.window import load_window
from probity.ports import DetectorTracker, TrackRequest

REPO = Path(__file__).resolve().parents[2]
SYNTH = REPO / "fixtures" / "synthetic" / "plate_translate_v1"
OCR_KEYS = frozenset({"quoted_text", "plate_text", "ocr", "ocr_text", "text", "characters"})
PLATE_CLASS_NAMES = frozenset({"license_plate", "traffic_sign", "vehicle", "sign", "plate"})


def _live_can_infer() -> bool:
    weights = Path(os.environ.get("PROBITY_YOLO_WEIGHTS", ""))
    return ultralytics_available() and weights.is_file()


def _fixture_adapter() -> FixtureYoloAdapter:
    return FixtureYoloAdapter(SYNTH)


def _live_adapter() -> LiveYoloAdapter:
    if not _live_can_infer():
        pytest.skip("ultralytics and a local YOLO checkpoint are required; live inference skipped")
    return LiveYoloAdapter(weights_path=Path(os.environ["PROBITY_YOLO_WEIGHTS"]))


_FACTORY_IDS = ("fixture", "live")
_FACTORIES = (_fixture_adapter, _live_adapter)


def _window():
    return load_window(SYNTH)


def _request(win) -> TrackRequest:
    seed = next(det for det in win.detections if det.frame_id == win.target_frame_id)
    return TrackRequest(
        track_id=win.track_id,
        case_id=win.case_id,
        video_id=win.video_id,
        source_sha256=win.source_sha256,
        subject_type=SubjectType.LICENSE_PLATE,
        seed_frame_id=win.target_frame_id,
        seed_bbox_px=win.target_bbox_px,
        seed_detection_id=seed.detection_id,
    )


def _assert_no_ocr(record: Detection | Track) -> None:
    payload = record.model_dump()
    assert OCR_KEYS.isdisjoint(payload)
    dumped = str(payload)
    assert "quoted_text" not in dumped
    if isinstance(record, Detection):
        assert " " not in record.class_name
        assert not any(ch.isdigit() for ch in record.class_name)
        assert record.class_name in PLATE_CLASS_NAMES or record.class_name.startswith("class_")


def _assert_health(health: AdapterHealth, adapter: DetectorTracker) -> None:
    assert health.adapter_name == adapter.adapter_name
    assert health.schema_version == adapter.schema_version == "1.0"
    assert health.mode is adapter.mode
    assert health.status in HealthStatus
    assert health.latency_ms >= 0


@pytest.mark.parametrize("factory", _FACTORIES, ids=_FACTORY_IDS)
def test_adapter_satisfies_detector_tracker_protocol(
    factory: Callable[[], DetectorTracker],
) -> None:
    adapter = factory()
    assert isinstance(adapter, DetectorTracker)
    assert adapter.adapter_name == "yolo-bytetrack"
    assert adapter.schema_version == "1.0"
    assert adapter.mode in {AdapterMode.FIXTURE, AdapterMode.LIVE, AdapterMode.DISABLED}


@pytest.mark.parametrize("factory", _FACTORIES, ids=_FACTORY_IDS)
def test_health_never_raises(factory: Callable[[], DetectorTracker]) -> None:
    adapter = factory()
    health = asyncio.run(adapter.health())
    _assert_health(health, adapter)


@pytest.mark.parametrize("factory", _FACTORIES, ids=_FACTORY_IDS)
def test_detect_and_track_share_domain_contract(
    factory: Callable[[], DetectorTracker],
) -> None:
    adapter = factory()
    win = _window()
    detections = asyncio.run(adapter.detect(win.frames, {"license_plate"}))
    assert detections
    assert all(isinstance(det, Detection) for det in detections)
    for det in detections:
        _assert_no_ocr(det)
        assert det.video_id == win.video_id
        assert 0.0 <= det.confidence <= 1.0
        assert det.class_name == "license_plate"
    again = asyncio.run(adapter.detect(win.frames, {"license_plate"}))
    assert again == detections

    empty = asyncio.run(adapter.detect(win.frames, {"vehicle"}))
    assert empty == ()

    track = asyncio.run(adapter.track(_request(win)))
    assert isinstance(track, Track)
    _assert_no_ocr(track)
    assert track.video_id == win.video_id
    assert track.tracker == "bytetrack"
    assert track.window_end_us > track.window_start_us
    assert all(0.0 <= obs.confidence <= 1.0 for obs in track.observations)
    assert track.detector_observation_count == sum(
        1 for obs in track.observations
        if obs.accepted and obs.source is ObservationSource.DETECTOR
    )


def test_fixture_replays_synthetic_goldens() -> None:
    win = _window()
    adapter = FixtureYoloAdapter(SYNTH)
    detections = asyncio.run(adapter.detect(win.frames, {"license_plate"}))
    assert detections == win.detections
    assert asyncio.run(adapter.track(_request(win))) == win.track


def test_live_coreweave_is_not_configured() -> None:
    adapter = LiveYoloAdapter(coreweave=True, weights_path=Path("unused.pt"))
    health = asyncio.run(adapter.health())
    assert health.status is HealthStatus.UNAVAILABLE
    assert health.mode is AdapterMode.DISABLED
    assert health.detail is not None and "CoreWeave" in health.detail
    with pytest.raises(NotConfigured, match="CoreWeave"):
        adapter.run_on_coreweave()
    with pytest.raises(NotConfigured, match="CoreWeave"):
        asyncio.run(adapter.detect((), {"license_plate"}))
    request = TrackRequest(
        track_id="01a1219b-7a81-7637-9c64-97d31e3071ff",
        case_id="01a1219b-7a80-71a9-89ee-ba5341bea516",
        video_id="01a1219b-7a80-7bd1-b62c-670da892173e",
        source_sha256="d31ad69e9591d11df9dd7a1a4d7301d9092790c3868a547430879f85f5addf05",
        subject_type=SubjectType.LICENSE_PLATE,
        seed_frame_id="01a1219b-7a80-7bd1-b62c-670da892173e:f37",
        seed_bbox_px=(86, 111, 306, 183),
    )
    with pytest.raises(NotConfigured, match="CoreWeave"):
        asyncio.run(adapter.track(request))


def test_live_without_checkpoint_is_not_configured() -> None:
    adapter = LiveYoloAdapter(weights_path=Path("yolov8n.pt"))
    health = asyncio.run(adapter.health())
    assert health.status is HealthStatus.DISABLED
    detail = (health.detail or "").lower()
    assert "download" in detail or "checkpoint" in detail
    with pytest.raises(NotConfigured):
        asyncio.run(adapter.detect((), {"license_plate"}))
    assert not Path("yolov8n.pt").exists()
    assert ultralytics_available() or health.detail is not None
