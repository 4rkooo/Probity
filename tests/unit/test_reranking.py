"""Rerank exact scores, clamping, tie-breaks, and time-IoU collapse at 0.6 vs >0.6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probity.domain.enums import IndexState, InferenceMode, VisibilityKind
from probity.domain.ids import segment_id
from probity.domain.models import (
    ClassConfidence,
    SearchPlan,
    VideoSegment,
)
from probity.domain.policy import default_policy
from probity.ports import RetrievedSegment
from probity.search.rerank import (
    clamp01,
    collapse_overlaps,
    rerank_candidates,
    score_retrieved,
    time_iou,
)

ROOT = Path(__file__).resolve().parents[2]
QUERIES = json.loads((ROOT / "fixtures/demo/search/queries.json").read_text(encoding="utf-8"))
SEGMENTS = [
    VideoSegment.model_validate(row)
    for row in json.loads(
        (ROOT / "fixtures/contracts/video_segments.json").read_text(encoding="utf-8")
    )
]
VIDEO_ID = SEGMENTS[0].video_id
SOURCE_SHA = SEGMENTS[0].source_sha256
PLAN = SearchPlan.model_validate(QUERIES["prepared_query"]["expected_plan"])


def _seg(
    *,
    ordinal: int,
    start: int,
    end: int,
    description: str = "scene",
    search_terms: tuple[str, ...] = (),
    classes: tuple[ClassConfidence, ...] = (),
    vis: tuple[VisibilityKind, ...] = (VisibilityKind.CLEAR,),
) -> VideoSegment:
    return VideoSegment.create(
        segment_id=segment_id(VIDEO_ID, ordinal),
        video_id=VIDEO_ID,
        source_sha256=SOURCE_SHA,
        ordinal=ordinal,
        start_pts_us=start,
        end_pts_us=end,
        start_frame=0,
        end_frame=1,
        description=description,
        description_model_id="fixture/cosmos-reason-describe-v1",
        search_terms=search_terms,
        detected_classes=classes,
        visibility_tags=vis,
        embedding_ref=f"demo-plate-90s/embeddings.npy#{ordinal}",
        embedding_model_id="fixture/cosmos-embed-v1",
        embedding_dimension=64,
        index_state=IndexState.INDEXED,
        mode=InferenceMode.FIXTURE,
    )


def test_golden_ranks_match_fixture_query_to_1e9() -> None:
    policy = default_policy()
    retrieved = []
    gold = QUERIES["prepared_query"]["golden_ranks"]
    # Rebuild retrieved rows from the golden similarities (vector rank by sim desc).
    by_ordinal = {seg.ordinal: seg for seg in SEGMENTS}
    ordered = sorted(gold, key=lambda row: -row["similarity"])
    for rank, row in enumerate(ordered, start=1):
        retrieved.append(
            RetrievedSegment(
                segment=by_ordinal[row["ordinal"]],
                similarity=row["similarity"],
                vector_rank=rank,
            )
        )
    ranked = rerank_candidates(retrieved, PLAN, policy.search)
    collapsed = collapse_overlaps(ranked, policy.search.overlap_collapse_time_iou)
    kept = [(c, item) for c, item in collapsed if c.collapsed_into is None]
    assert [item.segment.ordinal for _c, item in kept] == [row["ordinal"] for row in gold]
    for (candidate, item), row in zip(kept, gold, strict=True):
        assert item.segment.ordinal == row["ordinal"]
        assert abs(candidate.score - row["score"]) < 1e-9
        assert abs(candidate.similarity - row["similarity"]) < 1e-9
        assert abs(candidate.components.cosine - row["components"]["cosine"]) < 1e-9
        lex = candidate.components.lexical_overlap
        det = candidate.components.detection_match
        vis = candidate.components.visibility_match
        assert abs(lex - row["components"]["lexical_overlap"]) < 1e-9
        assert abs(det - row["components"]["detection_match"]) < 1e-9
        assert abs(vis - row["components"]["visibility_match"]) < 1e-9


def test_cosine_is_clamped() -> None:
    segment = _seg(ordinal=0, start=0, end=1_000_000)
    item = RetrievedSegment(segment=segment, similarity=-0.4, vector_rank=1)
    plan = SearchPlan(semantic_query="road", objective_terms=("road",))
    scored = score_retrieved(item, plan, default_policy().search.weights)
    assert scored.components.cosine == 0.0
    assert clamp01(1.2) == 1.0
    assert clamp01(-0.3) == 0.0
    assert 0.0 <= scored.score <= 1.0


def test_tie_breaks_score_then_sim_then_ordinal() -> None:
    weights = default_policy().search.weights
    plan = SearchPlan(semantic_query="road", objective_terms=())
    a = RetrievedSegment(
        segment=_seg(ordinal=5, start=0, end=1_000_000), similarity=0.5, vector_rank=1
    )
    b = RetrievedSegment(
        segment=_seg(ordinal=1, start=2_000_000, end=3_000_000), similarity=0.5, vector_rank=2
    )
    c = RetrievedSegment(
        segment=_seg(ordinal=2, start=4_000_000, end=5_000_000), similarity=0.9, vector_rank=3
    )
    ranked = rerank_candidates([a, b, c], plan, default_policy().search)
    ordinals = [item.segment.ordinal for _c, item in ranked]
    # Higher cosine/score first (c), then equal scores break by ordinal ascending (1 before 5).
    assert ordinals[0] == 2
    assert ordinals[1:] == [1, 5]
    assert ranked[1][0].score == ranked[2][0].score
    _ = weights


def test_iou_equal_0_6_is_not_collapsed() -> None:
    # [0,10] and [4,10]: inter=6, union=10, iou=0.6
    left = _seg(ordinal=0, start=0, end=10_000_000, description="blue sedan")
    right = _seg(ordinal=1, start=4_000_000, end=10_000_000, description="blue sedan")
    assert abs(time_iou(left, right) - 0.6) < 1e-12
    plan = SearchPlan(semantic_query="sedan", objective_terms=("blue sedan",))
    ranked = rerank_candidates(
        [
            RetrievedSegment(segment=left, similarity=0.9, vector_rank=1),
            RetrievedSegment(segment=right, similarity=0.8, vector_rank=2),
        ],
        plan,
        default_policy().search,
    )
    collapsed = collapse_overlaps(ranked, 0.6)
    kept = [item.segment.ordinal for c, item in collapsed if c.collapsed_into is None]
    assert kept == [0, 1]


def test_iou_greater_than_0_6_is_collapsed() -> None:
    # [0,10] and [3,10]: inter=7, union=10, iou=0.7
    left = _seg(ordinal=0, start=0, end=10_000_000, description="blue sedan")
    right = _seg(ordinal=1, start=3_000_000, end=10_000_000, description="blue sedan")
    assert time_iou(left, right) > 0.6
    plan = SearchPlan(semantic_query="sedan", objective_terms=("blue sedan",))
    ranked = rerank_candidates(
        [
            RetrievedSegment(segment=left, similarity=0.9, vector_rank=1),
            RetrievedSegment(segment=right, similarity=0.85, vector_rank=2),
        ],
        plan,
        default_policy().search,
    )
    collapsed = collapse_overlaps(ranked, 0.6)
    kept = [(c, item) for c, item in collapsed if c.collapsed_into is None]
    dropped = [(c, item) for c, item in collapsed if c.collapsed_into is not None]
    assert [item.segment.ordinal for _c, item in kept] == [0]
    assert dropped[0][1].segment.ordinal == 1
    assert dropped[0][0].collapsed_into == left.segment_id


def test_unavailable_components_listed_as_zero() -> None:
    segment = _seg(
        ordinal=0,
        start=0,
        end=1_000_000,
        vis=(),
        classes=(),
        description="empty",
    )
    item = RetrievedSegment(segment=segment, similarity=0.4, vector_rank=1)
    scored = score_retrieved(item, SearchPlan(semantic_query="x"), default_policy().search.weights)
    assert scored.components.lexical_overlap == 0.0
    assert scored.components.detection_match == 0.0
    assert scored.components.visibility_match == 0.0
    assert "lexical_overlap" in scored.components.unavailable
    assert "detection_match" in scored.components.unavailable
    assert "visibility_match" in scored.components.unavailable


def test_s0001_s0002_overlap_below_collapse() -> None:
    a, b = SEGMENTS[1], SEGMENTS[2]
    # 2s overlap on 8s windows => 2/14 < 0.6
    assert time_iou(a, b) < 0.6
    np.testing.assert_allclose(time_iou(a, b), 2 / 14, atol=1e-12)
