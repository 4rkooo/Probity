"""Lane D provenance writer: LUT, full-frame arrays, and the never-partial record."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from probity.domain.enums import AlignmentMethod, Interpolation, ProvenanceClass, ReasonCode, SourceRole
from probity.domain.policy import PolicyConfig
from probity.reconstruction.io import sha256_bytes
from probity.reconstruction.provenance import (
    DonorLineageError,
    GeneratedPixelError,
    ProvenanceIncomplete,
    build_provenance_arrays,
    build_source_lut,
    encode_provenance,
    provenance_record,
    validate_arrays,
)
from probity.reconstruction.types import FusionResult, TileDecision

from ._builders import (
    OTHER_TRACK,
    OTHER_VIDEO,
    RUN,
    START,
    SUBJECT,
    W,
    H,
    build_scene,
    make_donor,
    npz_uri,
    scenario,
    single_donor,
    view,
)


def _lut_and_arrays(built, policy: PolicyConfig):
    lut = build_source_lut(built.target, built.donors, built.ids)
    arrays = build_provenance_arrays(built.target, built.fusion, built.donors, lut)
    return lut, arrays


def test_lut_row0_is_identity_target(rng: np.random.Generator, policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    lut, _ = _lut_and_arrays(built, policy)
    assert lut[0].index == 0
    assert lut[0].role is SourceRole.TARGET
    assert lut[0].frame_id == built.target.frame_id
    assert lut[0].alignment_method is AlignmentMethod.IDENTITY
    assert lut[0].matrix is None
    assert lut[0].interpolation is Interpolation.IDENTITY
    assert lut[0].decision_ids == ()


def test_homography_lut_stores_nine_values(rng: np.random.Generator, policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    lut, _ = _lut_and_arrays(built, policy)
    donor = lut[1]
    assert donor.alignment_method is AlignmentMethod.AKAZE_HOMOGRAPHY
    assert donor.matrix is not None and len(donor.matrix) == 9
    assert donor.interpolation is Interpolation.LANCZOS4
    assert donor.decision_ids == tuple(built.ids[donor.frame_number])
    assert donor.transform_id == f"h_{donor.frame_number}"


def test_ecc_affine_lut_stores_six_values(rng: np.random.Generator, policy: PolicyConfig) -> None:
    built = single_donor(rng, policy, Q=0.8, A=0.9, method=AlignmentMethod.ECC_AFFINE)
    lut = build_source_lut(built.target, built.donors, built.ids)
    assert lut[1].alignment_method is AlignmentMethod.ECC_AFFINE
    assert lut[1].matrix is not None and len(lut[1].matrix) == 6
    assert lut[1].matrix == (1.0, 0.0, 3.0, 0.0, 1.0, 1.0)
    assert lut[1].transform_id == f"a_{built.donors[0].obs.frame_number}"


def test_lut_requires_rank_order(rng: np.random.Generator, policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    reversed_donors = tuple(reversed(built.donors))
    with pytest.raises(ProvenanceIncomplete, match="LUT order") as exc:
        build_source_lut(built.target, reversed_donors, built.ids)
    assert exc.value.reason_code is ReasonCode.PROVENANCE_INCOMPLETE


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"video_id": OTHER_VIDEO}, "video"),
        ({"track_id": OTHER_TRACK}, "track"),
        ({}, "cannot be its own donor"),
        ({"detector_backed": False, "bridged": True}, "not a detector-backed"),
    ],
)
def test_foreign_or_bridged_donor_is_lineage_error(
    rng: np.random.Generator, policy: PolicyConfig, kwargs: dict, fragment: str
) -> None:
    sc = scenario(rng, policy)
    frame_n = 30 if "own" in fragment else 26
    bad = make_donor(sc.target, view(sc.world, 3, 1), frame_n, policy, tx=3.0, ty=1.0, R=0.9,
                     **kwargs)
    with pytest.raises(DonorLineageError, match=fragment) as exc:
        build_source_lut(sc.target, (bad,), {bad.obs.frame_number: ("placeholder",)})
    assert exc.value.reason_code is ReasonCode.DONOR_TRACK_MISMATCH


def test_identity_init_then_borrowed_tiles_only(rng: np.random.Generator,
                                                policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    lut, arrays = _lut_and_arrays(built, policy)
    assert arrays.lut == lut
    assert arrays.cls.shape == (H, W)
    assert not (arrays.cls == ProvenanceClass.GENERATED_BLEND).any()
    borrowed = arrays.cls == ProvenanceClass.BORROWED
    original = arrays.cls == ProvenanceClass.ORIGINAL
    assert borrowed.any() and original.any()
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    assert np.array_equal(arrays.source_x[original], xs[original])
    assert np.array_equal(arrays.source_y[original], ys[original])
    assert np.all(arrays.source_index[original] == 0)
    x1, y1, x2, y2 = SUBJECT
    outside = np.ones((H, W), dtype=bool)
    outside[y1:y2, x1:x2] = False
    assert not (borrowed & outside).any()
    for t in built.fusion.tile_decisions:
        if t.reason_code is not ReasonCode.BORROW_TILE_ACCEPTED:
            continue
        tx1, ty1, tx2, ty2 = t.box
        assert np.all(arrays.cls[ty1:ty2, tx1:tx2] == ProvenanceClass.BORROWED)
        assert np.all(arrays.source_index[ty1:ty2, tx1:tx2] == t.source_index)


def test_writer_never_emits_generated_blend(rng: np.random.Generator,
                                            policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    _, arrays = _lut_and_arrays(built, policy)
    assert int((arrays.cls == ProvenanceClass.GENERATED_BLEND).sum()) == 0
    counts = validate_arrays(arrays, W, H, arrays.lut,
                             {e.index: (W, H) for e in arrays.lut})
    assert counts.GENERATED_BLEND == 0
    assert counts.ORIGINAL + counts.BORROWED == W * H


def test_overlapping_borrowed_tiles_are_refused(rng: np.random.Generator,
                                                policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    first = next(t for t in built.fusion.tile_decisions
                 if t.reason_code is ReasonCode.BORROW_TILE_ACCEPTED)
    overlap = TileDecision(99, first.box, first.source_index, 1.0,
                           ReasonCode.BORROW_TILE_ACCEPTED, 0.5, 2.0)
    fusion = FusionResult(built.fusion.result_frame,
                          built.fusion.tile_decisions + (overlap,),
                          built.fusion.donor_lut_order)
    lut = build_source_lut(built.target, built.donors, built.ids)
    with pytest.raises(ProvenanceIncomplete, match="overlaps") as exc:
        build_provenance_arrays(built.target, fusion, built.donors, lut)
    assert exc.value.reason_code is ReasonCode.PROVENANCE_INCOMPLETE


def test_two_problems_are_reported_together(rng: np.random.Generator,
                                            policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    lut = build_source_lut(built.target, built.donors, built.ids)
    kept = next(t for t in built.fusion.tile_decisions
                if t.reason_code is ReasonCode.KEEP_ORIGINAL_CLEAR)
    bad_keep = dataclasses.replace(kept, source_index=1)
    outside = TileDecision(98, (0, 0, 8, 8), 1, 1.0, ReasonCode.BORROW_TILE_ACCEPTED, 0.5, 2.0)
    tiles = tuple(bad_keep if t.tile_index == kept.tile_index else t
                  for t in built.fusion.tile_decisions) + (outside,)
    fusion = FusionResult(built.fusion.result_frame, tiles, built.fusion.donor_lut_order)
    with pytest.raises(ProvenanceIncomplete) as exc:
        build_provenance_arrays(built.target, fusion, built.donors, lut)
    text = " ".join(exc.value.problems)
    assert "keeps the original but" in text
    assert "outside the subject box" in text


def test_provenance_record_is_complete_or_absent(rng: np.random.Generator,
                                                 policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    lut, arrays = _lut_and_arrays(built, policy)
    npz = encode_provenance(arrays)
    digest = sha256_bytes(npz)
    uri = npz_uri()
    rec = provenance_record(arrays, run_id=RUN, subject_box=SUBJECT, npz_uri=uri,
                            npz_sha256=digest, created_at=START)
    assert rec.coverage_complete
    assert rec.coverage_counts.GENERATED_BLEND == 0
    assert rec.artifact_sha256 == digest
    assert rec.source_lut == lut
    with pytest.raises(ProvenanceIncomplete, match="npz_sha256") as exc:
        provenance_record(arrays, run_id=RUN, subject_box=SUBJECT, npz_uri=uri,
                          npz_sha256="0" * 64, created_at=START)
    assert exc.value.reason_code is ReasonCode.PROVENANCE_INCOMPLETE


def test_record_rejects_borrowed_pixels_outside_subject(rng: np.random.Generator,
                                                        policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    _, arrays = _lut_and_arrays(built, policy)
    arrays.cls[0, 0] = ProvenanceClass.BORROWED
    arrays.source_index[0, 0] = 1
    arrays.source_x[0, 0] = np.float32(1.0)
    arrays.source_y[0, 0] = np.float32(1.0)
    npz = encode_provenance(arrays)
    with pytest.raises(ProvenanceIncomplete, match="outside the subject box") as exc:
        provenance_record(arrays, run_id=RUN, subject_box=SUBJECT, npz_uri=npz_uri(),
                          npz_sha256=sha256_bytes(npz), created_at=START)
    assert exc.value.reason_code is ReasonCode.PROVENANCE_INCOMPLETE


def test_one_generated_pixel_is_generated_semantic(rng: np.random.Generator,
                                                   policy: PolicyConfig) -> None:
    built = build_scene(rng, policy)
    _, arrays = _lut_and_arrays(built, policy)
    arrays.cls[SUBJECT[1], SUBJECT[0]] = ProvenanceClass.GENERATED_BLEND
    with pytest.raises(GeneratedPixelError) as exc:
        provenance_record(arrays, run_id=RUN, subject_box=SUBJECT, npz_uri=npz_uri(),
                          npz_sha256=sha256_bytes(encode_provenance(arrays)), created_at=START)
    assert exc.value.reason_code is ReasonCode.GENERATED_SEMANTIC_PIXEL


def test_writer_matches_readonly_golden_invariants(golden_completed) -> None:
    g = golden_completed
    assert g.arrays is not None and g.provenance is not None
    assert not (g.arrays.cls == ProvenanceClass.GENERATED_BLEND).any()
    rec = provenance_record(
        g.arrays, run_id=g.run.run_id, subject_box=g.provenance.subject_bbox_px,
        npz_uri=g.provenance.class_map_uri, npz_sha256=g.provenance.artifact_sha256,
        created_at=g.provenance.created_at)
    assert rec.coverage_complete
    assert rec.coverage_counts == g.provenance.coverage_counts
    assert rec.source_lut == g.provenance.source_lut
