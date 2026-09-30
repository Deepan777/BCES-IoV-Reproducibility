from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bces.data.adequacy import AdequacyStatus, evaluate_adequacy
from bces.data.adapters import validate_dataset_file
from bces.data.datasets import DATASET_PROFILES, EvidenceRole, dataset_profile
from bces.data.manifest import Catalog, CatalogEntry, DataRole
from bces.data.multidataset import aggregate_adequacy, load_multidataset_plan
from bces.data.splits import (
    AccessPurpose,
    Partition,
    PartitionAccessError,
    assert_scenario_integrity,
    assign_intersections,
    assign_intersections_by_capacity,
    require_partition_access,
)
from bces.data.tfd import validate_tfd_csv, validate_tfd_map
from bces.data.workflow import (
    build_data_state,
    load_verified_manifest,
    write_data_state,
)

pytestmark = pytest.mark.phase3


def make_entry(**overrides: object) -> CatalogEntry:
    values = {
        "entry_id": "scene-1-cooperative",
        "url": "https://raw.githubusercontent.com/AIR-THU/DAIR-V2X-Seq/main/scene.csv",
        "destination": "cooperative/scene-1.csv",
        "role": DataRole.COOPERATIVE,
        "scene_id": "scene-1",
        "intersection_id": "1",
        "selection_reason": "unit-test representative scene",
        "release": "official-example-2024",
        "license_url": "https://github.com/AIR-THU/DAIR-V2X-Seq",
        "expected_bytes": 10,
        "expected_valid_windows": 1,
        "behavior_families": ("keep",),
    }
    values.update(overrides)
    return CatalogEntry(**values)  # type: ignore[arg-type]


def test_catalog_rejects_forbidden_or_unselective_entries() -> None:
    with pytest.raises(ValueError, match="official"):
        make_entry(url="https://example.com/scene.csv")
    with pytest.raises(ValueError, match="forbidden"):
        make_entry(url="https://github.com/AIR-THU/camera-scene.csv")
    with pytest.raises(ValueError, match="CSV/JSON"):
        make_entry(destination="V2X-Seq-TFD-Example.zip")
    with pytest.raises(ValueError, match="contained"):
        make_entry(destination="../outside.csv")


def test_catalog_requires_exact_allowlisted_url() -> None:
    entry = make_entry()
    catalog = Catalog("v2xseq_tfd_compact", "ready", (), (entry,))
    assert catalog.require_url(entry.url) is entry
    with pytest.raises(PermissionError, match="allowlist"):
        catalog.require_url("https://github.com/AIR-THU/other.csv")


def test_catalog_requires_matching_tfd_bundle_before_transfer() -> None:
    incomplete = Catalog("v2xseq_tfd_compact", "ready", (), (make_entry(),))
    with pytest.raises(ValueError, match="matching roles"):
        incomplete.validate_complete_bundles()
    roles = (
        DataRole.COOPERATIVE,
        DataRole.VEHICLE,
        DataRole.INFRASTRUCTURE,
        DataRole.TRAFFIC_LIGHT,
    )
    entries = tuple(
        make_entry(
            entry_id=role.value,
            role=role,
            destination=f"{role.value}/scene-1.csv",
        )
        for role in roles
    ) + (
        make_entry(
            entry_id="map",
            role=DataRole.MAP,
            destination="map/intersection-1.json",
        ),
    )
    Catalog("v2xseq_tfd_compact", "ready", (), entries).validate_complete_bundles()


@given(st.sets(st.integers(min_value=0, max_value=100).map(str), min_size=1))
def test_intersection_split_is_deterministic_and_disjoint(ids: set[str]) -> None:
    first = assign_intersections(ids, salt="registered-salt")
    second = assign_intersections(ids, salt="registered-salt")
    assert first == second
    assert set(first.by_intersection) == ids
    assert len(first.by_intersection) == len(ids)


def test_split_enforces_two_test_intersections_at_confirmatory_scale() -> None:
    assignment = assign_intersections({str(i) for i in range(8)}, salt="fixed")
    assert sum(value is Partition.TEST for value in assignment.by_intersection.values()) >= 2
    assert len(set(assignment.by_intersection.values())) == 4


def test_capacity_aware_split_uses_metadata_only_and_balances_partitions() -> None:
    capacities = {
        "yizhuang#11-1_po": 1400,
        "yizhuang#12-1_po": 921,
        "yizhuang#13-1_po": 639,
        "yizhuang#14-1_po": 287,
        "yizhuang#20-1_po": 1349,
        "yizhuang#25-1_po": 12,
        "yizhuang#4-1_po": 2703,
        "yizhuang#7-1_po": 421,
    }
    assignment = assign_intersections_by_capacity(capacities, salt="fixed-capacity")
    assert assignment.method == "capacity_balanced_whole_intersection_v1"
    assert assignment.capacity_basis == capacities
    assert assignment.capacity_objective is not None
    assert assignment.capacity_objective["selection_inputs"].startswith(
        "official scenario counts only"
    )
    assert sum(value is Partition.TEST for value in assignment.by_intersection.values()) == 2
    assert sum(
        value is Partition.TRAIN for value in assignment.by_intersection.values()
    ) == 4
    assert assignment == assign_intersections_by_capacity(
        capacities, salt="fixed-capacity"
    )


def test_scenario_and_final_test_leakage_guards() -> None:
    assignment = assign_intersections({"a", "b", "c", "d"}, salt="fixed")
    assert_scenario_integrity([("s1", "a"), ("s1", "a")], assignment)
    with pytest.raises(ValueError, match="crosses"):
        assert_scenario_integrity([("s1", "a"), ("s1", "b")], assignment)
    with pytest.raises(PartitionAccessError, match="cannot be opened"):
        require_partition_access(Partition.TEST, AccessPurpose.MODEL_SELECTION)
    require_partition_access(Partition.TEST, AccessPurpose.FINAL_EVALUATION)


def test_exact_confirmatory_adequacy_thresholds() -> None:
    records = []
    for index in range(30):
        records.append(
            {
                "scenario_id": f"s{index}",
                "intersection_id": str(index % 8),
                "valid_windows": 17 if index < 20 else 16,
                "behavior_families": ["keep", "brake", "left"],
            }
        )
    assignments = {
        key: value.value
        for key, value in assign_intersections(
            {str(i) for i in range(8)}, salt="fixed"
        ).by_intersection.items()
    }
    decisions = {
        partition.value: {behavior: 50 for behavior in ("keep", "brake", "left")}
        for partition in Partition
    }
    report = evaluate_adequacy(
        records,
        split_assignments=assignments,
        validity_decisions_by_split_behavior=decisions,
    )
    assert report.status is AdequacyStatus.CONFIRMATORY
    assert report.passes
    assert report.counts["valid_windows"] == 500
    decisions[Partition.TEST.value]["left"] = 49
    failed = evaluate_adequacy(
        records,
        split_assignments=assignments,
        validity_decisions_by_split_behavior=decisions,
    )
    assert not failed.passes
    assert "validity_decisions_per_split_behavior" in failed.unmet


def test_blocker_takes_precedence_over_pilot() -> None:
    report = evaluate_adequacy(
        [], split_assignments={}, blockers=("oversized_archive",)
    )
    assert report.status is AdequacyStatus.BLOCKED
    assert not report.passes


def _trajectory_csv() -> str:
    columns = [
        "city",
        "timestamp",
        "id",
        "type",
        "sub_type",
        "tag",
        "x",
        "y",
        "z",
        "length",
        "width",
        "height",
        "theta",
        "v_x",
        "v_y",
        "intersect_id",
        "vic_tag",
        "from_side",
        "car_side_id",
        "road_side_id",
    ]
    rows = [
        ["B", "0.0", "v1", "Vehicle", "Car", "AV", "0", "0", "0", "4", "2", "1", "0", "1", "0", "1", "1", "i", "c", "r"],
        ["B", "0.1", "v1", "Vehicle", "Car", "AV", "0.1", "0", "0", "4", "2", "1", "0", "1", "0", "1", "1", "i", "c", "r"],
        ["B", "0.1", "v1", "Vehicle", "Car", "AV", "0.1", "0", "0", "4", "2", "1", "0", "1", "0", "1", "1", "i", "c", "r"],
        ["B", "0.4", "v1", "Vehicle", "Car", "AV", "0.4", "0", "0", "4", "2", "1", "0", "1", "0", "1", "1", "i", "c", "r"],
    ]
    return ",".join(columns) + "\n" + "\n".join(",".join(row) for row in rows) + "\n"


def test_streaming_tfd_validation_records_drops_and_gaps(tmp_path: Path) -> None:
    path = tmp_path / "scene.csv"
    path.write_text(_trajectory_csv(), encoding="utf-8")
    result = validate_tfd_csv(path, make_entry(intersection_id="1"))
    assert result.row_count == 4
    assert result.valid_row_count == 3
    assert result.dropped_rows[0]["line"] == 4
    assert len(result.continuity_gaps) == 1


def test_map_schema_validation(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "LANE": {"lane-1": {"centerline": [[0, 0], [1, 0]]}},
                "STOPLINE": {},
                "CROSSWALK": {},
            }
        ),
        encoding="utf-8",
    )
    result = validate_tfd_map(path)
    assert result.valid_row_count == 1


def test_manifest_workflow_rejects_unallowlisted_raw_file(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "surprise.csv").write_text("not allowlisted", encoding="utf-8")
    catalog = Catalog("v2xseq_tfd_compact", "ready", (), ())
    with pytest.raises(PermissionError, match="not allowlisted"):
        build_data_state(catalog, raw_root=raw_root)


def test_manifest_uses_registered_schema_and_verified_hash(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    path = raw_root / "cooperative" / "scene-1.csv"
    path.parent.mkdir(parents=True)
    path.write_text(_trajectory_csv(), encoding="utf-8")
    entry = make_entry(expected_bytes=None, expected_sha256=None)
    catalog = Catalog("v2xseq_tfd_compact", "ready", (), (entry,))
    state = build_data_state(catalog, raw_root=raw_root)
    manifest_path = tmp_path / "manifest.json"
    write_data_state(
        state,
        catalog=catalog,
        manifest_path=manifest_path,
        split_path=tmp_path / "split.json",
        adequacy_path=tmp_path / "adequacy.json",
    )
    records = load_verified_manifest(manifest_path)
    assert len(records) == 1
    assert records[0].scenario_id == "scene-1"
    assert records[0].source_role is DataRole.COOPERATIVE
    assert records[0].split == "test"


def test_manifest_hash_detects_tampering(tmp_path: Path) -> None:
    catalog = Catalog("v2xseq_tfd_compact", "blocked", (), ())
    state = build_data_state(catalog, raw_root=tmp_path / "raw")
    manifest_path = tmp_path / "manifest.json"
    write_data_state(
        state,
        catalog=catalog,
        manifest_path=manifest_path,
        split_path=tmp_path / "split.json",
        adequacy_path=tmp_path / "adequacy.json",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = "confirmatory"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_verified_manifest(manifest_path)


def test_dataset_roles_and_primary_gate_are_frozen() -> None:
    assert set(DATASET_PROFILES) == {
        "v2xtraj_primary",
        "tumtraf_v2x_external",
        "interaction_sensitivity",
        "v2xseq_tfd_compact",
    }
    assert dataset_profile("v2xtraj_primary").may_unlock_phase4
    assert all(
        not profile.may_unlock_phase4
        for key, profile in DATASET_PROFILES.items()
        if key != "v2xtraj_primary"
    )
    with pytest.raises(ValueError, match="evidence_role"):
        Catalog("v2xtraj_primary", "blocked", (), (), "sensitivity")


def test_v2xtraj_requires_native_complete_bundle() -> None:
    roles = (
        DataRole.EGO,
        DataRole.VEHICLE,
        DataRole.INFRASTRUCTURE,
        DataRole.TRAFFIC_LIGHT,
    )
    entries = tuple(
        make_entry(
            entry_id=f"v2xtraj-{role.value}",
            url=f"https://raw.githubusercontent.com/AIR-THU/V2X-Graph/main/{role.value}.csv",
            destination=f"{role.value}/scene-1.csv",
            role=role,
        )
        for role in roles
    ) + (
        make_entry(
            entry_id="v2xtraj-map",
            url="https://raw.githubusercontent.com/AIR-THU/V2X-Graph/main/map.json",
            destination="map/intersection-1.json",
            role=DataRole.MAP,
        ),
    )
    Catalog("v2xtraj_primary", "ready", (), entries, "primary").validate_complete_bundles()
    wrong = (make_entry(role=DataRole.COOPERATIVE),)
    with pytest.raises(ValueError, match="matching roles"):
        Catalog("v2xtraj_primary", "ready", (), wrong, "primary").validate_complete_bundles()


def test_plan_has_one_primary_and_explicit_network_provenance() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_multidataset_plan(root / "configs" / "data" / "multi_dataset_plan.yaml")
    assert plan.primary_dataset_id == "v2xtraj_primary"
    assert plan.network_impairments == "replayed_or_simulated_not_observed_packet_logs"
    assert len(plan.sha256) == 64


def test_auxiliary_success_cannot_unlock_phase4() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = load_multidataset_plan(root / "configs" / "data" / "multi_dataset_plan.yaml")
    blocked = evaluate_adequacy([], split_assignments={}, blockers=("primary_blocked",))
    passing = evaluate_adequacy(
        [
            {
                "scenario_id": f"s{i}",
                "intersection_id": str(i % 8),
                "valid_windows": 17 if i < 20 else 16,
                "behavior_families": ["keep", "brake", "left"],
            }
            for i in range(30)
        ],
        split_assignments={
            key: value.value
            for key, value in assign_intersections({str(i) for i in range(8)}, salt="fixed").by_intersection.items()
        },
        validity_decisions_by_split_behavior={
            partition.value: {behavior: 50 for behavior in ("keep", "brake", "left")}
            for partition in Partition
        },
    )
    reports = {item.dataset_id: passing for item in plan.datasets}
    reports["v2xtraj_primary"] = blocked
    aggregate = aggregate_adequacy(plan, reports)
    assert aggregate["phase4_permitted"] is False
    assert aggregate["datasets"]["tumtraf_v2x_external"]["passes"] is True
    assert aggregate["gate_rule"] == "primary_dataset_only"


def test_manifest_records_dataset_and_evidence_role(tmp_path: Path) -> None:
    catalog = Catalog("v2xtraj_primary", "blocked", (), (), EvidenceRole.PRIMARY.value)
    state = build_data_state(catalog, raw_root=tmp_path / "raw", blockers=("preflight",))
    path = tmp_path / "manifest.json"
    write_data_state(
        state,
        catalog=catalog,
        manifest_path=path,
        split_path=tmp_path / "split.json",
        adequacy_path=tmp_path / "adequacy.json",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "v2xtraj_primary"
    assert payload["evidence_role"] == "primary"


def test_unreviewed_external_adapter_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "scene.csv"
    path.write_text(_trajectory_csv(), encoding="utf-8")
    with pytest.raises(ValueError, match="not activated"):
        validate_dataset_file(path, make_entry(), dataset_id="tumtraf_v2x_external")


def test_v2xtraj_native_adapter_accepts_verified_schema(tmp_path: Path) -> None:
    path = tmp_path / "scene.csv"
    path.write_text(_trajectory_csv(), encoding="utf-8")
    result = validate_dataset_file(
        path, make_entry(role=DataRole.EGO), dataset_id="v2xtraj_primary"
    )
    assert result.valid_row_count == 3


def test_map_records_do_not_inflate_adequacy_counts() -> None:
    report = evaluate_adequacy(
        [
            {"scenario_id": "s1", "intersection_id": "i1", "valid_windows": 1, "source_role": "ego"},
            {"scenario_id": "map-i2", "intersection_id": "i2", "valid_windows": 0, "source_role": "map"},
        ],
        split_assignments={"i1": "train", "i2": "test"},
    )
    assert report.counts["scenarios"] == 1
    assert report.counts["intersections"] == 1
    assert report.counts["valid_windows"] == 1
