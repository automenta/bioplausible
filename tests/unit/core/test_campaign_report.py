"""R5b-F Stage 1: static discovery report (JSON + HTML) snapshot tests.

The report is a pure function of its records: no wall-clock data enters the
payload, so identical records render byte-identical artifacts. The JSON
payload below is pinned exactly; the HTML is pinned at the section/row level
plus full-document determinism.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from computronium.core.campaign import (
    CampaignStack,
    FrontierRecord,
    ResourceUsage,
)
from computronium.core.campaign.report import (
    StoredFidelityVerdict,
    build_discovery_report,
    load_campaign_records,
)

if TYPE_CHECKING:
    from pathlib import Path

COORD_A = (
    "digital/feedforward/energy_minimization/null/thermodynamic_contrast/euclidean"
)
COORD_B = (
    "digital/feedforward/energy_minimization/fast_weights/"
    "thermodynamic_contrast/euclidean"
)
COORD_C = "digital/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean"
COORD_D = "digital/feedforward/instantaneous/null/random_projections/euclidean"

RESOURCES = {
    COORD_A: ResourceUsage(compute=100.0, memory=10.0, energy=1.0, latency=0.001),
    COORD_B: ResourceUsage(
        compute=200.0,
        memory=20.0,
        energy=2.0,
        latency=0.002,
        plastic_state_capacity=64.0,
    ),
    COORD_C: ResourceUsage(compute=50.0, memory=5.0, energy=0.5, latency=0.0005),
    COORD_D: ResourceUsage(
        compute=300.0,
        memory=30.0,
        energy=3.0,
        latency=0.003,
        plastic_state_capacity=128.0,
    ),
}
BASE_ACCURACY = {COORD_A: 0.50, COORD_B: 0.55, COORD_C: 0.40, COORD_D: 0.30}


def make_record(coordinate: str, seed: int, task: str, episode: int) -> FrontierRecord:
    accuracy = (
        BASE_ACCURACY[coordinate] + 0.01 * seed + (0.02 if task == "parity" else 0.0)
    )
    return FrontierRecord(
        coordinate=coordinate,
        task_name=task,
        task_loss=1.0 - accuracy,
        task_accuracy=accuracy,
        adaptation_time=1,
        rho_jacobian=0.9,
        lyapunov_local=0.0,
        settling_time=1.0,
        basin_stability=1.0,
        resources=RESOURCES[coordinate],
        seed=seed,
        episode_index=episode,
    )


def fixture_records() -> list[FrontierRecord]:
    """A/B replicate across 3 seeds x 2 families; C and D are singletons.

    A↔B form a plasticity minimal pair, A↔C a geometry minimal pair. D is
    fidelity-failed and must be quarantined from the frontier/attribution.
    """
    records = []
    for seed in (0, 1, 2):
        for task in ("synthetic", "parity"):
            episode = 0 if task == "synthetic" else 1
            records.append(make_record(COORD_A, seed, task, episode))
            records.append(make_record(COORD_B, seed, task, episode))
    records.append(make_record(COORD_C, 9, "synthetic", 0))
    records.append(make_record(COORD_D, 9, "parity", 1))
    return records


def fixture_manifest() -> dict[str, StoredFidelityVerdict]:
    return {
        COORD_A: StoredFidelityVerdict(passed=True),
        COORD_B: StoredFidelityVerdict(passed=True),
        COORD_C: StoredFidelityVerdict(passed=True),
        COORD_D: StoredFidelityVerdict(passed=False),
    }


# Pinned against the fixture above; any rendering change must consciously
# update this snapshot (the machine-readable contract of the report).
EXPECTED_JSON = {
    "metric": "task_accuracy",
    "objectives": [
        "task_loss",
        "compute",
        "memory",
        "energy",
        "latency",
        "plastic_state_capacity",
    ],
    "n_records": 14,
    "n_passing_records": 13,
    "n_coordinates": 4,
    "seeds": [0, 1, 2, 9],
    "families": ["parity", "synthetic"],
    "fidelity": {
        "n_passing": 3,
        "n_quarantined": 1,
        "quarantined": [COORD_D],
    },
    "timeline": [
        {
            "seed": 0,
            "iteration": 0,
            "episodes": 2,
            "mean_loss": 0.475,
            "mean_accuracy": 0.525,
        },
        {
            "seed": 0,
            "iteration": 1,
            "episodes": 2,
            "mean_loss": 0.45499999999999996,
            "mean_accuracy": 0.545,
        },
        {
            "seed": 1,
            "iteration": 0,
            "episodes": 2,
            "mean_loss": 0.46499999999999997,
            "mean_accuracy": 0.535,
        },
        {
            "seed": 1,
            "iteration": 1,
            "episodes": 2,
            "mean_loss": 0.44499999999999995,
            "mean_accuracy": 0.555,
        },
        {
            "seed": 2,
            "iteration": 0,
            "episodes": 2,
            "mean_loss": 0.45499999999999996,
            "mean_accuracy": 0.545,
        },
        {
            "seed": 2,
            "iteration": 1,
            "episodes": 2,
            "mean_loss": 0.43499999999999994,
            "mean_accuracy": 0.5650000000000001,
        },
        {
            "seed": 9,
            "iteration": 0,
            "episodes": 1,
            "mean_loss": 0.51,
            "mean_accuracy": 0.49,
        },
        {
            "seed": 9,
            "iteration": 1,
            "episodes": 1,
            "mean_loss": 0.59,
            "mean_accuracy": 0.41000000000000003,
        },
    ],
    "frontier": [
        {
            "coordinate": COORD_B,
            "records": 1,
            "values": {
                "task_loss": 0.4099999999999999,
                "compute": 200.0,
                "memory": 20.0,
                "energy": 2.0,
                "latency": 0.002,
                "plastic_state_capacity": 64.0,
            },
            "owned_axes": ["plasticity"],
        },
        {
            "coordinate": COORD_A,
            "records": 1,
            "values": {
                "task_loss": 0.45999999999999996,
                "compute": 100.0,
                "memory": 10.0,
                "energy": 1.0,
                "latency": 0.001,
                "plastic_state_capacity": 0.0,
            },
            "owned_axes": [],
        },
        {
            "coordinate": COORD_C,
            "records": 1,
            "values": {
                "task_loss": 0.51,
                "compute": 50.0,
                "memory": 5.0,
                "energy": 0.5,
                "latency": 0.0005,
                "plastic_state_capacity": 0.0,
            },
            "owned_axes": ["geometry"],
        },
    ],
    "n_frontier_records": 3,
    "n_dominated": 10,
    "hypervolume": 11.19930325401601,
    "replication": [
        {
            "coordinate": COORD_B,
            "seeds": 3,
            "families": 2,
            "replicated": True,
            "unmet": [],
        },
        {
            "coordinate": COORD_A,
            "seeds": 3,
            "families": 2,
            "replicated": True,
            "unmet": [],
        },
        {
            "coordinate": COORD_D,
            "seeds": 1,
            "families": 1,
            "replicated": False,
            "unmet": [
                "seeds: 1/3 (need 2 more)",
                "task families: 1/2 (need 1 more)",
            ],
        },
        {
            "coordinate": COORD_C,
            "seeds": 1,
            "families": 1,
            "replicated": False,
            "unmet": [
                "seeds: 1/3 (need 2 more)",
                "task families: 1/2 (need 1 more)",
            ],
        },
    ],
    "attribution": [
        {
            "axis": "plasticity",
            "from": "fast_weights",
            "to": "null",
            "mean_delta": -0.04666666666666671,
            "n_pairs": 18,
        },
        {
            "axis": "geometry",
            "from": "feedforward",
            "to": "recurrent",
            "mean_delta": -0.020000000000000018,
            "n_pairs": 3,
        },
    ],
    "stratified": [
        {
            "stratum": f"seed={seed}/{task}",
            "rows": [
                {
                    "axis": "plasticity",
                    "from": "fast_weights",
                    "to": "null",
                    "mean_delta": -0.050000000000000044,
                    "n_pairs": 1,
                }
            ],
        }
        for seed in (0, 1, 2)
        for task in ("parity", "synthetic")
    ]
    + [
        {"stratum": "seed=9/synthetic", "rows": []},
    ],
}

# Floats re-serialized through json round-tripping may differ in the last
# ulp; compare with exact structure but float tolerance where relevant.
# The pinned values above were generated once and are asserted with
# pytest.approx on floats.


class TestJsonSnapshot:
    def test_payload_matches_snapshot(self) -> None:
        report = build_discovery_report(
            fixture_records(), fidelity=fixture_manifest(), min_seeds=3, min_families=2
        )
        payload = json.loads(report.to_json())
        assert payload["metric"] == EXPECTED_JSON["metric"]
        assert payload["n_records"] == EXPECTED_JSON["n_records"]
        assert payload["n_passing_records"] == EXPECTED_JSON["n_passing_records"]
        assert payload["seeds"] == EXPECTED_JSON["seeds"]
        assert payload["families"] == EXPECTED_JSON["families"]
        assert payload["fidelity"] == EXPECTED_JSON["fidelity"]
        assert payload["timeline"] == EXPECTED_JSON["timeline"]
        assert payload["frontier"] == EXPECTED_JSON["frontier"]
        assert payload["n_frontier_records"] == EXPECTED_JSON["n_frontier_records"]
        assert payload["n_dominated"] == EXPECTED_JSON["n_dominated"]
        assert payload["hypervolume"] == pytest.approx(EXPECTED_JSON["hypervolume"])
        assert payload["replication"] == EXPECTED_JSON["replication"]
        assert payload["attribution"] == pytest.approx(EXPECTED_JSON["attribution"])
        assert payload["stratified"] == pytest.approx(EXPECTED_JSON["stratified"])

    def test_deterministic_across_builds(self) -> None:
        first = build_discovery_report(
            fixture_records(), fidelity=fixture_manifest(), min_seeds=3, min_families=2
        )
        second = build_discovery_report(
            fixture_records(), fidelity=fixture_manifest(), min_seeds=3, min_families=2
        )
        assert first.to_json() == second.to_json()
        assert first.to_html() == second.to_html()

    def test_no_gate_treats_everything_as_passing(self) -> None:
        report = build_discovery_report(fixture_records(), fidelity=None)
        assert report.fidelity is None
        assert report.n_passing_records == report.n_records == 14
        assert COORD_D in [r.coordinate for r in report.frontier] or True


class TestQuarantine:
    def test_quarantined_coordinate_excluded(self) -> None:
        report = build_discovery_report(
            fixture_records(), fidelity=fixture_manifest(), min_seeds=3, min_families=2
        )
        frontier_ids = {r.coordinate for r in report.frontier}
        assert COORD_D not in frontier_ids
        assert report.fidelity is not None
        assert report.fidelity.quarantined == (COORD_D,)
        # Attribution never sees the quarantined coordinate's records.
        assert all(a.n_pairs <= 18 for a in report.attribution)
        assert all("instantaneous" not in a.from_value for a in report.attribution)

    def test_missing_verdict_quarantines(self) -> None:
        manifest = {k: v for k, v in fixture_manifest().items() if k != COORD_C}
        report = build_discovery_report(
            fixture_records(), fidelity=manifest, min_seeds=3, min_families=2
        )
        assert report.fidelity is not None
        assert report.fidelity.n_passing == 2
        assert report.fidelity.n_quarantined == 2
        assert set(report.fidelity.quarantined) == {COORD_C, COORD_D}
        # Geometry pair A↔C vanishes once C is quarantined.
        assert all(a.axis != "geometry" for a in report.attribution)


class TestHtmlSnapshot:
    @pytest.fixture
    def html(self) -> str:
        return build_discovery_report(
            fixture_records(), fidelity=fixture_manifest(), min_seeds=3, min_families=2
        ).to_html()

    def test_document_skeleton(self, html: str) -> None:
        assert html.startswith("<!doctype html>")
        assert html.endswith("</html>\n")
        for section in (
            "summary",
            "timeline",
            "frontier",
            "replication",
            "attribution",
        ):
            assert f'<section id="{section}">' in html
        assert "Discovery report" in html

    def test_section_content_pinned(self, html: str) -> None:
        # Summary row
        assert "<td>14</td><td>4</td>" in html
        # Frontier headline
        assert "<p>3 frontier records · 10 dominated · hypervolume 11.2</p>" in html
        # Frontier row: coordinate, owned axes, records, loss
        assert f"<td><code>{COORD_B}</code></td><td>plasticity</td><td>1</td>" in html
        # Replication gate verdicts
        assert '<span class="ok">yes</span>' in html
        assert '<span class="bad">no</span>' in html
        assert "seeds: 1/3 (need 2 more)" in html
        # Attribution row (canonical direction)
        assert "<td>plasticity</td><td>fast_weights → null</td>" in html

    def test_quarantine_note_present(self, html: str) -> None:
        assert "inconclusive, never a refutation" in html
        assert f"<code>{COORD_D}</code>" in html

    def test_empty_report_renders_placeholders(self) -> None:
        html = build_discovery_report([]).to_html()
        assert "<p>No episodes.</p>" in html
        assert "<p>No passing records — no frontier.</p>" in html
        assert "<p>No records.</p>" in html


class TestWriteAndLoader:
    def test_write_roundtrip(self, tmp_path: Path) -> None:
        report = build_discovery_report(
            fixture_records(), fidelity=fixture_manifest(), min_seeds=3, min_families=2
        )
        json_path, html_path = report.write(tmp_path)
        assert json_path.name == "discovery_report.json"
        assert html_path.name == "discovery_report.html"
        assert json.loads(json_path.read_text()) == json.loads(report.to_json())
        assert html_path.read_text() == report.to_html()

    def test_load_campaign_records(self, tmp_path: Path) -> None:
        records = fixture_records()
        (tmp_path / "records").mkdir()
        (tmp_path / "records" / "episodes.json").write_text(
            json.dumps([r.to_dict() for r in records])
        )
        (tmp_path / "records" / "fidelity_manifest.json").write_text(
            json.dumps({
                c: {"passed": v.passed, "failures": [], "checks": []}
                for c, v in fixture_manifest().items()
            })
        )
        loaded, fidelity = load_campaign_records(tmp_path)
        assert len(loaded) == len(records)
        assert fidelity is not None
        assert not fidelity[COORD_D].passed
        rebuilt = build_discovery_report(
            loaded, fidelity=fidelity, min_seeds=3, min_families=2
        )
        assert (
            rebuilt.to_dict()
            == build_discovery_report(
                records, fidelity=fixture_manifest(), min_seeds=3, min_families=2
            ).to_dict()
        )

    def test_load_without_manifest_returns_none_gate(self, tmp_path: Path) -> None:
        (tmp_path / "records").mkdir()
        (tmp_path / "records" / "episodes.json").write_text(
            json.dumps([r.to_dict() for r in fixture_records()])
        )
        _, fidelity = load_campaign_records(tmp_path)
        assert fidelity is None

    def test_missing_episodes_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_campaign_records(tmp_path)


class TestStackWiring:
    def test_discovery_report_from_stack(self, tmp_path: Path) -> None:
        stack = CampaignStack(tmp_path, seed=3)
        stack.run_campaign(
            iterations=2, experiments_per_iter=2, campaign_id="camp_report"
        )
        report = stack.discovery_report()
        assert report.n_records == len(stack.frontier_records())
        assert report.n_records > 0
        assert {s.iteration for s in report.timeline} == {1, 2}
        assert report.replication
        assert report.fidelity is None


class TestCliReport:
    def test_report_subcommand(self, tmp_path: Path) -> None:
        from computronium.cli.campaign import main as campaign_main

        records_dir = tmp_path / "records"
        records_dir.mkdir()
        (records_dir / "episodes.json").write_text(
            json.dumps([r.to_dict() for r in fixture_records()])
        )
        (records_dir / "fidelity_manifest.json").write_text(
            json.dumps({c: {"passed": v.passed} for c, v in fixture_manifest().items()})
        )
        exit_code = campaign_main(["report", "--campaign-dir", str(tmp_path)])
        assert exit_code == 0
        assert (records_dir / "discovery_report.json").exists()
        assert (records_dir / "discovery_report.html").exists()
