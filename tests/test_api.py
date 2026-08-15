"""Tests for the JSON API in ``server/``.

Runs against a synthetic dataset written in the real on-disk schema, so it needs
no fetched episodes and no network — matching the skip-when-empty pattern in
``tests/test_pipeline.py``.

The assertion that earns its keep here is **strict JSON parseability**. FastAPI's
default encoder will happily emit a bare ``NaN`` token, which is invalid JSON:
``curl`` shows a response that looks fine while ``JSON.parse`` in the browser
throws. This pipeline produces NaN routinely — step 1 of a curriculum has no
incoming edge weight, review rows deliberately blank their ramp cost, and
``silhouette`` is undefined below three clusters — so every response body is
re-parsed with ``json.loads(..., parse_constant=...)`` to prove no non-finite
literal escaped :func:`server.serialize.jsonable`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("fastapi", reason="fastapi not installed; pip install -r requirements.txt")
# starlette's TestClient raises RuntimeError (not ImportError) when its HTTP client
# is missing, which would abort collection rather than skip. Check it explicitly.
pytest.importorskip("httpx2", reason="starlette's TestClient requires httpx2")

from fastapi.testclient import TestClient  # noqa: E402

from scripts.generate_synthetic_data import generate_dataset  # noqa: E402
from server import api as api_module  # noqa: E402


def _reject_constant(name: str) -> None:
    """``json.loads`` hook: raise on NaN/Infinity instead of accepting them.

    Python's ``json`` accepts these non-standard literals by default, which would
    let exactly the bug this test targets slip through.
    """
    raise AssertionError(f"response contains the non-JSON literal {name!r}")


def strict_json(response) -> Any:
    """Parse a response body, failing if it holds NaN or Infinity."""
    assert response.status_code == 200, f"{response.request.url} -> {response.status_code}"
    return json.loads(response.text, parse_constant=_reject_constant)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory) -> Path:
    """A synthetic dataset, including the pathological episodes the loader rejects."""
    out = tmp_path_factory.mktemp("api_synth")
    generate_dataset(out, n_episodes=18, n_duplicates=3, seed=11, inject_edge_cases=True)
    return out


@pytest.fixture(scope="module")
def client(synth_dir) -> TestClient:
    # The module-level caches are keyed on the request, so a per-module client is
    # safe and keeps the expensive DTW matrix computed once for the whole file.
    api_module._PIPELINE_CACHE.clear()
    api_module._CONTEXT_CACHE.clear()
    with TestClient(api_module.app) as test_client:
        test_client.data_dir = str(synth_dir)  # type: ignore[attr-defined]
        yield test_client


def config_for(client: TestClient) -> Dict[str, Any]:
    return {"data_dir": client.data_dir}  # type: ignore[attr-defined]


class TestHealth:
    def test_health_is_ok(self, client):
        assert strict_json(client.get("/api/health"))["ok"] is True


class TestSnapshot:
    @pytest.fixture(scope="class")
    @classmethod
    def snapshot(cls, client):
        return strict_json(client.post("/api/run", json=config_for(client)))

    def test_reports_usable_and_rejected_counts(self, snapshot):
        assert snapshot["n_episodes"] >= 2
        # inject_edge_cases writes episodes the loader must refuse; if none were
        # rejected the loader silently accepted degenerate pose streams.
        assert snapshot["n_skipped"] > 0
        assert len(snapshot["skipped"]) == snapshot["n_skipped"]
        assert all(row["reason"] for row in snapshot["skipped"])

    def test_episode_rows_align_with_count(self, snapshot):
        assert len(snapshot["episodes"]) == snapshot["n_episodes"]
        assert len(snapshot["trajectories"]) == snapshot["n_episodes"]

    def test_trajectories_are_decimated_and_centred(self, snapshot):
        for trajectory in snapshot["trajectories"]:
            points = trajectory["points"]
            assert 2 <= len(points) <= 240, "hero payload must stay bounded"
            # n_frames is the true length and must survive decimation unchanged,
            # because the margin log reports it as the episode's real duration.
            assert trajectory["n_frames"] >= len(points)
            mean_x = sum(p[0] for p in points) / len(points)
            mean_y = sum(p[1] for p in points) / len(points)
            assert abs(mean_x) < 1e-6 and abs(mean_y) < 1e-6

    def test_agreement_matrix_accounts_for_every_episode(self, snapshot):
        """Counted plus excluded must equal the dataset.

        The matrix drops episodes carrying no ``task_name``, matching what the ARI is
        scored over -- otherwise "unknown" appears as a task label spanning every
        group and the table contradicts the number printed beside it. So the invariant
        is counted + excluded, not counted alone.
        """
        matrix = snapshot["agreement_matrix"]
        total = sum(sum(row) for row in matrix["counts"])
        assert total + matrix["excluded"] == snapshot["n_episodes"]
        assert len(matrix["counts"]) == len(matrix["clusters"])
        assert all(len(row) == len(matrix["tasks"]) for row in matrix["counts"])
        assert "unknown" not in [t.lower() for t in matrix["tasks"]]

    def test_diversity_score_is_positive(self, snapshot):
        assert snapshot["diversity_metrics"]["diversity_score"] > 0

    def test_config_round_trips(self, snapshot, client):
        assert snapshot["config"]["data_dir"] == config_for(client)["data_dir"]


class TestDomains:
    """The curated task groups behind the worked examples.

    They are imported from ``find_path.py`` rather than restated, so the browser demo
    and ``--domain`` on the CLI cannot drift; these tests pin the contract, not the
    contents.
    """

    def test_lists_domains_with_clip_counts(self, client):
        payload = strict_json(client.get("/api/domains", params=config_for(client)))
        assert payload, "expected at least one curated domain"
        for name, info in payload.items():
            assert isinstance(name, str)
            assert isinstance(info["tasks"], list)
            assert info["n_clips"] >= 0

    def test_domains_are_narrowed_to_the_loaded_dataset(self, client):
        payload = strict_json(client.get("/api/domains", params=config_for(client)))
        snapshot = strict_json(client.post("/api/run", json=config_for(client)))
        present = set(snapshot["tasks"])
        for info in payload.values():
            # A preset task absent from this dataset must not be reported as present.
            assert set(info["tasks"]) <= present

    def test_unknown_domain_is_rejected(self, client):
        payload = {**config_for(client), "goal": "anything", "domain": "no_such_domain"}
        assert client.post("/api/path", json=payload).status_code == 422

    def test_domain_with_no_matching_tasks_falls_back_to_full_graph(self, client):
        """A preset is a convenience, not a constraint the caller asked to enforce.

        Synthetic episodes carry none of the real preset task names, so every preset
        intersects to empty here -- which must route over the whole graph rather than
        error.
        """
        payload = {**config_for(client), "goal": "reach and place the object", "domain": "garments"}
        response = client.post("/api/path", json=payload)
        assert response.status_code == 200
        assert len(strict_json(response)["steps"]) >= 1


class TestGraph:
    @pytest.fixture(scope="class")
    @classmethod
    def graph(cls, client):
        return strict_json(client.post("/api/graph", json=config_for(client)))

    def test_nodes_are_indexed_in_order(self, graph):
        assert [node["index"] for node in graph["nodes"]] == list(range(len(graph["nodes"])))

    def test_edges_reference_real_clips_and_exclude_start(self, graph):
        n = len(graph["nodes"])
        for edge in graph["edges"]:
            # The virtual START node has no layout position, so it must never
            # appear as an endpoint here; it is reported via start_clips instead.
            assert isinstance(edge["from"], int) and isinstance(edge["to"], int)
            assert 0 <= edge["from"] < n and 0 <= edge["to"] < n

    def test_every_clip_is_reachable_from_start(self, graph):
        """`build_clip_graph` guarantees reachability, repairing edges if needed."""
        n = len(graph["nodes"])
        reachable = set(graph["start_clips"])
        adjacency: Dict[int, list] = {i: [] for i in range(n)}
        for edge in graph["edges"]:
            adjacency[edge["from"]].append(edge["to"])
        frontier = list(reachable)
        while frontier:
            for neighbour in adjacency[frontier.pop()]:
                if neighbour not in reachable:
                    reachable.add(neighbour)
                    frontier.append(neighbour)
        assert reachable == set(range(n))


class TestPath:
    @pytest.fixture(scope="class")
    @classmethod
    def path(cls, client):
        payload = {**config_for(client), "goal": "reach and place the object"}
        return strict_json(client.post("/api/path", json=payload))

    def test_step_count_equals_route_plus_reviews(self, path):
        assert len(path["steps"]) == len(path["route"]) + path["n_reviews"]

    def test_steps_are_numbered_consecutively(self, path):
        assert [step["step"] for step in path["steps"]] == list(range(1, len(path["steps"]) + 1))

    def test_first_step_has_no_incoming_edge(self, path):
        """Step 1 must report null, not 0.0 — nothing precedes it."""
        first = path["steps"][0]
        assert first["edge_weight"] is None
        assert first["ramp_cost"] is None
        assert first["difficulty_delta"] is None

    def test_review_steps_blank_ramp_cost_and_revisit_earlier_steps(self, path):
        for step in path["steps"]:
            if not step["is_review"]:
                continue
            # A rehearsal step is meant to drop in difficulty, so its ramp cost is
            # deliberately not measured; interference stays, because it is real.
            assert step["ramp_cost"] is None
            assert step["edge_weight"] is None
            assert step["reviews_step"] is not None
            assert step["reviews_step"] < step["step"]

    def test_route_ends_at_the_target(self, path):
        assert path["route"][-1] == path["target_index"]

    def test_match_reports_candidates_in_descending_score(self, path):
        scores = [candidate["score"] for candidate in path["match"]["candidates"]]
        assert scores == sorted(scores, reverse=True)

    def test_coverage_curve_is_non_decreasing(self, path):
        curve = path["coverage_curve"]
        assert len(curve) == len(path["steps"])
        assert all(b >= a for a, b in zip(curve, curve[1:]))

    def test_comparison_includes_the_path_itself(self, path):
        assert len(path["comparison"]) >= 1
        for row in path["comparison"].values():
            assert "spearman" in row

    def test_unknown_search_method_is_rejected(self, client):
        payload = {**config_for(client), "goal": "anything", "search": "bogus"}
        assert client.post("/api/path", json=payload).status_code == 422


class TestMatrix:
    def test_matrix_is_square_float32(self, client):
        response = client.post("/api/matrix", json=config_for(client))
        assert response.status_code == 200
        n = int(response.headers["x-matrix-n"])
        # float32 => 4 bytes per cell; a mismatch means a dtype or shape slip.
        assert len(response.content) == n * n * 4
        assert len(response.headers["x-episode-order"].split(",")) == n


class TestTrajectory:
    def test_returns_xyz_for_a_known_episode(self, client):
        snapshot = strict_json(client.post("/api/run", json=config_for(client)))
        episode_id = snapshot["episodes"][0]["episode_id"]
        payload = strict_json(
            client.get(f"/api/trajectory/{episode_id}", params=config_for(client))
        )
        assert payload["episode_id"] == episode_id
        assert payload["points"] and len(payload["points"][0]) == 3

    def test_unknown_episode_is_404(self, client):
        response = client.get("/api/trajectory/does__not__exist", params=config_for(client))
        assert response.status_code == 404


class TestErrors:
    def test_empty_directory_is_422_with_actionable_detail(self, client, tmp_path):
        response = client.post("/api/run", json={"data_dir": str(tmp_path)})
        assert response.status_code == 422
        # The message has to say what to do about it; an empty data dir is a setup
        # problem, and a bare "unprocessable entity" leaves the user stuck.
        assert "fetch_egoverse_data.py" in response.json()["detail"]

    def test_empty_task_scope_is_422(self, client):
        payload = {**config_for(client), "tasks": ["no_such_task"]}
        assert client.post("/api/graph", json=payload).status_code == 422
