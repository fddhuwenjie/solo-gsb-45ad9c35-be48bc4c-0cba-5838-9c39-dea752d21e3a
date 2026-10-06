"""End-to-end tests for POST /sessions/{id}/history (incremental reads).

Every committed revision persists an immutable snapshot of the document
merged from the initial document, so a history read returns each
revision's ``/route-changes`` events relative to the *previous*
revision.  Late-arriving concurrent edits change the latest merge but
must never rewrite events already committed for an earlier revision.
"""
from __future__ import annotations

import glob
import json
import multiprocessing
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.models import Operation
from app.sessions import HISTORY_PAGE_SIZE, SessionStore


def op(actor, seq, version, path, action, value=None, index=None):
    body = {
        "actor": actor,
        "seq": seq,
        "version": version,
        "path": path,
        "action": action,
        "value": value,
    }
    if index is not None:
        body["index"] = index
    return body


def sub(subscriber_id, include=("/**",), exclude=()):
    return {
        "subscriber_id": subscriber_id,
        "include": list(include),
        "exclude": list(exclude),
    }


def read_history(client, session_id, after_revision, subscriptions=None):
    return client.post(
        f"/sessions/{session_id}/history",
        json={
            "after_revision": after_revision,
            "subscriptions": subscriptions or [sub("all")],
        },
    )


def events_of(body, revision, subscriber_id="all"):
    entry = next(r for r in body["revisions"] if r["revision"] == revision)
    route = next(r for r in entry["routes"] if r["subscriber_id"] == subscriber_id)
    return route["events"]


def create_with(client, document):
    return client.post(
        "/sessions", json={"document": document, "version": {}}
    ).json()["session_id"]


def commit(client, session_id, expected, operations):
    resp = client.post(
        f"/sessions/{session_id}/batches",
        json={"expected_revision": expected, "operations": operations},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# basic incremental reading
# ---------------------------------------------------------------------------


def test_history_of_brand_new_session_is_empty_at_cursor_zero(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {"a": 1})
    resp = read_history(client, sid, 0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["revisions"] == []
    assert body["last_revision"] == 0  # cursor echoed with no new revisions
    assert body["latest_revision"] == 0
    assert body["has_more"] is False


def test_revision_events_are_relative_to_previous_revision(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {"a": 1, "b": [1]})
    commit(client, sid, 0, [
        op("alice", 1, {"alice": 1}, "/a", "set", 2),
        op("alice", 2, {"alice": 2}, "/b", "insert", 9, index=0),
    ])
    commit(client, sid, 1, [
        op("bob", 1, {"bob": 1}, "/c", "set", "x"),
    ])
    resp = read_history(client, sid, 0)
    assert resp.status_code == 200
    body = resp.json()
    assert [r["revision"] for r in body["revisions"]] == [1, 2]
    assert body["last_revision"] == 2
    assert body["latest_revision"] == 2
    assert body["has_more"] is False

    assert events_of(body, 1) == [
        {"type": "replace", "path": "/a", "old": 1, "new": 2},
        {"type": "replace", "path": "/b", "old": [1], "new": [9, 1]},
    ]
    # revision 2 is compared with revision 1's document, not the initial one
    assert events_of(body, 2) == [
        {"type": "add", "path": "/c", "new": "x"},
    ]


def test_events_respect_subscriber_include_exclude_rules(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {"users": {"u1": 1}, "secret": 0})
    commit(client, sid, 0, [
        op("alice", 1, {"alice": 1}, "/users/u1", "set", 2),
        op("alice", 2, {"alice": 2}, "/secret", "set", 1),
    ])
    resp = read_history(
        client, sid, 0,
        [sub("public", include=("/**",), exclude=("/secret",))],
    )
    body = resp.json()
    entry = body["revisions"][0]
    assert entry["routes"] == [
        {
            "subscriber_id": "public",
            "events": [
                {"type": "replace", "path": "/users/u1", "old": 1, "new": 2}
            ],
        }
    ]
    assert entry["unrouted"] == [
        {"type": "replace", "path": "/secret", "old": 0, "new": 1}
    ]


def test_subscriptions_can_differ_between_pages(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {"a": 1, "b": 1})
    commit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 2)])
    commit(client, sid, 1, [op("alice", 2, {"alice": 2}, "/b", "set", 3)])

    only_a = read_history(client, sid, 0, [sub("a", include=("/a",))]).json()
    assert [r["revision"] for r in only_a["revisions"]] == [1, 2]
    assert only_a["revisions"][0]["routes"][0]["events"] == [
        {"type": "replace", "path": "/a", "old": 1, "new": 2}
    ]
    assert only_a["revisions"][1]["routes"][0]["events"] == []

    resume = read_history(
        client, sid, 1, [sub("b", include=("/b",))]
    ).json()
    assert [r["revision"] for r in resume["revisions"]] == [2]
    assert resume["revisions"][0]["routes"][0]["events"] == [
        {"type": "replace", "path": "/b", "old": 1, "new": 3}
    ]


# ---------------------------------------------------------------------------
# cursor semantics
# ---------------------------------------------------------------------------


def test_cursor_advances_and_resumes_without_repeating_revisions(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {"k": 0})
    commit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/k", "set", 1)])

    first = read_history(client, sid, 0).json()
    assert [r["revision"] for r in first["revisions"]] == [1]
    assert first["last_revision"] == 1

    # nothing committed since the cursor -> empty page, cursor echoed
    idle = read_history(client, sid, first["last_revision"]).json()
    assert idle["revisions"] == []
    assert idle["last_revision"] == 1
    assert idle["latest_revision"] == 1

    commit(client, sid, 1, [op("alice", 2, {"alice": 2}, "/k", "set", 2)])
    second = read_history(client, sid, idle["last_revision"]).json()
    assert [r["revision"] for r in second["revisions"]] == [2]
    assert events_of(second, 2) == [
        {"type": "replace", "path": "/k", "old": 1, "new": 2}
    ]


def test_committed_revision_with_unchanged_document_keeps_empty_events(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {"k": 1})
    # writing the same value still bumps the revision (a new op is stored);
    # the document does not change, so the revision entry has empty events
    commit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/k", "set", 1)])
    body = read_history(client, sid, 0).json()
    assert [r["revision"] for r in body["revisions"]] == [1]
    assert body["revisions"][0]["routes"][0]["events"] == []
    assert body["revisions"][0]["unrouted"] == []
    assert body["last_revision"] == 1


def test_retransmitted_all_duplicate_batch_adds_no_history(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {})
    batch = [op("alice", 1, {"alice": 1}, "/x", "set", 1)]
    commit(client, sid, 0, batch)
    again = client.post(
        f"/sessions/{sid}/batches",
        json={"expected_revision": 1, "operations": batch},
    )
    assert again.status_code == 200
    assert again.json()["revision"] == 1  # no new revision
    body = read_history(client, sid, 0).json()
    assert [r["revision"] for r in body["revisions"]] == [1]


def test_failed_batch_adds_no_history(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {})
    commit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    # invalid batch (gap in alice's seq) -> 422, revision stays at 1
    bad = client.post(
        f"/sessions/{sid}/batches",
        json={
            "expected_revision": 1,
            "operations": [op("alice", 3, {"alice": 3}, "/y", "set", 2)],
        },
    )
    assert bad.status_code == 422
    # stale revision -> 409, also no new revision
    stale = client.post(
        f"/sessions/{sid}/batches",
        json={
            "expected_revision": 0,
            "operations": [op("bob", 1, {"bob": 1}, "/y", "set", 2)],
        },
    )
    assert stale.status_code == 409
    body = read_history(client, sid, 0).json()
    assert [r["revision"] for r in body["revisions"]] == [1]
    assert body["latest_revision"] == 1


# ---------------------------------------------------------------------------
# immutable per-revision history vs late concurrent edits
# ---------------------------------------------------------------------------


def test_late_concurrent_edit_changes_current_merge_but_not_prior_events(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {"title": "draft"})
    commit(client, sid, 0, [
        op("alice", 1, {"alice": 1}, "/title", "set", "alice-v"),
    ])
    first = read_history(client, sid, 0).json()
    rev1_before = first["revisions"][0]["routes"][0]["events"]
    assert rev1_before == [
        {"type": "replace", "path": "/title", "old": "draft", "new": "alice-v"}
    ]

    # bob's concurrent write arrives a whole revision later
    commit(client, sid, 1, [
        op("bob", 1, {"bob": 1}, "/title", "set", "bob-v"),
    ])
    state = client.get(f"/sessions/{sid}").json()
    assert state["document"] == {"title": "bob-v"}  # ("bob",1) wins now
    assert len(state["conflicts"]) == 1

    after = read_history(client, sid, 0).json()
    assert [r["revision"] for r in after["revisions"]] == [1, 2]
    # revision 1's events are byte-for-byte what they were at commit time
    assert after["revisions"][0]["routes"][0]["events"] == rev1_before
    # revision 2 diffs against revision 1's snapshot
    assert after["revisions"][1]["routes"][0]["events"] == [
        {"type": "replace", "path": "/title", "old": "alice-v", "new": "bob-v"}
    ]


# ---------------------------------------------------------------------------
# paging: at most 20 revisions per request
# ---------------------------------------------------------------------------


def test_history_pages_through_at_most_twenty_revisions(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {"k": 0})
    revision = 0
    for i in range(1, HISTORY_PAGE_SIZE * 2 + 6):
        revision = commit(
            client, sid, revision,
            [op("alice", i, {"alice": i}, "/k", "set", i)],
        )["revision"]
    assert revision == HISTORY_PAGE_SIZE * 2 + 5

    cursor = 0
    seen: list[int] = []
    for _ in range(10):
        page = read_history(client, sid, cursor).json()
        revs = [r["revision"] for r in page["revisions"]]
        assert len(revs) <= HISTORY_PAGE_SIZE
        assert revs == sorted(revs)
        assert all(r > cursor for r in revs)
        seen.extend(revs)
        assert page["last_revision"] == (revs[-1] if revs else cursor)
        assert page["has_more"] is (page["last_revision"] < revision)
        cursor = page["last_revision"]
        if not page["has_more"]:
            break
    assert seen == list(range(1, revision + 1))

    tail = read_history(client, sid, revision).json()
    assert tail["revisions"] == []
    assert tail["last_revision"] == revision
    assert tail["has_more"] is False


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_unknown_session_history_returns_404(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    missing = "0" * 32
    resp = read_history(client, missing, 0)
    assert resp.status_code == 404


def test_cursor_above_current_revision_returns_422(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {})
    commit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    resp = read_history(client, sid, 5)
    assert resp.status_code == 422
    assert "outside the readable history" in resp.json()["detail"]


def test_negative_cursor_returns_422(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {})
    resp = client.post(
        f"/sessions/{sid}/history",
        json={
            "after_revision": -1,
            "subscriptions": [sub("all")],
        },
    )
    assert resp.status_code == 422


def test_invalid_subscription_pattern_returns_422(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {})
    commit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    resp = read_history(client, sid, 0, [sub("bad", include=("/x*-y",))])
    assert resp.status_code == 422
    assert "wildcard must be an entire segment" in resp.json()["detail"]


def test_duplicate_subscriber_ids_return_422(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {})
    resp = client.post(
        f"/sessions/{sid}/history",
        json={
            "after_revision": 0,
            "subscriptions": [
                sub("s", include=("/a",)),
                sub("s", include=("/b",)),
            ],
        },
    )
    assert resp.status_code == 422
    assert "duplicate subscriber_id" in resp.json()["detail"]


def test_subscription_count_out_of_bounds_returns_422(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {})
    none = client.post(
        f"/sessions/{sid}/history",
        json={"after_revision": 0, "subscriptions": []},
    )
    assert none.status_code == 422
    many = client.post(
        f"/sessions/{sid}/history",
        json={
            "after_revision": 0,
            "subscriptions": [
                sub(f"s{i}") for i in range(101)
            ],
        },
    )
    assert many.status_code == 422


def test_missing_session_beats_invalid_subscription_for_404(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    resp = read_history(
        client, "0" * 32, 0, [sub("bad", include=("/a*",))]
    )
    assert resp.status_code == 404


def test_cursor_alias_revision_is_accepted(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    client = TestClient(app)
    sid = create_with(client, {})
    commit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    resp = client.post(
        f"/sessions/{sid}/history",
        json={"revision": 0, "subscriptions": [sub("all")]},
    )
    assert resp.status_code == 200
    assert [r["revision"] for r in resp.json()["revisions"]] == [1]


# ---------------------------------------------------------------------------
# durability across restarts
# ---------------------------------------------------------------------------


def test_history_survives_restart(tmp_path):
    store_dir = str(tmp_path / "sessions")
    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    sid = create_with(client, {"k": 0})
    commit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/k", "set", 1)])
    commit(client, sid, 1, [op("alice", 2, {"alice": 2}, "/k", "set", 2)])

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    body = read_history(client, sid, 0).json()
    assert [r["revision"] for r in body["revisions"]] == [1, 2]
    assert events_of(body, 1) == [
        {"type": "replace", "path": "/k", "old": 0, "new": 1}
    ]
    assert events_of(body, 2) == [
        {"type": "replace", "path": "/k", "old": 1, "new": 2}
    ]

    commit(client, sid, 2, [op("alice", 3, {"alice": 3}, "/k", "set", 3)])
    body = read_history(client, sid, 2).json()
    assert [r["revision"] for r in body["revisions"]] == [3]
    assert events_of(body, 3) == [
        {"type": "replace", "path": "/k", "old": 2, "new": 3}
    ]


# ---------------------------------------------------------------------------
# sessions written by older builds
# ---------------------------------------------------------------------------


def _strip_history_files(store_dir: str) -> None:
    for path in glob.glob(str(Path(store_dir) / "*.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data.pop("history_start", None)
        data.pop("snapshots", None)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)


def test_legacy_session_history_start_is_current_revision_on_first_history_read(
    tmp_path,
):
    store_dir = str(tmp_path / "sessions")
    store = SessionStore(store_dir)
    rec = store.create({"count": 0}, {})
    store.submit_batch(
        rec.session_id, 0,
        [Operation(actor="a", seq=1, version={"a": 1}, path="/count",
                   action="set", value=1)],
    )
    store.submit_batch(
        rec.session_id, 1,
        [Operation(actor="a", seq=2, version={"a": 2}, path="/count",
                   action="set", value=2)],
    )
    _strip_history_files(store_dir)

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    sid = rec.session_id

    # cursor older than the visible current revision -> 422
    early = read_history(client, sid, 1)
    assert early.status_code == 422
    assert "outside the readable history" in early.json()["detail"]
    zero = read_history(client, sid, 0)
    assert zero.status_code == 422

    # current revision is the boundary: empty page, cursor echoed
    current = read_history(client, sid, 2)
    assert current.status_code == 200
    body = current.json()
    assert body["revisions"] == []
    assert body["last_revision"] == 2
    assert body["latest_revision"] == 2

    # the boundary persisted to disk (a restarted store sees the same one)
    raw = json.loads(Path(store_dir, f"{sid}.json").read_text())
    assert raw["history_start"] == 2
    assert raw["snapshots"] == [{"count": 2}]

    # new commits then extend history from the boundary
    commit(client, sid, 2, [
        op("a", 3, {"a": 3}, "/count", "set", 3),
    ])
    body = read_history(client, sid, 2).json()
    assert [r["revision"] for r in body["revisions"]] == [3]
    assert events_of(body, 3) == [
        {"type": "replace", "path": "/count", "old": 2, "new": 3}
    ]
    # still no way to read revisions older than the adopted boundary
    assert read_history(client, sid, 2).status_code == 200
    assert read_history(client, sid, 1).status_code == 422


def test_legacy_session_boundary_established_on_get_state(tmp_path):
    store_dir = str(tmp_path / "sessions")
    store = SessionStore(store_dir)
    rec = store.create({"x": 1}, {})
    store.submit_batch(
        rec.session_id, 0,
        [Operation(actor="a", seq=1, version={"a": 1}, path="/x",
                   action="set", value=2)],
    )
    _strip_history_files(store_dir)

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    assert client.get(f"/sessions/{rec.session_id}").status_code == 200
    raw = json.loads(Path(store_dir, f"{rec.session_id}.json").read_text())
    assert raw["history_start"] == 1
    assert raw["snapshots"] == [{"x": 2}]
    # a cursor older than the adopted boundary is rejected
    assert read_history(client, rec.session_id, 0).status_code == 422


def test_legacy_session_at_revision_zero_has_boundary_zero(tmp_path):
    store_dir = str(tmp_path / "sessions")
    rec = SessionStore(store_dir).create({"x": 1}, {})
    _strip_history_files(store_dir)

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    resp = read_history(client, rec.session_id, 0)
    assert resp.status_code == 200
    assert resp.json()["revisions"] == []


def test_legacy_session_boundary_established_on_first_valid_batch(tmp_path):
    store_dir = str(tmp_path / "sessions")
    store = SessionStore(store_dir)
    rec = store.create({"count": 0}, {})
    store.submit_batch(
        rec.session_id, 0,
        [Operation(actor="a", seq=1, version={"a": 1}, path="/count",
                   action="set", value=1)],
    )
    _strip_history_files(store_dir)  # history_start absent, current rev = 1

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    sid = rec.session_id
    # a stale batch (409) must not establish/persist the boundary
    stale = client.post(
        f"/sessions/{sid}/batches",
        json={
            "expected_revision": 0,
            "operations": [op("b", 1, {"b": 1}, "/y", "set", 1)],
        },
    )
    assert stale.status_code == 409
    raw = json.loads(Path(store_dir, f"{sid}.json").read_text())
    assert "history_start" not in raw

    # an invalid batch (422) must not establish/persist it either
    invalid = client.post(
        f"/sessions/{sid}/batches",
        json={
            "expected_revision": 1,
            "operations": [op("a", 3, {"a": 3}, "/y", "set", 1)],
        },
    )
    assert invalid.status_code == 422
    raw = json.loads(Path(store_dir, f"{sid}.json").read_text())
    assert "history_start" not in raw

    # the valid next commit adopts rev 1 as boundary and snapshots rev 2
    commit(client, sid, 1, [op("a", 2, {"a": 2}, "/count", "set", 7)])
    body = read_history(client, sid, 1).json()
    assert [r["revision"] for r in body["revisions"]] == [2]
    assert events_of(body, 2) == [
        {"type": "replace", "path": "/count", "old": 1, "new": 7}
    ]
    assert read_history(client, sid, 0).status_code == 422


def test_legacy_session_all_duplicate_retransmission_establishes_boundary(
    tmp_path,
):
    store_dir = str(tmp_path / "sessions")
    store = SessionStore(store_dir)
    rec = store.create({"x": 1}, {})
    dup = Operation(actor="a", seq=1, version={"a": 1}, path="/x",
                    action="set", value=2)
    store.submit_batch(rec.session_id, 0, [dup])
    _strip_history_files(store_dir)

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    outcome = client.post(
        f"/sessions/{rec.session_id}/batches",
        json={"expected_revision": 1, "operations": [dup.model_dump()]},
    )
    assert outcome.status_code == 200
    assert outcome.json()["revision"] == 1
    assert outcome.json()["accepted"] == 0
    raw = json.loads(Path(store_dir, f"{rec.session_id}.json").read_text())
    assert raw["history_start"] == 1  # boundary persisted, no new history rev
    assert raw["snapshots"] == [{"x": 2}]


# ---------------------------------------------------------------------------
# cross-process history boundary + commits share one lock
# ---------------------------------------------------------------------------


def _legacy_racing_worker(store_dir, session_id, actor, barrier, queue):
    """First-touch worker: establish a legacy boundary while racing a commit."""
    from app.models import Operation
    from app.sessions import RevisionConflictError, SessionStore

    store = SessionStore(store_dir)
    barrier.wait(timeout=15)
    operation = Operation(
        actor=actor, seq=1, version={actor: 1},
        path=f"/{actor}", action="set", value=actor,
    )
    try:
        outcome = store.submit_batch(session_id, 0, [operation])
        status = f"committed:{outcome.revision}"
    except RevisionConflictError:
        status = "conflict"
    queue.put(status)


def test_legacy_boundary_and_revision_share_cross_process_commit(tmp_path):
    from app.models import Subscription

    store_dir = str(tmp_path / "sessions")
    # legacy file at revision 0 without history
    rec = SessionStore(store_dir).create({}, {})
    _strip_history_files(store_dir)

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_legacy_racing_worker,
            args=(store_dir, rec.session_id, f"actor{i}", barrier, queue),
        )
        for i in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    statuses = sorted(queue.get(timeout=10) for _ in procs)
    # exactly one commit wins; the loser conflicts (no lost revision)
    assert statuses == ["committed:1", "conflict"], statuses

    raw = json.loads(Path(store_dir, f"{rec.session_id}.json").read_text())
    # winner atomically established boundary 0 plus the revision-1 snapshot
    assert raw["history_start"] == 0
    assert raw["snapshots"][0] == {}
    assert len(raw["snapshots"]) == 2

    # a fresh store process reads the persisted history straight away
    fresh = SessionStore(store_dir)
    body = fresh.read_history(
        rec.session_id, 0, [Subscription(subscriber_id="all", include=["/**"])]
    )
    assert [r["revision"] for r in body["revisions"]] == [1]
    assert body["last_revision"] == 1 and body["has_more"] is False
