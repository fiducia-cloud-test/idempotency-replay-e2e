#!/usr/bin/env python3
"""Cross-organization idempotency and replay tests for DEN-2797 recovery."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import unittest
from pathlib import Path

NOW = "2026-08-08T21:30:00Z"


def configure_source(source_root: Path) -> None:
    source_root = source_root.resolve()
    for path in (source_root / "tools", source_root / "scripts"):
        if not path.is_dir():
            raise RuntimeError(f"missing pinned source directory: {path}")
        sys.path.insert(0, str(path))


class ArtifactRecoveryIdempotencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import artifact_recovery_ledger as recovery  # type: ignore
        import build_artifact_recovery_backfill as backfill  # type: ignore

        cls.recovery = recovery
        cls.backfill = backfill

    def fixture(self) -> dict:
        return self.backfill.build_fixture()

    def reconcile(self, value: dict, previous: dict | None = None):
        return self.recovery.reconcile(
            value,
            previous,
            now=NOW,
            batch_size=50,
            target_task_id=self.recovery.DEFAULT_CLI_TASK_ID,
        )

    @staticmethod
    def identity(item: dict) -> str:
        return item["target"]["identity"].lower()

    def test_reversing_observation_order_is_byte_stable(self) -> None:
        forward = self.fixture()
        reverse = copy.deepcopy(forward)
        reverse["items"].reverse()

        forward_ledger, forward_queue = self.reconcile(forward)
        reverse_ledger, reverse_queue = self.reconcile(reverse)

        self.assertEqual(forward_ledger, reverse_ledger)
        self.assertEqual(forward_queue, reverse_queue)
        self.assertEqual(
            json.dumps(forward_ledger, sort_keys=True, separators=(",", ":")),
            json.dumps(reverse_ledger, sort_keys=True, separators=(",", ":")),
        )

    def test_remote_completion_updates_one_row_and_clears_one_cli_item(self) -> None:
        initial = self.fixture()
        first_ledger, first_queue = self.reconcile(initial)
        target_identity = "apostille-me/apme-e2e"
        self.assertIn(
            target_identity,
            {
                f"{item['owner'].lower()}/{item['repository'].lower()}"
                for item in first_queue["items"]
            },
        )

        refreshed = copy.deepcopy(initial)
        observation = next(
            item for item in refreshed["items"] if self.identity(item) == target_identity
        )
        owner = observation["target"]["owner"]
        repository = observation["target"]["repository"]
        repository_url = f"https://github.com/{owner}/{repository}"
        branch = (
            observation["intent"].get("branch")
            or observation["local"].get("branch")
            or "agent/den-2797-recovery-review"
        )
        head_sha = observation["local"].get("head_sha")
        self.assertRegex(head_sha or "", r"^[0-9a-f]{40}$")
        pull_url = f"{repository_url}/pull/1"

        observation["local"]["remote_present"] = True
        observation["remote"] = {
            "collected": True,
            "repository": {
                "exists": True,
                "visibility": observation["target"]["visibility"],
                "default_branch": "main",
                "url": repository_url,
            },
            "branches": [
                {"name": "main", "sha": head_sha},
                {"name": branch, "sha": head_sha},
            ],
            "commits": [
                {"sha": head_sha, "url": f"{repository_url}/commit/{head_sha}"}
            ],
            "pull_requests": [
                {
                    "number": 1,
                    "url": pull_url,
                    "head": branch,
                    "base": "main",
                    "state": "open",
                    "draft": True,
                }
            ],
        }
        observation["claims"] = {
            "repository_url": repository_url,
            "commit_sha": head_sha,
            "branch": branch,
            "pull_request_url": pull_url,
        }

        second_ledger, second_queue = self.reconcile(refreshed, first_ledger)
        first_entries = {
            entry["observation"]["target"]["identity"].lower(): entry
            for entry in first_ledger["entries"].values()
        }
        second_entries = {
            entry["observation"]["target"]["identity"].lower(): entry
            for entry in second_ledger["entries"].values()
        }

        self.assertEqual(set(first_entries), set(second_entries))
        self.assertEqual(
            second_entries[target_identity]["attempts"],
            first_entries[target_identity]["attempts"] + 1,
        )
        self.assertEqual(
            second_entries[target_identity]["classification"]["status"], "complete"
        )
        self.assertEqual(
            second_entries[target_identity]["classification"]["next_action"], "none"
        )
        for identity, entry in first_entries.items():
            if identity != target_identity:
                self.assertEqual(
                    entry["attempts"],
                    second_entries[identity]["attempts"],
                    identity,
                )
        self.assertNotIn(
            target_identity,
            {
                f"{item['owner'].lower()}/{item['repository'].lower()}"
                for item in second_queue["items"]
            },
        )
        self.assertEqual(
            second_queue["summary"]["items"], first_queue["summary"]["items"] - 1
        )

    def test_same_origin_cannot_collapse_distinct_owner_repository_targets(self) -> None:
        source = self.fixture()
        first = next(
            copy.deepcopy(item)
            for item in source["items"]
            if self.identity(item) == "apostille-me/apme-e2e"
        )
        second = copy.deepcopy(first)
        second["target"].update(
            owner="example-test-owner",
            repository="example-recovery-e2e",
            identity="example-test-owner/example-recovery-e2e",
        )
        second["claims"] = {
            "repository_url": None,
            "commit_sha": None,
            "branch": None,
            "pull_request_url": None,
        }
        value = copy.deepcopy(source)
        value["items"] = [first, second]

        ledger, queue = self.reconcile(value)
        self.assertEqual(ledger["summary"]["entries"], 2)
        self.assertEqual(ledger["summary"]["actionable"], 2)
        self.assertEqual(len(ledger["entries"]), 2)
        queued = {
            f"{item['owner'].lower()}/{item['repository'].lower()}"
            for item in queue["items"]
        }
        self.assertEqual(
            queued,
            {
                "apostille-me/apme-e2e",
                "example-test-owner/example-recovery-e2e",
            },
        )

    def test_repeated_material_changes_are_monotonic_and_exact_replay_is_stable(self) -> None:
        original = self.fixture()
        first, _ = self.reconcile(original)

        changed_once = copy.deepcopy(original)
        changed_once["items"][0]["note"] += " first-refresh"
        second, second_queue = self.reconcile(changed_once, first)

        changed_twice = copy.deepcopy(changed_once)
        changed_twice["items"][0]["note"] += " second-refresh"
        third, third_queue = self.reconcile(changed_twice, second)
        replay, replay_queue = self.reconcile(changed_twice, third)

        key = next(
            key
            for key, entry in first["entries"].items()
            if entry["observation"]["origin"]["id"]
            == original["items"][0]["origin"]["id"]
            and entry["observation"]["target"]["identity"]
            == original["items"][0]["target"]["identity"]
        )
        self.assertEqual(second["entries"][key]["attempts"], 2)
        self.assertEqual(third["entries"][key]["attempts"], 3)
        self.assertEqual(third, replay)
        self.assertEqual(third_queue, replay_queue)
        self.assertEqual(second_queue["summary"], third_queue["summary"])

    def test_green_draft_remote_evidence_is_terminal_for_delivery_recovery(self) -> None:
        ledger, queue = self.reconcile(self.fixture())
        complete_drafts = {
            entry["observation"]["target"]["identity"].lower(): entry
            for entry in ledger["entries"].values()
            if entry["classification"]["status"] == "complete"
            and any(
                pull.get("draft") and pull.get("state") == "open"
                for pull in entry["observation"]["remote"]["pull_requests"]
            )
        }
        self.assertTrue(complete_drafts)
        for identity, entry in complete_drafts.items():
            with self.subTest(identity=identity):
                self.assertEqual(entry["classification"]["next_action"], "none")
                self.assertTrue(any("/pull/" in link for link in entry["evidence_links"]))
        queued = {
            f"{item['owner'].lower()}/{item['repository'].lower()}"
            for item in queue["items"]
        }
        self.assertTrue(set(complete_drafts).isdisjoint(queued))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    configure_source(args.source_root)
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
