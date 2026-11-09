"""Contract checks for parsing, sampling, cache integrity and EDA denominators."""

import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from util.loghub_systems import SOURCES, ensure_archive, load_thunderbird, parse_lines
from util.loghub_eda import (
    alert_episodes, dense_minutes, entity_counts, mine_templates,
    local_clock_offsets, minute_counts, reconstruct_events, session_candidates, window_summary,
)


class ParsingTests(unittest.TestCase):
    def test_syslog_year_rollover_and_unknown_timezone(self):
        frame = parse_lines("OpenSSH", [(1, "Dec 31 23:59:59 host sshd[7]: closing"), (2, "Jan  1 00:00:01 host sshd[7]: opening")])
        self.assertEqual((frame.timestamp.iloc[1] - frame.timestamp.iloc[0]).total_seconds(), 2)
        self.assertIsNone(frame.timestamp.dt.tz)
        self.assertTrue(frame.is_anomaly.isna().all())
        self.assertTrue(frame.level.isna().all())

    def test_mac_spaces_continuations_repeats_and_orphans(self):
        frame = parse_lines("Mac", [(1, "orphan"), (2, "Jul  1 09:00:55 host Microsoft Word[42]: first"), (3, "\ttrace"), (4, "Jul  1 09:00:56 --- last message repeated 2 times ---")])
        events = reconstruct_events(frame)
        self.assertEqual(events.physical_line_count.sum(), 4)
        self.assertEqual(len(events), 3)
        self.assertFalse(events.iloc[0].parse_ok)
        self.assertEqual(events.iloc[1].component, "Microsoft Word")
        self.assertEqual(events.iloc[1].message, "first\n\ttrace")
        self.assertEqual(events.iloc[2].component, "syslog_repeat")
        self.assertIsNone(events.iloc[2].host)

    def test_zookeeper_nested_context_and_milliseconds(self):
        frame = parse_lines("Zookeeper", [(1, "2015-07-29 17:41:44,747 - INFO  [QuorumPeer[myid=1]/0:0:0:0:0:0:0:0:2181:FastLeaderElection@774] - Notification time out: 3200")])
        self.assertEqual(frame.iloc[0].component, "FastLeaderElection")
        self.assertEqual(frame.iloc[0].timestamp.microsecond, 747000)
        self.assertIsNone(frame.iloc[0].host)
        self.assertEqual(frame.iloc[0].level, "INFO")

    def test_thunderbird_label_and_epoch(self):
        frame = parse_lines("Thunderbird", [(1, "- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 crond(pam_unix)[2915]: session closed"), (2, "FAIL 1131566462 2005.11.09 dn228 Nov 9 12:01:02 dn228/dn228 kernel: disk failed"), (3, "broken")])
        self.assertEqual(frame.is_anomaly.iloc[0], False)
        self.assertEqual(frame.is_anomaly.iloc[1], True)
        self.assertTrue(pd.isna(frame.is_anomaly.iloc[2]))
        self.assertEqual(str(frame.timestamp.dt.tz), "UTC")
        self.assertEqual(frame.iloc[0].component, "crond(pam_unix)")


class AnalysisTests(unittest.TestCase):
    def test_local_clock_year_boundary(self):
        events = pd.DataFrame({"timestamp": pd.to_datetime(["2006-01-01 07:59:00"], utc=True), "local_time": ["Dec 31 23:59:00"]})
        self.assertEqual(local_clock_offsets(events).iloc[0], -8)

    def test_empty_windows_and_count_conservation(self):
        events = parse_lines("OpenSSH", [(1, "Dec 10 00:00:00 host sshd[1]: first"), (2, "Dec 10 00:02:00 host sshd[2]: second")])
        minutes = minute_counts(events)
        dense = dense_minutes(minutes)
        self.assertEqual(dense["count"].tolist(), [1, 0, 1])
        self.assertEqual(window_summary(minutes).loc[1, "empty"], 1)
        self.assertEqual(window_summary(minutes).loc[5, "max"], 2)

    def test_alert_episode_gap(self):
        index = pd.date_range("2005-01-01", periods=4, freq="min", tz="UTC")
        minutes = pd.DataFrame({"count": [3, 2, 0, 1], "alerts": [2, 1, 0, 1]}, index=index)
        episodes = alert_episodes(minutes)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes.active_minutes.max(), 2)
        self.assertEqual(episodes.alerts.sum(), 4)

    def test_entity_validation_and_explicit_sessions(self):
        events = pd.DataFrame({"message": ["from 192.0.2.1 port 65535 bad 999.1.2.3 port 99999 ::1 session id 0xabc12345 /var/log/a https://example.org/x"]})
        summary, examples = entity_counts(events)
        self.assertEqual(summary.loc["ipv4", "unique"], 1)
        self.assertEqual(summary.loc["ipv6", "unique"], 1)
        self.assertEqual(summary.loc["port", "mentions"], 1)
        self.assertEqual(examples["session_id"].iloc[0].value, "0xabc12345")

    def test_pid_reuse_is_split_by_idle_time(self):
        events = parse_lines("OpenSSH", [(1, "Dec 10 00:00:00 host sshd[1]: first"), (2, "Dec 10 01:00:00 host sshd[1]: second")])
        self.assertEqual(len(session_candidates(events, "OpenSSH")), 2)

    def test_templates_use_final_cluster_text_and_deterministic_ids(self):
        events = parse_lines("OpenSSH", [(1, "Dec 10 00:00:00 host sshd[1]: user alice logged in"), (2, "Dec 10 00:00:01 host sshd[2]: user bob logged in"), (3, "Dec 10 00:00:02 host sshd[3]: user alice logged in")])
        frame, templates = mine_templates(events)
        second, _ = mine_templates(events)
        self.assertEqual(templates["count"].sum(), 3)
        self.assertTrue(frame.groupby("template_id").template.nunique().eq(1).all())
        self.assertEqual(frame.template_id.tolist(), second.template_id.tolist())
        self.assertIn("<*>", frame.template.iloc[0])


class StreamingTests(unittest.TestCase):
    def test_full_scan_cache_and_disjoint_analysis_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "Thunderbird.tar.gz"
            content = "".join(f"{'ALERT' if i % 10 < 5 else '-'} {1131566460+i} 2005.11.09 dn1 Nov 9 12:01:01 dn1/dn1 kernel: event {i}\n" for i in range(100)).encode()
            with tarfile.open(archive, "w:gz") as handle:
                member = tarfile.TarInfo("../../Thunderbird.log")
                member.size = len(content)
                handle.addfile(member, io.BytesIO(content))
            source = {"archive": archive.name, "md5": hashlib.md5(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size, "published_lines": 100}
            with patch.dict(SOURCES, {"Thunderbird": source}), patch("util.loghub_systems.SCAN_BATCH_BYTES", 250):
                frame, meta, minutes = load_thunderbird(directory, sample_size=20, prefix_size=10)
                self.assertEqual(meta["physical_lines"], 100)
                self.assertEqual(minutes["count"].sum(), 100)
                self.assertEqual(minutes.alerts.sum(), 50)
                self.assertEqual(meta["longest_alert_run_file_order"], 5)
                self.assertEqual(meta["timestamp_backsteps"], 0)
                self.assertEqual(meta["sha256_raw"], hashlib.sha256(content).hexdigest())
                self.assertEqual(frame.sample_kind.eq("random").sum(), 20)
                self.assertTrue(frame.loc[frame.sample_kind.eq("random"), "line_no"].is_unique)
                self.assertEqual(frame.loc[frame.sample_kind.eq("prefix"), "line_no"].tolist(), list(range(1, 11)))
                again, _, cached_minutes = load_thunderbird(directory, sample_size=20, prefix_size=10)
                pd.testing.assert_frame_equal(frame, again)
                pd.testing.assert_frame_equal(minutes, cached_minutes)
                self.assertFalse((directory.parent / "Thunderbird.log").exists())
                cache = next(directory.glob("*-minutes.csv.gz"))
                cache.write_bytes(b"bad cache")
                with self.assertRaisesRegex(ValueError, "cache checksum"):
                    load_thunderbird(directory, sample_size=20, prefix_size=10)

    def test_corrupt_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "Mac.tar.gz").write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "Corrupt archive"):
                ensure_archive("Mac", temporary)


if __name__ == "__main__":
    unittest.main()
