"""Small raw fixtures exercise counts, boundary handling and sampling semantics."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
import hashlib

from util.loghub_eda import (
    Reservoir, _audit_windows_blocks, audit_dataset, entities, parse_line, parse_time,
    window_counts,
    exact_bgl_duplicates,
    reread_sample_records,
)


class LoghubAuditTests(unittest.TestCase):
    def test_bgl_empty_message_is_a_valid_record(self):
        row = parse_line("BGL", "- 1117838570 2005.06.03 R02 2005-06-03-15.42.50.363779 R02 RAS KERNEL INFO")
        self.assertEqual(row["message"], "")
        self.assertEqual(row["label"], "-")

    def test_healthapp_unpadded_date_and_milliseconds(self):
        self.assertEqual(parse_time("HealthApp", "201811-0:0:0:1"), datetime(2018, 1, 1, microsecond=1000))
        self.assertEqual(parse_time("HealthApp", "20171223-22:15:29:606"), datetime(2017, 12, 23, 22, 15, 29, 606000))
        with self.assertRaises(ValueError):
            parse_time("HealthApp", "2018111-0:0:0:0")
        self.assertIsNotNone(parse_line("HealthApp", "\x00201812-0:0:0:1|Step|42|a|b"))

    def test_reservoir_reproducible_and_batch_equivalent(self):
        a, b = Reservoir(17, 42), Reservoir(17, 42)
        for i in range(10000):
            a.add(i)
        for size in (1, 13, 100, 2345, 7541):
            for i in b.batch_indices(size):
                b.add(i)
        self.assertEqual(a.items, b.items)
        self.assertEqual(a.seen, b.seen)
        self.assertTrue(any(i > 9000 for i in a.items))

    def test_sample_inclusion_not_biased_to_file_prefix(self):
        counts = Counter()
        for seed in range(1000):
            sample = Reservoir(5, seed)
            for i in range(50):
                sample.add(i)
            counts.update(sample.items)
        self.assertGreater(min(counts.values()), 60)
        self.assertLess(max(counts.values()), 140)

    def test_windows_block_boundaries_bom_multiline_and_missing_final_lf(self):
        raw = (b"\xef\xbb\xbf2016-09-28 04:30:30, Info CBS first\r\n"
               b"continuation\r\n"
               b"2016-09-28 04:30:30, Warning CSI second\r\n"
               b"2016-09-28 04:29:00, Info CBS third\r\n"
               b"2016-09-28 04:34:00, Info CBS fourth")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Windows.log"
            path.write_bytes(raw)
            (Path(tmp) / "Windows.tar.gz").touch()
            # Force every header and continuation across read-block boundaries.
            result = _audit_windows_blocks(path, 100, 42, block_size=8)
            c = result["counts"]
            self.assertEqual(c["physical_lines"], 5)
            self.assertEqual(c["records"], 4)
            self.assertEqual(c["headerless_lines"], 1)
            self.assertEqual(c["backward_steps"], 1)
            self.assertEqual(c["adjacent_duplicate_timestamps"], 1)
            self.assertEqual(c["no_final_newline"], 1)
            self.assertEqual(c["UTF8_BOM"], 1)
            self.assertEqual(c["characters"], sum(len(s) for s in raw.decode("utf-8-sig").splitlines()))
            self.assertEqual(result["sample"].iloc[0].message, "first\ncontinuation")
            self.assertEqual(result["sample"].line_no.tolist(), [1, 3, 4, 5])
            windows = window_counts(result, "1min")
            self.assertEqual(int(windows.sum()), 4)
            self.assertEqual(int(windows.eq(0).sum()), 3)

    def test_apache_orphan_and_unterminated_continuation_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Apache.log"
            path.write_text("orphan\n[Thu Jun 09 06:07:04 2005] [notice] a\ncontinuation", encoding="utf-8")
            (Path(tmp) / "Apache.tar.gz").touch()
            result = audit_dataset("Apache", raw_dir=tmp, sample_size=10, use_cache=False)
            self.assertEqual(result["counts"]["orphan_lines"], 1)
            self.assertEqual(result["counts"]["records"], 1)
            self.assertEqual(result["sample"].iloc[0].message, "a\ncontinuation")
            self.assertEqual(result["counts"]["no_final_newline"], 1)

    def test_entity_ports_do_not_match_timestamps(self):
        result = entities(["at 2016/9/27:20:30:31.455 use 192.168.1.1:443 port=80 port=99999"])
        self.assertEqual(result["port_explicit"], Counter({"443": 1, "80": 1}))
        self.assertEqual(result["IPv4"], Counter({"192.168.1.1": 1}))

    def test_session_id_fields_not_noun_phrases(self):
        result = entities(["Session package changes", "Session mutexes ready", "session id: 142859",
                           "Session: 123_456 initialized", "correlation_id=ab12"])
        self.assertEqual(result["session_candidate"], Counter({"142859": 1, "123_456": 1, "ab12": 1}))

    def test_full_bgl_duplicates_require_identical_records_not_just_times(self):
        first = b"- 1117838570 2005.06.03 R02 2005-06-03-15.42.50.363779 R02 RAS KERNEL INFO one\n"
        second = first.replace(b" one", b" two")
        later = first.replace(b".363779", b".363780")
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir()
            path = raw_dir / "BGL.log"
            path.write_bytes(first + second + first + later)
            audit = {"name": "BGL", "path": str(path), "sha256": "fixture", "counts": {"backward_steps": 0}}
            self.assertEqual(exact_bgl_duplicates(audit), 1)
            self.assertEqual(exact_bgl_duplicates(audit), 1)
            audit["counts"]["backward_steps"] = 1
            with self.assertRaises(ValueError):
                exact_bgl_duplicates(audit)

    def test_trailing_blank_continuations_preserved_and_raw_recovery_agrees(self):
        raw = (b"2016-09-28 04:30:30, Info CBS first\r\n\r\n\r\n"
               b"2016-09-28 04:30:31, Info CBS second\r\nlast continuation")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Windows.log"
            path.write_bytes(raw)
            (Path(tmp) / "Windows.tar.gz").touch()
            for block_size in (8, 10000):
                result = _audit_windows_blocks(path, 100, 42, block_size=block_size)
                self.assertEqual(result["sample"].message.tolist(), ["first\n\n", "second\nlast continuation"])
                recovered = reread_sample_records(result, block_size=5)
                self.assertEqual(recovered.message.tolist(), result["sample"].message.tolist())
                self.assertEqual(recovered.line_no.tolist(), result["sample"].line_no.tolist())
            path.write_bytes(raw + b"changed")
            with self.assertRaises(ValueError):
                reread_sample_records(result)


if __name__ == "__main__":
    unittest.main()
