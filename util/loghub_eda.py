"""Streaming raw-file audit and bounded-memory EDA for LogHub.

Full-file counters are kept separately from uniformly sampled logical records.
Caches are local, disposable products under data/processed (not source data).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import math
from pathlib import Path
import pickle
import random
import re
import json

import pandas as pd

from util.loghub import DATASETS, ensure_dataset

VERSION = {"BGL": 2, "Windows": 2, "HealthApp": 1, "Apache": 1}
FORMATS = {
    "BGL": "%Y-%m-%d-%H.%M.%S.%f",
    "Windows": "%Y-%m-%d %H:%M:%S",
    "HealthApp": "%Y%m%d-%H:%M:%S:%f",
    "Apache": "%a %b %d %H:%M:%S %Y",
}
HEADERS = {
    "BGL": re.compile(r"^(\S+) (\d+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)(?:\s(.*))?$"),
    "Windows": re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\s*(\S+)\s+(\S+)(?:\s+(.*))?$"),
    "HealthApp": re.compile(r"^(\d{6,8}-\d{1,2}:\d{1,2}:\d{1,2}:\d+)\|([^|]*)\|([^|]*)\|(.*)$"),
    "Apache": re.compile(r"^\[([^]]+)\] \[([^]]+)\](?: (.*))?$"),
}


class Reservoir:
    """Vitter algorithm L: uniform sample without an RNG call for every row."""
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError("sample size must be positive")
        self.size, self.rng, self.items = size, random.Random(seed), []
        self.seen, self.next, self.weight = 0, size, None

    def add(self, value):
        i = self.seen
        self.seen += 1
        if i < self.size:
            self.items.append(value)
            if i == self.size - 1:
                self.weight = math.exp(math.log(self.rng.random()) / self.size)
                self._advance(i)
        elif i == self.next:
            self.items[self.rng.randrange(self.size)] = value
            self.weight *= math.exp(math.log(self.rng.random()) / self.size)
            self._advance(i)

    def _advance(self, index):
        self.next = index + 1 + math.floor(math.log(self.rng.random()) / math.log1p(-self.weight))

    def batch_indices(self, count):
        """Yield selected global indices; caller must add each yielded value."""
        stop = self.seen + count
        while self.seen < stop:
            index = self.seen if self.seen < self.size else self.next
            if index >= stop:
                self.seen = stop
                break
            self.seen = index
            yield index


def parse_line(name, text):
    """Parse headers, preserving empty messages and both BGL node fields."""
    m = HEADERS[name].match(text.lstrip("\x00"))
    if not m:
        return None
    g = m.groups()
    if name == "BGL":
        return dict(label=g[0], unix_ts=int(g[1]), date=g[2], host=g[3],
                    raw_timestamp=g[4], node_repeat=g[5], component=g[6] + "/" + g[7],
                    level=g[8], message=g[9] or "")
    if name == "Windows":
        return dict(raw_timestamp=g[0], level=g[1], component=g[2], message=g[3] or "")
    if name == "HealthApp":
        return dict(raw_timestamp=g[0], component=g[1], thread_id=g[2], message=g[3])
    return dict(raw_timestamp=g[0], level=g[1], message=g[2] or "")


def parse_time(name, value):
    if name == "BGL":
        return datetime.fromisoformat(value[:10] + "T" + value[11:19].replace(".", ":") + value[19:])
    if name == "Windows":
        return datetime.fromisoformat(value)
    if name == "HealthApp":
        day, clock = value.split("-", 1)
        # This file uses yyyyMd (201811 = 2018-01-01), not fixed yyyyMMdd.
        candidates = []
        for split in (5, 6):
            try:
                candidates.append(datetime(int(day[:4]), int(day[4:split]), int(day[split:])))
            except ValueError:
                pass
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous HealthApp calendar date: {day}")
        h, m, s, ms = map(int, clock.split(":"))
        return candidates[0].replace(hour=h, minute=m, second=s, microsecond=ms * 1000)
    return datetime.strptime(value, FORMATS[name])


def audit_dataset(name, sample_size=100_000, seed=42, raw_dir=None, use_cache=True):
    """Audit every physical line; sample complete header-delimited records.

    Headerless lines are retained in the preceding record with a quality flag.
    This is a grouping convention, not proof that they are continuations.
    Full duplicate detection is intentionally not claimed: see sample outputs.
    """
    path = ensure_dataset(name, raw_dir)
    stat = path.stat()
    signature = (VERSION[name], name, stat.st_size, stat.st_mtime_ns, sample_size, seed)
    cache_dir = path.parent.parent / "processed"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"{name}-eda-{sample_size}-{seed}.pkl"
    if use_cache and cache.exists():
        with cache.open("rb") as handle:
            saved = pickle.load(handle)
        if saved[0] == signature:
            return saved[1]
    if name == "Windows":
        result = _audit_windows_blocks(path, sample_size, seed)
        with cache.open("wb") as handle:
            pickle.dump((signature, result), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return result
    records, raw_sample = Reservoir(sample_size, seed), Reservoir(100, seed + 1)
    counts = Counter()
    minutes, levels, components, hosts, labels = (Counter() for _ in range(5))
    monthly = {field: Counter() for field in ("level", "component", "host")}
    anomaly_minutes, anomaly_hosts, anomaly_levels = Counter(), Counter(), Counter()
    length_by_label = {False: Counter(), True: Counter()}
    record_length_by_label = {False: Counter(), True: Counter()}
    first, unusual, anomalies = [], [], []
    current, previous_time, first_time, last_time = None, None, None, None
    min_time, max_time = None, None
    digest = hashlib.sha256()

    def finish(row):
        if row is None:
            return
        row["message"] = "\n".join(row.pop("parts"))
        if "label" in row:
            record_length_by_label[row["label"] != "-"][row["record_chars"]] += 1
        records.add(row)
        if row.get("label", "-") != "-" and len(anomalies) < 10:
            anomalies.append(row.copy())

    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            digest.update(raw)
            counts["physical_lines"] += 1
            counts["bytes_with_terminator"] += len(raw)
            counts["CRLF" if raw.endswith(b"\r\n") else "LF" if raw.endswith(b"\n") else "no_final_newline"] += 1
            if line_no == 1:
                counts["UTF8_BOM"] = int(raw.startswith(b"\xef\xbb\xbf"))
            try:
                text = raw.decode("utf-8-sig" if line_no == 1 else "utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                counts["invalid_utf8_lines"] += 1
                text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            counts["characters"] += len(text)
            counts["empty_lines"] += int(not text.strip())
            counts["replacement_char_lines"] += int("\ufffd" in text)
            counts["NUL_lines"] += int("\x00" in text)
            if len(first) < 100:
                first.append((line_no, text))
            raw_sample.add((line_no, text))
            row = parse_line(name, text)
            if row is None:
                counts["headerless_lines"] += 1
                if len(unusual) < 20:
                    unusual.append((line_no, text))
                if current is None:
                    counts["orphan_lines"] += 1
                else:
                    current["parts"].append(text)
                    current["physical_line_count"] += 1
                    current["record_chars"] += len(text)
                continue
            finish(current)
            current = row
            current.update(line_no=line_no, physical_line_count=1, parts=[row.pop("message")],
                           header_line_chars=len(text), record_chars=len(text))
            counts["records"] += 1
            try:
                ts = parse_time(name, row["raw_timestamp"])
            except (ValueError, OverflowError):
                counts["invalid_timestamp_records"] += 1
                continue
            row["normalized_timestamp"] = ts.isoformat()
            first_time = first_time or ts
            last_time = ts
            min_time = ts if min_time is None else min(min_time, ts)
            max_time = ts if max_time is None else max(max_time, ts)
            if previous_time is not None:
                counts["backward_steps"] += int(ts < previous_time)
                counts["adjacent_duplicate_timestamps"] += int(ts == previous_time)
            previous_time = ts
            minute = ts.isoformat(timespec="minutes")
            minutes[minute] += 1
            month = minute[:7]
            for field, counter in (("level", levels), ("component", components), ("host", hosts)):
                if field in row:
                    counter[row[field]] += 1
                    monthly[field][(month, row[field])] += 1
            if "label" in row:
                labels[row["label"]] += 1
                anomaly = row["label"] != "-"
                length_by_label[anomaly][len(current["parts"][0])] += 1
                if anomaly:
                    anomaly_minutes[minute] += 1
                    anomaly_hosts[row["host"]] += 1
                    anomaly_levels[row["level"]] += 1
            if line_no % 10_000_000 == 0:
                print(f"{name}: {line_no:,} physical lines", flush=True)
    finish(current)
    sample = pd.DataFrame(sorted(records.items, key=lambda r: r["line_no"]))
    if "normalized_timestamp" in sample:
        sample["timestamp"] = pd.to_datetime(sample["normalized_timestamp"], format="ISO8601", errors="coerce")
    else:
        sample["timestamp"] = pd.to_datetime(sample["raw_timestamp"], format=FORMATS[name], errors="coerce")
    if "label" in sample:
        sample["is_anomaly"] = sample["label"].ne("-")
    result = dict(name=name, path=str(path), size_bytes=stat.st_size,
                  archive_bytes=(path.parent / DATASETS[name]["archive"]).stat().st_size,
                  sha256=digest.hexdigest(), counts=dict(counts), first=first,
                  raw_sample=sorted(raw_sample.items), unusual=unusual, sample=sample,
                  minutes=minutes, levels=levels, components=components, hosts=hosts,
                  labels=labels, monthly=monthly, anomaly_minutes=anomaly_minutes,
                  anomaly_hosts=anomaly_hosts, anomaly_levels=anomaly_levels,
                  length_by_label=length_by_label, anomalies=anomalies,
                  record_length_by_label=record_length_by_label,
                  first_time=first_time, last_time=last_time, min_time=min_time, max_time=max_time)
    with cache.open("wb") as handle:
        pickle.dump((signature, result), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def _audit_windows_blocks(path, sample_size, seed, block_size=32 << 20):
    """Vectorized header census; decode/parse individual records only if sampled.

    Blocks end at physical line boundaries. A sampled final record stays mutable
    until the next block closes it, so multiline records are never cut at blocks.
    """
    import numpy as np
    header = re.compile(rb"(?m)^(?:\xef\xbb\xbf)?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),[ \t]*(\S+)[ \t]+(\S+)(?:[ \t]+[^\r\n]*)?\r?$")
    records, raw_sample = Reservoir(sample_size, seed), Reservoir(100, seed + 1)
    counts, minutes, levels, components = (Counter() for _ in range(4))
    monthly = {field: Counter() for field in ("level", "component", "host")}
    first, unusual, pending = [], [], None
    digest = hashlib.sha256()
    first_time = last_time = min_time = max_time = previous = None

    def continuation_text(raw):
        parts = raw.decode("utf-8", errors="replace").split("\n")
        if raw.endswith(b"\n"):
            parts.pop()
        return "\n".join(part.rstrip("\r") for part in parts)

    def extend_pending(prefix):
        if pending is not None and prefix:
            pending["message"] += "\n" + continuation_text(prefix)
            pending["physical_line_count"] += len(prefix.splitlines())

    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            if not block.endswith(b"\n"):
                block += handle.readline()
            digest.update(block)
            lines = block.split(b"\n")
            if lines[-1] == b"":
                lines.pop()
            base_line, base_record = counts["physical_lines"], counts["records"]
            counts["physical_lines"] += len(lines)
            counts["bytes_with_terminator"] += len(block)
            crlf, lf = block.count(b"\r\n"), block.count(b"\n")
            counts["CRLF"] += crlf
            counts["LF"] += lf - crlf
            counts["no_final_newline"] += int(not block.endswith(b"\n"))
            if base_line == 0:
                counts["UTF8_BOM"] = int(block.startswith(b"\xef\xbb\xbf"))
            try:
                decoded = block.decode("utf-8-sig" if base_line == 0 else "utf-8")
            except UnicodeDecodeError:
                decoded = block.decode("utf-8", errors="replace")
                for raw in lines:
                    try:
                        raw.decode("utf-8")
                    except UnicodeDecodeError:
                        counts["invalid_utf8_lines"] += 1
            counts["characters"] += len(decoded) - lf - crlf
            counts["empty_lines"] += sum(not raw.strip() for raw in lines)
            if "\ufffd" in decoded:
                counts["replacement_char_lines"] += sum("\ufffd" in raw.decode("utf-8", errors="replace") for raw in lines)
            if b"\x00" in block:
                counts["NUL_lines"] += sum(b"\x00" in raw for raw in lines)
            for local, raw in enumerate(lines[:max(0, 100 - len(first))]):
                first.append((base_line + local + 1, raw.decode("utf-8-sig" if base_line + local == 0 else "utf-8", errors="replace").rstrip("\r\n")))
            for index in raw_sample.batch_indices(len(lines)):
                raw_sample.add((index + 1, lines[index - base_line].decode("utf-8-sig" if index == 0 else "utf-8", errors="replace").rstrip("\r\n")))
            matches = list(header.finditer(block))
            counts["records"] += len(matches)
            counts["headerless_lines"] += len(lines) - len(matches)
            if len(unusual) < 20 and len(lines) != len(matches):
                for i, raw in enumerate(lines):
                    if parse_line("Windows", raw.decode("utf-8-sig", errors="replace").rstrip("\r\n")) is None:
                        unusual.append((base_line + i + 1, raw.decode("utf-8", errors="replace").rstrip("\r\n")))
                        if len(unusual) == 20:
                            break
            if not matches:
                extend_pending(block)
                if base_record == 0:
                    counts["orphan_lines"] += len(lines)
                continue
            prefix = block[:matches[0].start()]
            extend_pending(prefix)
            if base_record == 0:
                counts["orphan_lines"] += len(prefix.splitlines())
            pending = None
            # High repetition within servicing bursts makes this census much smaller than the stream.
            census = Counter((m[1], m[2], m[3]) for m in matches)
            valid_times = {}
            for (raw_ts, raw_level, raw_component), number in census.items():
                timestamp = raw_ts.decode("ascii")
                try:
                    ts = datetime.fromisoformat(timestamp)
                except ValueError:
                    counts["invalid_timestamp_records"] += number
                    valid_times[raw_ts] = False
                    continue
                valid_times[raw_ts] = True
                min_time = ts if min_time is None else min(min_time, ts)
                max_time = ts if max_time is None else max(max_time, ts)
                minute, month = timestamp[:16].replace(" ", "T"), timestamp[:7]
                level, component = raw_level.decode("utf-8", errors="replace"), raw_component.decode("utf-8", errors="replace")
                minutes[minute] += number
                levels[level] += number
                components[component] += number
                monthly["level"][(month, level)] += number
                monthly["component"][(month, component)] += number
            ordered = [m[1] for m in matches if valid_times[m[1]]]
            if ordered:
                first_time = first_time or datetime.fromisoformat(ordered[0].decode())
                last_time = datetime.fromisoformat(ordered[-1].decode())
                array = np.asarray(([previous] if previous else []) + ordered, dtype="S19")
                counts["backward_steps"] += int(np.count_nonzero(array[1:] < array[:-1]))
                counts["adjacent_duplicate_timestamps"] += int(np.count_nonzero(array[1:] == array[:-1]))
                previous = ordered[-1]
            # batch_indices advances only as each selected value is added, including replacements.
            offsets = None
            for index in records.batch_indices(len(matches)):
                local = index - base_record
                start = matches[local].start()
                end = matches[local + 1].start() if local + 1 < len(matches) else len(block)
                if offsets is None:
                    offsets = np.cumsum([len(raw) + 1 for raw in lines])
                line_no = base_line + int(np.searchsorted(offsets, start, side="right")) + 1
                raw_record = block[start:end]
                header_line, separator, continuation = raw_record.partition(b"\n")
                row = parse_line("Windows", header_line.decode("utf-8-sig", errors="replace").rstrip("\r"))
                row.update(line_no=line_no, physical_line_count=raw_record.count(b"\n") + int(not raw_record.endswith(b"\n")))
                if continuation:
                    row["message"] += "\n" + continuation_text(continuation)
                records.add(row)
                if local + 1 == len(matches):
                    pending = row
            if counts["physical_lines"] // 10_000_000 != base_line // 10_000_000:
                print(f"Windows: {counts['physical_lines']:,} physical lines (block audit)", flush=True)
    sample = pd.DataFrame(sorted(records.items, key=lambda r: r["line_no"]))
    sample["timestamp"] = pd.to_datetime(sample["raw_timestamp"], format=FORMATS["Windows"], errors="coerce")
    return dict(name="Windows", path=str(path), size_bytes=path.stat().st_size,
                archive_bytes=(path.parent / DATASETS["Windows"]["archive"]).stat().st_size,
                sha256=digest.hexdigest(), counts=dict(counts), first=first, raw_sample=sorted(raw_sample.items),
                unusual=unusual, sample=sample, minutes=minutes, levels=levels, components=components, hosts=Counter(),
                labels=Counter(), monthly=monthly, anomaly_minutes=Counter(), anomaly_hosts=Counter(), anomaly_levels=Counter(),
                length_by_label={False: Counter(), True: Counter()}, anomalies=[], first_time=first_time, last_time=last_time,
                min_time=min_time, max_time=max_time)


def reread_sample_records(audit, block_size=32 << 20):
    """Recover sampled records verbatim from physical line numbers, in one pass.

    Useful for checking the parser against raw records without retaining all data.
    Source SHA256 must match the audit, including any orphan/unsampled text.
    """
    path, name = Path(audit["path"]), audit["name"]
    requested = list(audit["sample"][["line_no", "physical_line_count"]].itertuples(index=False, name=None))
    rows, index, base, pending = [], 0, 0, None
    digest = hashlib.sha256()

    def finish(parts, request):
        line_no, physical_count = request
        assert len(parts) == physical_count
        texts = [raw.decode("utf-8-sig" if i == 0 else "utf-8", errors="replace").rstrip("\r") for i, raw in enumerate(parts)]
        row = parse_line(name, texts[0])
        if row is None:
            raise ValueError(f"Missing sampled header at line {line_no}")
        row["message"] = "\n".join([row["message"], *texts[1:]])
        row.update(line_no=line_no, physical_line_count=physical_count)
        rows.append(row)

    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            if not block.endswith(b"\n"):
                block += handle.readline()
            digest.update(block)
            lines = block.split(b"\n")
            if lines[-1] == b"":
                lines.pop()
            end = base + len(lines)
            if pending is not None:
                parts, request = pending
                need = request[1] - len(parts)
                parts.extend(lines[:need])
                if len(parts) == request[1]:
                    finish(parts, request)
                    pending = None
            while index < len(requested) and requested[index][0] <= end:
                request = requested[index]
                start, count = request[0] - 1 - base, request[1]
                if start < 0 or pending is not None:
                    raise ValueError("Sampled records overlap or are out of order")
                parts = lines[start:start + count]
                if len(parts) == count:
                    finish(parts, request)
                else:
                    pending = (parts, request)
                index += 1
            base = end
    if digest.hexdigest() != audit["sha256"] or len(rows) != len(requested) or pending is not None:
        raise ValueError("Raw file differs from the audited source or has incomplete sampled records")
    result = pd.DataFrame(rows)
    timestamps = []
    for value in result["raw_timestamp"]:
        try:
            timestamps.append(parse_time(name, value))
        except (ValueError, OverflowError):
            timestamps.append(pd.NaT)
    result["timestamp"] = pd.to_datetime(timestamps)
    if "label" in result:
        result["is_anomaly"] = result["label"].ne("-")
    return result


def minute_series(audit, anomaly=False):
    counts = audit["anomaly_minutes" if anomaly else "minutes"]
    result = pd.Series(counts, dtype="int64")
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def exact_bgl_duplicates(audit):
    """Exact full-line duplicates in BGL with bounded memory.

    Identical records must have identical timestamp tokens. Verified monotonicity
    means all equal tokens are adjacent, so only one timestamp group is retained.
    A differing byte order or backward timestamp invalidates this shortcut.
    """
    if audit["name"] != "BGL" or audit["counts"].get("backward_steps", 0):
        raise ValueError("This exact duplicate pass requires monotonic BGL records")
    path = Path(audit["path"])
    cache = path.parent.parent / "processed/BGL-exact-duplicates.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["raw_sha256"] == audit["sha256"]:
            return saved["duplicate_excess"]
    previous, group, duplicates = None, set(), 0
    with path.open("rb") as handle:
        for raw in handle:
            timestamp = raw.split(None, 5)[4]
            if previous is not None and timestamp < previous:
                raise ValueError("Raw timestamp order is not monotonic")
            if timestamp != previous:
                group.clear()
            duplicates += int(raw in group)
            group.add(raw)
            previous = timestamp
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"raw_sha256": audit["sha256"], "duplicate_excess": duplicates}), encoding="utf-8")
    return duplicates


def window_counts(audit, rule="1h", anomaly=False):
    series = minute_series(audit, anomaly)
    start, end = pd.Timestamp(audit["min_time"]).floor(rule), pd.Timestamp(audit["max_time"]).floor(rule)
    index = pd.date_range(start, end, freq=rule)
    return series.resample(rule).sum().reindex(index, fill_value=0)


ENTITY_PATTERNS = {
    "IPv4": r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])",
    "IPv6_candidate": r"(?<!\w)(?:[0-9a-fA-F]{1,4}:){2,}[0-9a-fA-F:]+",
    "port_explicit": r"(?i)\bport[\s=:]+(\d{1,5})\b|(?:\d{1,3}\.){3}\d{1,3}:(\d{1,5})\b|\[[a-f0-9:]+\]:(\d{1,5})\b",
    "UUID": r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b",
    "device_id_candidate": r"\bR\d{2}-M\d-[^\s,;]+|\b(?:device|imei|android_id)[\s=:]+[\w.-]+",
    "path": r"[A-Za-z]:\\[^\s\"<>|]+|(?<![\w:])/(?:[\w.-]+/)*[\w.-]+",
    "URL": r"https?://[^\s<>\"]+",
    "hash_candidate": r"\b(?:[0-9a-fA-F]{64}|[0-9a-fA-F]{40}|[0-9a-fA-F]{32})\b",
    "session_candidate": r"(?i)\b(?:session(?:[ _]?id)?|correlation(?:[ _]?id)?|activity(?:[ _]?id)?)[\s:=]+([\w{}.-]+)",
}


def entities(messages):
    """Regex candidates; validate IPv4/IPv6 and ports before counting."""
    import ipaddress
    results = {}
    for kind, pattern in ENTITY_PATTERNS.items():
        counter = Counter()
        regex = re.compile(pattern)
        for message in messages:
            for match in regex.finditer(message):
                value = next((v for v in match.groups() if v), match.group(0))
                if kind in ("IPv4", "IPv6_candidate"):
                    try:
                        ipaddress.ip_address(value)
                    except ValueError:
                        continue
                if kind == "port_explicit" and not 0 <= int(value) <= 65535:
                    continue
                if kind == "session_candidate" and not any(char.isdigit() for char in value):
                    continue
                counter[value] += 1
        results[kind] = counter
    return results


MASKS = [
    (re.compile(ENTITY_PATTERNS[k]), f"<{k}>")
    for k in ("URL", "UUID", "path", "IPv4", "hash_candidate", "device_id_candidate")
] + [(re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
     (re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)*(?!\w)"), "<NUM>")]


def system_template(message):
    """Explicit format-aware normalization baseline, not Drain clusters.

    The order protects Windows paths/UUIDs from premature numeric masking.
    CSI counters are hexadecimal fields; HealthApp embeds times after colons.
    """
    value = re.sub(r"^\s*[0-9a-fA-F]{8}(?=@|\s)", "<CSI_ID>", message)
    for pattern, replacement in MASKS:
        value = pattern.sub(lambda m: replacement, value)
    return value


def template_lifetimes(df):
    table = df.groupby("template", observed=True).agg(
        count=("message", "size"), first_seen=("timestamp", "min"), last_seen=("timestamp", "max"))
    table["observed_span"] = table.last_seen - table.first_seen
    return table.sort_values("count", ascending=False)


def mine_templates(df):
    """Drain3 training in file order, with final templates for stable cluster IDs."""
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth = 4
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)
    ids = []
    normalized = df["message"].map(system_template)
    for value in normalized:
        ids.append(miner.add_log_message(value)["cluster_id"])
    final = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    result = df.copy(deep=False)
    result["template_id"] = ids
    result["template"] = result["template_id"].map(final)
    return result


def text_profile(messages):
    """Weighted word/bigram, character-class and whitespace-token counters."""
    words, bigrams, classes, token_classes = Counter(), Counter(), Counter(), Counter()
    for message, frequency in messages.value_counts(dropna=False).items():
        tokens = re.findall(r"[^\W\d_]+", message.lower(), flags=re.UNICODE)
        words.update({k: v * frequency for k, v in Counter(tokens).items()})
        bigrams.update({" ".join(k): v * frequency for k, v in Counter(zip(tokens, tokens[1:])).items()})
        local = Counter()
        for char in message:
            kind = "letter" if char.isalpha() else "digit" if char.isdigit() else "space" if char.isspace() else "punctuation_or_symbol"
            local[kind] += 1
        classes.update({k: v * frequency for k, v in local.items()})
        local_tokens = Counter()
        for token in message.split():
            kind = ("number" if token.isdecimal() else "letters" if token.isalpha()
                    else "punctuation" if all(not c.isalnum() and c != "_" for c in token) else "mixed")
            local_tokens[kind] += 1
        token_classes.update({k: v * frequency for k, v in local_tokens.items()})
    return words, bigrams, classes, token_classes


def categorical_drift(audit, field):
    """Monthly total-variation distance to preceding month, range [0, 1]."""
    counter = audit["monthly"][field]
    if not counter:
        return pd.DataFrame()
    rows = [(month, category, count) for (month, category), count in counter.items()]
    table = pd.DataFrame(rows, columns=["month", "category", "count"]).pivot(
        index="month", columns="category", values="count").fillna(0).sort_index()
    share = table.div(table.sum(axis=1), axis=0)
    return pd.DataFrame({"events": table.sum(axis=1),
                         "TV_to_previous_month": share.diff().abs().sum(axis=1).mask(share.index == share.index[0]),
                         "largest_category": share.idxmax(axis=1), "largest_share": share.max(axis=1)})


def gap_table(audit, top=10):
    """Empty minute bins between occupied minutes (not a claim about downtime)."""
    index = minute_series(audit).index
    gaps = pd.DataFrame({"previous_occupied_minute": index[:-1], "next_occupied_minute": index[1:]})
    gaps["empty_minutes"] = (gaps.next_occupied_minute - gaps.previous_occupied_minute).dt.total_seconds() / 60 - 1
    return gaps[gaps.empty_minutes > 0].nlargest(top, "empty_minutes")


def anomaly_episodes(audit, max_gap_minutes=5):
    """Operational alert bursts; a temporal grouping, not ground truth incidents."""
    series = minute_series(audit, anomaly=True)
    series = series[series > 0]
    if series.empty:
        return pd.DataFrame()
    frame = series.rename("alerts").rename_axis("minute").reset_index()
    groups = frame.minute.diff().gt(pd.Timedelta(minutes=max_gap_minutes)).cumsum()
    result = frame.groupby(groups).agg(start=("minute", "min"), end=("minute", "max"),
                                       alerts=("alerts", "sum"), occupied_minutes=("minute", "size"))
    result["span_minutes"] = (result.end - result.start).dt.total_seconds() / 60 + 1
    return result.sort_values("alerts", ascending=False)
