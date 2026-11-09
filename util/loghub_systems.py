"""Verified LogHub sources and lossless physical-line parsing for four systems.

Thunderbird is scanned once in a bounded-memory pass: exact global aggregates,
a seeded uniform sample for text analysis, and a contiguous prefix for inspection.
No archive paths are extracted and no untrusted pickle caches are loaded.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "systems"
ZENODO = "https://zenodo.org/records/8196385/files"
CACHE_VERSION = 2
SCAN_BATCH_BYTES = 16 * 1024 * 1024
SOURCES = {
    "Thunderbird": {"archive": "Thunderbird.tar.gz", "md5": "0891b048df2919dc78c99c4428686b44", "bytes": 2016100298, "published_lines": 211212192},
    "OpenSSH": {"archive": "SSH.tar.gz", "md5": "a44e40b4348697dabf2bba56885e0a38", "bytes": 4596666, "published_lines": 655146},
    "Zookeeper": {"archive": "Zookeeper.tar.gz", "md5": "11458b4f9fa0911c238f5c1189d4f2d4", "bytes": 459911, "published_lines": 74380},
    "Mac": {"archive": "Mac.tar.gz", "md5": "e5d0558b6ec661c739e77c7ff1b71498", "bytes": 1489372, "published_lines": 117283},
}


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def ensure_archive(name: str, data_dir: Path | str = DATA_DIR) -> Path:
    """Download atomically and verify the checksum published by Zenodo.

    The original archives remain cached; interrupted .part files are never used.
    Existing archives are verified too, so a corrupt cache cannot silently pass.
    """
    meta = SOURCES[name]
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / meta["archive"]
    if path.exists():
        if path.stat().st_size == meta["bytes"] and file_digest(path, "md5") == meta["md5"]:
            return path
        raise ValueError(f"Corrupt archive: {path}. Remove it and retry.")
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(f"{ZENODO}/{meta['archive']}?download=1", headers={"User-Agent": "LogHub-EDA/1.0"})
    from tqdm.auto import tqdm

    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        with tqdm(total=meta["bytes"], unit="B", unit_scale=True, desc=name, mininterval=10) as progress:
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
                progress.update(len(chunk))
    if temporary.stat().st_size != meta["bytes"] or file_digest(temporary, "md5") != meta["md5"]:
        raise ValueError(f"Checksum mismatch: {temporary}")
    temporary.replace(path)
    return path


@contextmanager
def log_stream(archive: Path):
    """Open the sole .log member as a binary stream without extracting paths."""
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.lower().endswith(".log"):
                handle = tar.extractfile(member)
                if handle is None:
                    raise ValueError(f"Cannot read {member.name}")
                with handle:
                    yield member, handle
                return
    raise ValueError(f"No .log member in {archive}")


SYSLOG = re.compile(r"^(?P<local_time>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<component>[^\[:]+?)(?:\[(?P<pid>\d+)\])?(?:\s*\((?P<process_context>[^)]+)\))?:\s?(?P<message>.*)$")
REPEAT = re.compile(r"^(?P<local_time>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?P<message>--- last message repeated (?P<repeated>\d+) times? ---)$")
THUNDERBIRD = re.compile(r"^(?P<label>\S+)\s+(?P<epoch>\d+)\s+(?P<date>\d{4}\.\d{2}\.\d{2})\s+(?P<host>\S+)\s+(?P<local_time>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?P<origin>\S+)\s+(?P<body>.*)$")
TB_BODY = re.compile(r"^(?P<component>[^\s\[:]+)(?:\[(?P<pid>\d+)\])?:\s?(?P<message>.*)$")
ZOOKEEPER = re.compile(r"^(?P<local_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+-\s+(?P<level>\w+)\s+\[(?P<context>.*)\]\s+-\s?(?P<message>.*)$")
HEADER_START = {
    "Thunderbird": re.compile(r"^\S+ \d{9,11} \d{4}\.\d{2}\.\d{2} "),
    "OpenSSH": re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2} \d{2}:\d{2}:\d{2} "),
    "Mac": re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2} \d{2}:\d{2}:\d{2} "),
    "Zookeeper": re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} "),
}
MONTHS = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def parse_lines(name: str, lines: list[tuple[int, str]], synthetic_year: int = 2000) -> pd.DataFrame:
    """Retain every physical line, including empty and unparsed continuations.

    Syslog has no year/zone. Use an explicitly synthetic year and detect only
    Dec -> Jan rollovers in file order. Do not pretend local wall time is UTC.
    """
    rows = []
    year = synthetic_year
    previous_month = None
    pattern = THUNDERBIRD if name == "Thunderbird" else ZOOKEEPER if name == "Zookeeper" else SYSLOG
    for line_no, raw in lines:
        match = pattern.match(raw)
        if match is None and name == "Mac":
            match = REPEAT.match(raw)
        row = {"line_no": line_no, "raw": raw, "parse_ok": match is not None,
               "header_start": bool(HEADER_START[name].match(raw)), "message": raw,
               "host": None, "component": None, "level": None, "pid": None,
               "label": None, "local_time": None, "time_text": None}
        if match:
            row.update(match.groupdict())
            if name == "Mac" and "repeated" in row:
                row["component"] = "syslog_repeat"
            if name == "Thunderbird":
                body = TB_BODY.match(row.pop("body"))
                if body:
                    row.update(body.groupdict())
                else:
                    row["message"] = match["body"]
                row["time_text"] = row["epoch"]
            elif name == "Zookeeper":
                context = row["context"]
                row["component"] = context.rsplit(":", 1)[-1].split("@", 1)[0]
                row["time_text"] = row["local_time"]
            else:
                month = MONTHS[row["local_time"][:3]]
                if previous_month == 12 and month == 1:
                    year += 1
                previous_month = month
                day, clock = row["local_time"][3:].split()
                row["time_text"] = f"{year}-{month:02d}-{int(day):02d} {clock}"
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No lines in {name}")
    if name == "Thunderbird":
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["time_text"], errors="coerce"), unit="s", utc=True, errors="coerce")
        df["is_anomaly"] = df["label"].ne("-").where(df["parse_ok"]).astype("boolean")
    else:
        fmt = "%Y-%m-%d %H:%M:%S,%f" if name == "Zookeeper" else "%Y-%m-%d %H:%M:%S"
        df["timestamp"] = pd.to_datetime(df["time_text"], format=fmt, errors="coerce")
        df["is_anomaly"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df["line_length"] = df["raw"].str.len()
    df["message_length"] = df["message"].str.len()
    return df


def load_small(name: str, data_dir: Path | str = DATA_DIR) -> tuple[pd.DataFrame, dict]:
    """Load all physical lines of OpenSSH, ZooKeeper or Mac via a checked source."""
    if name == "Thunderbird":
        raise ValueError("Use load_thunderbird for bounded-memory analysis")
    archive = ensure_archive(name, data_dir)
    lines = []
    counts = Counter()
    digest = hashlib.sha256()
    with log_stream(archive) as (member, stream):
        for line_no, binary in enumerate(stream, 1):
            digest.update(binary)
            counts["bytes"] += len(binary)
            counts["crlf"] += binary.endswith(b"\r\n")
            counts["lf"] += binary.endswith(b"\n") and not binary.endswith(b"\r\n")
            counts["no_final_lf"] += not binary.endswith(b"\n")
            try:
                raw = binary.decode("utf-8")
            except UnicodeDecodeError:
                counts["invalid_utf8_lines"] += 1
                raw = binary.decode("utf-8", errors="replace")
            lines.append((line_no, raw.rstrip("\r\n")))
        meta = {"name": name, "member": member.name, "member_bytes": member.size,
                "physical_lines": len(lines), "sha256_raw": digest.hexdigest(), **dict(counts)}
    meta.update({"scope": "full", "url": f"{ZENODO}/{SOURCES[name]['archive']}?download=1", "archive_md5": SOURCES[name]["md5"]})
    return parse_lines(name, lines), meta


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_thunderbird(data_dir: Path | str = DATA_DIR, sample_size: int = 100_000,
                     prefix_size: int = 50_000, seed: int = 42) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Exact global scan plus a simple random sample without replacement.

    Published line count defines the population. It is checked after the scan;
    a mismatch aborts instead of silently biasing inclusion probabilities.
    Text statistics must use sample_kind == 'random'; the contiguous prefix
    and alert examples are separate views, never mixed into prevalence estimates.
    Cached results are bound to source MD5, parameters and scanner version.
    """
    source = SOURCES["Thunderbird"]
    population = source["published_lines"]
    if not 1 <= sample_size <= population or not 0 <= prefix_size <= population:
        raise ValueError("Invalid sample/prefix size")
    archive = ensure_archive("Thunderbird", data_dir)
    key = f"thunderbird-v{CACHE_VERSION}-{seed}-{sample_size}-{prefix_size}"
    directory = Path(data_dir)
    meta_path = directory / f"{key}.json"
    sample_path = directory / f"{key}.jsonl.gz"
    minutes_path = directory / f"{key}-minutes.csv.gz"
    if meta_path.exists() and sample_path.exists() and minutes_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta["archive_md5"] == source["md5"] and
                meta["sample_sha256"] == file_digest(sample_path) and
                meta["minutes_sha256"] == file_digest(minutes_path)):
            with gzip.open(sample_path, "rt", encoding="utf-8") as handle:
                selected = [json.loads(line) for line in handle]
            df = parse_lines("Thunderbird", [(r[0], r[1]) for r in selected])
            df["sample_kind"] = [r[2] for r in selected]
            minutes = pd.read_csv(minutes_path)
            minutes["timestamp"] = pd.to_datetime(minutes["timestamp"], utc=True)
            return df, meta, minutes.set_index("timestamp")
        raise ValueError("Thunderbird cache checksum mismatch; remove the three cache files and rerun")
    positions = np.sort(np.random.default_rng(seed).choice(population, size=sample_size, replace=False) + 1)
    target_index = 0
    selected = []
    minutes, alert_minutes, labels, hosts = Counter(), Counter(), Counter(), Counter()
    counts = Counter()
    digest = hashlib.sha256()
    first = last = previous = minimum = maximum = None
    alert_examples = 0
    longest_alert_run = current_alert_run = 0
    started = time.monotonic()
    line_no = 0
    next_progress = 5_000_000
    with log_stream(archive) as (member, stream):
        while binaries := stream.readlines(SCAN_BATCH_BYTES):
            payload = b"".join(binaries)
            digest.update(payload)
            counts["bytes"] += len(payload)
            crlf = payload.count(b"\r\n")
            counts["crlf"] += crlf
            counts["lf"] += payload.count(b"\n") - crlf
            counts["no_final_lf"] += not binaries[-1].endswith(b"\n")
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError:
                for binary in binaries:
                    try:
                        binary.decode("utf-8")
                    except UnicodeDecodeError:
                        counts["invalid_utf8_lines"] += 1
            parts = [binary.split(None, 4) for binary in binaries]
            valid = np.fromiter((len(part) == 5 and part[1].isdigit() for part in parts), dtype=bool, count=len(parts))
            counts["invalid_epoch_header"] += int((~valid).sum())
            counts["blank_lines"] += sum(not binaries[i].strip() for i in np.flatnonzero(~valid))
            good = [part for part, ok in zip(parts, valid) if ok]
            alert = np.zeros(len(parts), dtype=bool)
            alert[valid] = [part[0] != b"-" for part in good]
            if good:
                epochs = np.fromiter((int(part[1]) for part in good), dtype=np.int64, count=len(good))
                minute_keys, frequencies = np.unique(epochs // 60, return_counts=True)
                minutes.update(dict(zip(minute_keys.tolist(), frequencies.tolist())))
                minute_keys, frequencies = np.unique(epochs[alert[valid]] // 60, return_counts=True)
                alert_minutes.update(dict(zip(minute_keys.tolist(), frequencies.tolist())))
                labels.update(part[0] for part in good)
                hosts.update(part[3] for part in good)
                if first is None:
                    first = minimum = maximum = int(epochs[0])
                minimum = min(minimum, int(epochs.min()))
                maximum = max(maximum, int(epochs.max()))
                differences = np.diff(epochs)
                counts["timestamp_backsteps"] += int((differences < 0).sum()) + int(previous is not None and epochs[0] < previous)
                counts["adjacent_equal_timestamps"] += int((differences == 0).sum()) + int(previous == epochs[0])
                previous = last = int(epochs[-1])
            transitions = np.diff(np.r_[False, alert, False].astype(np.int8))
            starts = np.flatnonzero(transitions == 1)
            ends = np.flatnonzero(transitions == -1)
            if len(starts):
                lengths = ends - starts
                if starts[0] == 0:
                    lengths[0] += current_alert_run
                longest_alert_run = max(longest_alert_run, int(lengths.max()))
                current_alert_run = int(lengths[-1]) if ends[-1] == len(alert) else 0
            else:
                current_alert_run = 0
            block_end = line_no + len(binaries)
            target_end = int(np.searchsorted(positions, block_end, side="right"))
            for position in positions[target_index:target_end]:
                raw = binaries[int(position) - line_no - 1].decode("utf-8", errors="replace").rstrip("\r\n")
                selected.append((int(position), raw, "random"))
            target_index = target_end
            for index in range(max(0, min(prefix_size - line_no, len(binaries)))):
                selected.append((line_no + index + 1, binaries[index].decode("utf-8", errors="replace").rstrip("\r\n"), "prefix"))
            for index in np.flatnonzero(alert)[:max(0, 30 - alert_examples)]:
                selected.append((line_no + int(index) + 1, binaries[index].decode("utf-8", errors="replace").rstrip("\r\n"), "alert_example"))
                alert_examples += 1
            line_no = block_end
            if line_no >= next_progress:
                print(f"Thunderbird: {line_no:,}/{population:,} lines, {time.monotonic() - started:.0f}s", flush=True)
                next_progress += 5_000_000
        member_name, member_size = member.name, member.size
    if line_no != population or target_index != sample_size:
        raise ValueError(f"Population changed: expected {population}, observed {line_no}")
    selected.sort(key=lambda r: r[0])
    frame = pd.DataFrame({"minute": sorted(minutes)})
    frame["count"] = frame["minute"].map(minutes)
    frame["alerts"] = frame["minute"].map(alert_minutes)
    frame["timestamp"] = pd.to_datetime(frame.pop("minute") * 60, unit="s", utc=True)
    frame = frame.set_index("timestamp")
    with gzip.open(sample_path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    frame.to_csv(minutes_path, compression="gzip")
    meta = {"name": "Thunderbird", "scope": "full aggregates + uniform text sample",
            "url": f"{ZENODO}/{source['archive']}?download=1", "archive_md5": source["md5"],
            "physical_lines": line_no, "member": member_name, "member_bytes": member_size,
            "sha256_raw": digest.hexdigest(), "seed": seed, "sample_size": sample_size,
            "prefix_size": prefix_size, "labels": {k.decode("ascii", errors="replace"): v for k, v in labels.items()},
            "hosts": {k.decode("utf-8", errors="replace"): v for k, v in hosts.items()},
            "first_epoch": first, "last_epoch": last, "min_epoch": minimum, "max_epoch": maximum,
            "longest_alert_run_file_order": longest_alert_run, **dict(counts),
            "sample_sha256": file_digest(sample_path), "minutes_sha256": file_digest(minutes_path)}
    _write_json(meta_path, meta)
    df = parse_lines("Thunderbird", [(r[0], r[1]) for r in selected])
    df["sample_kind"] = [r[2] for r in selected]
    return df, meta, frame
