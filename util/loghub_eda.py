"""Reusable EDA calculations; notebooks keep the narrative and research choices."""

from collections import Counter
import ipaddress
import re

import numpy as np
import pandas as pd

from util.loghub_systems import SOURCES


def reconstruct_events(physical: pd.DataFrame) -> pd.DataFrame:
    """Join headerless consecutive lines only to the preceding valid header.

    Unknown header-shaped lines and leading orphans stay separate. This is an
    explicit heuristic for multiline logs, not a claim about original event IDs.
    Never call it on a discontinuous random sample.
    """
    if physical.parse_ok.all():
        result = physical.copy()
        result["physical_line_count"] = 1
        return result
    starts = physical.header_start | physical.parse_ok
    group = starts.cumsum()
    rows = []
    for _, part in physical.groupby(group, sort=False):
        first = part.iloc[0]
        if first.parse_ok:
            row = first.to_dict()
            row["message"] = "\n".join(part.message)
            row["raw"] = "\n".join(part.raw)
            row["physical_line_count"] = len(part)
            rows.append(row)
        else:
            for row in part.to_dict("records"):
                row["physical_line_count"] = 1
                rows.append(row)
    result = pd.DataFrame(rows)
    result["line_length"] = result.raw.str.len()
    result["message_length"] = result.message.str.len()
    return result


def quality_table(physical: pd.DataFrame, meta: dict, events: pd.DataFrame) -> pd.Series:
    """Physical metrics refer to the passed frame; global source facts are named."""
    return pd.Series({
        "source_physical_lines (full)": meta["physical_lines"],
        "published_lines (reference)": SOURCES[meta["name"]]["published_lines"],
        "source_bytes (full)": meta["bytes"],
        "analyzed_physical_lines": len(physical),
        "analyzed_events": len(events),
        "mean_line_characters": physical.line_length.mean(),
        "blank_lines": int(physical.raw.str.strip().eq("").sum()),
        "duplicate_raw_lines_beyond_first": int(physical.raw.duplicated().sum()),
        "unparsed_physical_lines": int((~physical.parse_ok).sum()),
        "without_header_start": int((~physical.header_start).sum()),
        "header_shaped_but_unparsed": int((physical.header_start & ~physical.parse_ok).sum()),
        "invalid_utf8_lines (full)": meta.get("invalid_utf8_lines", 0),
        "replacement_character_lines": int(physical.raw.str.contains("\ufffd", regex=False).sum()),
        "NUL_lines": int(physical.raw.str.contains("\x00", regex=False).sum()),
        "multiline_events": int(events.get("physical_line_count", pd.Series(1, index=events.index)).gt(1).sum()),
        "CRLF (full)": meta.get("crlf", 0),
        "LF (full)": meta.get("lf", 0),
        "line_without_final_LF (full)": meta.get("no_final_lf", 0),
    }, name="value")


def minute_counts(events: pd.DataFrame) -> pd.DataFrame:
    valid = events.loc[events.timestamp.notna()]
    result = valid.set_index("timestamp").resample("1min").size().to_frame("count")
    result["alerts"] = np.nan
    return result


def dense_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    return minutes.resample("1min").sum(min_count=1).reindex(
        pd.date_range(minutes.index.min(), minutes.index.max(), freq="1min"), fill_value=0
    ).assign(count=lambda d: d["count"].fillna(0))


def time_summary(events: pd.DataFrame) -> pd.Series:
    times = events.timestamp.dropna()
    delta = times.diff().dt.total_seconds()
    return pd.Series({"valid_timestamp_events": len(times), "missing_timestamps": events.timestamp.isna().sum(),
                      "first_in_file": times.iloc[0], "last_in_file": times.iloc[-1],
                      "chronological_min": times.min(), "chronological_max": times.max(),
                      "span": times.max() - times.min(), "backsteps_file_order": delta.lt(0).sum(),
                      "repeated_timestamps_beyond_first": times.duplicated().sum(),
                      "fractional_second_events": times.dt.microsecond.ne(0).sum()})


def largest_gaps(events: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    times = events.timestamp.dropna().drop_duplicates().sort_values().reset_index(drop=True)
    result = pd.DataFrame({"from": times.shift(), "to": times, "gap_seconds": times.diff().dt.total_seconds()})
    return result.nlargest(n, "gap_seconds")


def local_clock_offsets(events: pd.DataFrame) -> pd.Series:
    """Infer the nearest year for yearless Thunderbird local wall clocks.

    Comparing Dec 31 local time with a Jan 1 UTC epoch must not create a false
    one-year discrepancy. No timezone is assigned; offsets remain observations.
    """
    years = events.timestamp.dt.year
    wall_clock = events.local_time.str.replace(r"\s+", " ", regex=True)
    utc = events.timestamp.dt.tz_localize(None)
    candidates = {}
    for shift in [0, -1, 1]:
        local = pd.to_datetime((years + shift).astype(str) + " " + wall_clock,
                               format="%Y %b %d %H:%M:%S", errors="coerce")
        candidates[shift] = (local - utc).dt.total_seconds() / 3600
    offsets = pd.DataFrame(candidates)
    closest = offsets.abs().eq(offsets.abs().min(axis=1), axis=0)
    return offsets.where(closest).bfill(axis=1).iloc[:, 0]


def zero_runs(minutes: pd.DataFrame) -> pd.DataFrame:
    empty = dense_minutes(minutes)["count"].eq(0)
    groups = empty.ne(empty.shift()).cumsum()
    records = []
    for _, block in empty[empty].groupby(groups[empty]):
        records.append({"start": block.index.min(), "end_exclusive": block.index.max() + pd.Timedelta(minutes=1), "minutes": len(block)})
    return pd.DataFrame(records, columns=["start", "end_exclusive", "minutes"]).sort_values("minutes", ascending=False)


def distribution(events: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = events[column].fillna("<missing>").value_counts()
    return pd.DataFrame({"count": counts, "fraction": counts / counts.sum()})


def drift(events: pd.DataFrame, column: str) -> tuple[pd.DataFrame, float]:
    valid = events[events.timestamp.notna()].copy()
    midpoint = valid.timestamp.min() + (valid.timestamp.max() - valid.timestamp.min()) / 2
    valid["period"] = np.where(valid.timestamp <= midpoint, "first_half", "second_half")
    table = pd.crosstab(valid[column].fillna("<missing>"), valid.period, normalize="columns")
    table = table.reindex(columns=["first_half", "second_half"], fill_value=0)
    table["delta"] = table.second_half - table.first_half
    tvd = table.delta.abs().sum() / 2
    return table.reindex(table.delta.abs().sort_values(ascending=False).index), float(tvd)


def message_statistics(events: pd.DataFrame) -> dict:
    messages = events.message.fillna("")
    words, bigrams = Counter(), Counter()
    chars = Counter()
    for message in messages:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z_]{1,}", message.lower())
        words.update(tokens)
        bigrams.update(" ".join(pair) for pair in zip(tokens, tokens[1:]))
        for character in message:
            category = "letter" if character.isalpha() else "number" if character.isdigit() else "space" if character.isspace() else "punctuation_or_symbol"
            chars[category] += 1
    def table(counter):
        return pd.DataFrame(sorted(counter.items(), key=lambda item: (-item[1], item[0])), columns=["token", "count"])
    counts = messages.value_counts()
    q1, q3 = events.message_length.quantile([.25, .75])
    cutoff = q3 + 1.5 * (q3 - q1)
    return {"lengths": events.message_length.describe(percentiles=[.01, .5, .9, .95, .99]),
            "outlier_cutoff_IQR": cutoff, "outliers": int(events.message_length.gt(cutoff).sum()),
            "words": table(words), "bigrams": table(bigrams), "characters": pd.Series(chars),
            "messages": counts, "frequency_bands": pd.Series({"unique_messages": len(counts), "singletons": int(counts.eq(1).sum()), "seen_2_to_5": int(counts.between(2, 5).sum()), "seen_more_than_5": int(counts.gt(5).sum())})}


ENTITY_PATTERNS = {
    "ipv4": re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"),
    "ipv6": re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F]{0,4}(?![\w:])"),
    "port": re.compile(r"\bport[= :]+(\d{1,5})\b|(?:\d{1,3}\.){3}\d{1,3}:(\d{1,5})\b", re.I),
    "uuid": re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"),
    "path": re.compile(r"(?<![\w:/])/(?:[\w.@+~%-]+/)*[\w.@+~%-]+"),
    "url": re.compile(r"\bhttps?://[^\s<>\"']+", re.I),
    "hash_candidate": re.compile(r"\b(?:[0-9a-fA-F]{64}|[0-9a-fA-F]{40}|[0-9a-fA-F]{32})\b"),
    "device_candidate": re.compile(r"\b(?:dn\d+|disk\d+(?:s\d+)?|sd[a-z]\d*|eth\d+|en\d+|awdl\d+|wlan\d+)\b"),
    "session_id": re.compile(r"\bsession(?:\s+id)?\s*[:=]?\s*(0x[0-9a-f]+|[0-9a-f]{8,})\b", re.I),
}


def entity_counts(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counters = {name: Counter() for name in ENTITY_PATTERNS}
    rows_with = Counter()
    for message in events.message:
        for name, pattern in ENTITY_PATTERNS.items():
            found = []
            for match in pattern.finditer(message):
                value = next((s for s in match.groups() if s), match.group())
                if name in ("ipv4", "ipv6"):
                    try:
                        address = ipaddress.ip_address(value)
                    except ValueError:
                        continue
                    if address.version != (4 if name == "ipv4" else 6):
                        continue
                    value = str(address)
                if name == "port" and not 0 <= int(value) <= 65535:
                    continue
                found.append(value)
            counters[name].update(found)
            rows_with[name] += bool(found)
    summary = pd.DataFrame([{"entity": name, "unique": len(counter), "mentions": sum(counter.values()), "events_with_entity": rows_with[name]} for name, counter in counters.items()]).set_index("entity")
    tables = {name: pd.DataFrame(counter.most_common(10), columns=["value", "mentions"]) for name, counter in counters.items()}
    return summary, tables


def component_mentions(events: pd.DataFrame) -> pd.DataFrame:
    names = events.component.dropna().value_counts().head(20).index
    pairs = Counter()
    for target in names:
        mask = events.message.str.contains(r"(?<!\w)" + re.escape(target) + r"(?!\w)", regex=True) & events.component.ne(target)
        pairs.update({(str(source), target): int(count) for source, count in events.loc[mask, "component"].value_counts().items()})
    return pd.DataFrame([(a, b, n) for (a, b), n in pairs.most_common(15)], columns=["producer", "mentioned_component", "events"])


def session_candidates(events: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "Zookeeper":
        ids = events.message.str.extract(ENTITY_PATTERNS["session_id"], expand=False)
        valid = events.assign(session=ids).dropna(subset=["session"])
    elif name == "OpenSSH":
        valid = events.dropna(subset=["host", "pid", "timestamp"]).sort_values(["host", "pid", "timestamp"]).copy()
        idle = valid.groupby(["host", "pid"]).timestamp.diff().gt(pd.Timedelta(minutes=30))
        valid["segment"] = idle.groupby([valid.host, valid.pid]).cumsum()
        valid["session"] = valid.host + ":" + valid.pid + ":" + valid.segment.astype(str)
    else:
        return pd.DataFrame(columns=["events", "first", "last", "duration_seconds"])
    result = valid.groupby("session").agg(events=("message", "size"), first=("timestamp", "min"), last=("timestamp", "max"))
    result["duration_seconds"] = (result["last"] - result["first"]).dt.total_seconds()
    return result.sort_values("events", ascending=False)


def mine_templates(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train Drain3 in source order; map IDs to FINAL templates, not snapshots."""
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from drain3.masking import MaskingInstruction

    config = TemplateMinerConfig()
    config.drain_depth = 4
    config.drain_sim_th = .4
    config.profiling_enabled = False
    config.masking_instructions = [
        MaskingInstruction(ENTITY_PATTERNS["uuid"].pattern, "UUID"),
        MaskingInstruction(ENTITY_PATTERNS["ipv4"].pattern, "IP"),
        MaskingInstruction(r"\b0x[0-9a-fA-F]+\b", "HEX"),
        MaskingInstruction(r"\b\d+\b", "NUM"),
    ]
    miner = TemplateMiner(config=config)
    ids = [miner.add_log_message(message)["cluster_id"] for message in events.message]
    final = {cluster.cluster_id: cluster.get_template() for cluster in miner.drain.clusters}
    result = events.copy()
    result["template_id"] = ids
    result["template"] = result.template_id.map(final)
    templates = result.groupby("template_id").agg(count=("message", "size"), template=("template", "first"), first=("timestamp", "min"), last=("timestamp", "max"))
    templates["observed_days"] = result.groupby("template_id").timestamp.apply(lambda x: x.dt.floor("D").nunique())
    return result, templates.sort_values(["count", "template"], ascending=[False, True])


def window_summary(minutes: pd.DataFrame) -> pd.DataFrame:
    dense = dense_minutes(minutes)
    rows = []
    for size in [1, 5, 15, 60, 1440]:
        counts = dense["count"].resample(f"{size}min").sum()
        rows.append({"window_minutes": size, "windows": len(counts), "empty": int(counts.eq(0).sum()),
                     "empty_fraction": counts.eq(0).mean(), "mean": counts.mean(), "median": counts.median(),
                     "p99": counts.quantile(.99), "max": counts.max(),
                     "above_p99": int(counts.gt(counts.quantile(.99)).sum()),
                     "variance_over_mean": counts.var() / counts.mean() if counts.mean() else np.nan})
    return pd.DataFrame(rows).set_index("window_minutes")


def alert_episodes(minutes: pd.DataFrame) -> pd.DataFrame:
    dense = dense_minutes(minutes)
    active = dense.alerts.fillna(0).gt(0)
    groups = active.ne(active.shift()).cumsum()
    rows = []
    for _, part in dense[active].groupby(groups[active]):
        rows.append({"start": part.index.min(), "end_exclusive": part.index.max() + pd.Timedelta(minutes=1),
                     "active_minutes": len(part), "alerts": int(part.alerts.sum()), "all_records": int(part["count"].sum())})
    return pd.DataFrame(rows, columns=["start", "end_exclusive", "active_minutes", "alerts", "all_records"]).sort_values("alerts", ascending=False)
