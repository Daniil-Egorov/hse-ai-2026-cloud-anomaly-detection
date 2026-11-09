"""Full LogHub archives for Hadoop, OpenStack, Android_v1 and HPC.

No connection to util.loghub is required; the separate name avoids conflicts
when the four participants' EDA work is combined.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASETS = {
    'Hadoop': dict(md5='6dfc442b9e86adb519db9437d4e7d3d4',
                   domain='WordCount/PageRank; 5 machines, injected failures',
                   labels='application/job; not individual lines',
                   time='local time, milliseconds; timezone unspecified'),
    'OpenStack': dict(md5='66bd42c07837a094d9b0ea2d036b5713',
                      domain='CloudLab, OpenStack Nova; failure injection',
                      labels='4 VM instance IDs; file name is capture context only',
                      time='local time, milliseconds; timezone unspecified'),
    'Android': dict(md5='1a1bac1cf0ea95bc88e296f689f0258f',
                    domain='Android_v1; instrumented smartphone application framework',
                    labels='unavailable',
                    time='month-day, milliseconds; year and timezone unavailable'),
    'HPC': dict(md5='4a7f0483f7b4fdd8b278c266039db464',
                domain='LANL System 20 high-performance computing cluster',
                labels='unavailable; numeric field is not an anomaly label',
                time='Unix seconds, UTC'),
}

UUID = r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'
PATTERNS = {
    'Hadoop': re.compile(r'^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) (?P<level>\w+) \[(?P<thread>[^]]*)\] (?P<component>[^:]+):\s?(?P<message>.*)$'),
    'OpenStack': re.compile(r'^(?P<origin>\S+) (?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) (?P<pid>\d+) (?P<level>\w+) (?P<component>\S+)(?:\s+(?P<message>.*))?$'),
    'Android': re.compile(r'^(?P<time>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+(?P<pid>\d+)\s+(?P<thread>\d+)\s+(?P<level>[VDIWEFS])\s+(?P<component>[^:]+):\s?(?P<message>.*)$'),
    'HPC': re.compile(r'^(?P<record_id>\d+) (?P<prefix>.*?) (?P<time>-?\d+(?:\.\d+)?) (?P<numeric_flag>-?\d+)\s*(?P<message>.*)$'),
}
ANDROID_FALLBACK = re.compile(r'^(?P<time>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+(?P<pid>\d+)\s+(?P<thread>\d+)\s+(?P<level>[VDIWEFS])\s+(?P<message>.*)$')


@dataclass
class LogBundle:
    name: str
    events: pd.DataFrame
    files: pd.DataFrame
    raw_first: pd.DataFrame
    raw_random: pd.DataFrame
    quality: dict
    labels: pd.DataFrame
    manifest: dict


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch_loghub(name: str, data_dir: str | Path) -> Path:
    """Download, check MD5 and safely extract once. Fail loudly on corruption."""
    if name not in DATASETS:
        raise ValueError(f'Expected one of {list(DATASETS)}, got {name!r}')
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f'{name}.tar.gz'
    url = f'https://zenodo.org/api/records/3227177/files/{name}.tar.gz/content'
    if not archive_path.exists():
        temporary = archive_path.with_suffix('.gz.part')
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'HSE-LogHub-EDA/1.0'})
                with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as out:
                    shutil.copyfileobj(source, out)
                temporary.replace(archive_path)
                break
            except OSError:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    if _digest(archive_path, 'md5') != DATASETS[name]['md5']:
        raise ValueError(f'Corrupt archive: {archive_path}. Remove it and retry.')
    destination = root / name
    marker = destination / '.extracted.json'
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        ready = marker.exists() and all(
            (destination / m.name).is_file() and (destination / m.name).stat().st_size == m.size
            for m in members
        )
        if not ready:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                    raise ValueError(f'Unsafe archive member: {member.name}')
            destination.mkdir(parents=True, exist_ok=True)
            archive.extractall(destination, filter='data')
            marker.write_text(json.dumps({'md5': DATASETS[name]['md5'], 'url': url}), encoding='utf-8')
    return destination


def hadoop_labels(path: Path) -> pd.DataFrame:
    rows, workload, label = [], None, None
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('### '):
            workload = line[4:].strip()
        elif line.endswith(':'):
            label = line[:-1].strip()
        elif line.startswith('+ application_'):
            rows.append(dict(session_id=line[2:].strip(), workload=workload, context_label=label))
    result = pd.DataFrame(rows)
    if result.empty or result.session_id.duplicated().any():
        raise ValueError('Invalid Hadoop label mapping')
    return result


def load_loghub(name: str, data_dir: str | Path, seed: int = 42) -> LogBundle:
    """Read every physical line. Attach Hadoop continuation lines to their header.

    First/random raw samples and physical-line counters include blank and broken
    lines. Unparsed orphan lines are retained as separate events with NaT.
    Android uses a visibly synthetic leap year (2000), never a claimed real year.
    """
    import random

    root = fetch_loghub(name, data_dir)
    labels = hadoop_labels(root / 'abnormal_label.txt') if name == 'Hadoop' else pd.DataFrame()
    label_map = labels.set_index('session_id').to_dict('index') if not labels.empty else {}
    abnormal_ids = set(re.findall(UUID, (root / 'anomaly_labels.txt').read_text())) if name == 'OpenStack' else set()
    if abnormal_ids:
        labels = pd.DataFrame({'instance_id': sorted(abnormal_ids), 'context_label': 'injected VM'})
    rng = random.Random(seed)
    rows, files, first, reservoir = [], [], [], []
    raw_counts, seen_raw = Counter(), set()
    total = 0
    for path in sorted(root.rglob('*.log')):
        relative = path.relative_to(root).as_posix()
        pending = None
        file_counts = Counter()
        with path.open('rb') as stream:
            for line_no, raw in enumerate(stream, 1):
                total += 1
                text = raw.rstrip(b'\r\n').decode('utf-8', errors='replace')
                file_counts['physical_lines'] += 1
                file_counts['physical_characters'] += len(text)
                file_counts['lines_without_standard_header'] += int(PATTERNS[name].match(text) is None)
                file_counts['blank_lines'] += int(not text.strip())
                file_counts['invalid_utf8_lines'] += int('\ufffd' in text)
                file_counts['crlf_lines'] += int(raw.endswith(b'\r\n'))
                file_counts['missing_final_newline'] += int(not raw.endswith(b'\n'))
                fingerprint = hashlib.blake2b(raw.rstrip(b'\r\n'), digest_size=16).digest()
                raw_counts['duplicate_physical_lines'] += int(fingerprint in seen_raw)
                seen_raw.add(fingerprint)
                item = dict(source_file=relative, line_no=line_no, raw=text)
                if len(first) < 100:
                    first.append(item)
                if len(reservoir) < 100:
                    reservoir.append(item)
                else:
                    index = rng.randrange(total)
                    if index < 100:
                        reservoir[index] = item
                match = PATTERNS[name].match(text)
                variant = None
                if match is None and name == 'Android':
                    match = ANDROID_FALLBACK.match(text)
                    if match is not None:
                        variant = 'missing_tag_separator'
                        file_counts['missing_tag_separator'] += 1
                if match:
                    if pending is not None:
                        rows.append(pending)
                    fields = match.groupdict()
                    fields['message'] = fields.get('message') or ''
                    if variant:
                        fields.update(component='unknown', format_variant=variant)
                    fields.update(source_file=relative, line_start=line_no, line_end=line_no,
                                  physical_lines=1, parse_ok=True, raw_length=len(text))
                    if name == 'Hadoop':
                        fields['session_id'] = path.parent.name
                        fields.update(label_map.get(path.parent.name, {'context_label': None, 'workload': None}))
                    elif name == 'OpenStack':
                        fields['capture'] = path.stem
                        fields['request_id'] = next(iter(re.findall(r'req-(' + UUID + r')', text)), None)
                        ids = set(re.findall(r'(?:\[instance:\s*|/servers/|for instance\s+)(' + UUID + r')', text))
                        fields['instance_id'] = '|'.join(sorted(ids)) or None
                        fields['context_label'] = ('injected VM' if ids & abnormal_ids else
                                                   'normal capture' if path.stem in {'openstack_normal1', 'openstack_normal2'} else 'unlabelled in abnormal capture')
                    elif name == 'HPC':
                        prefix = fields.pop('prefix').split()
                        host_index = next((i for i, token in enumerate(prefix) if re.match(r'(?:node-|gige\d|Interconnect-)', token)), 0)
                        fields['resource'] = prefix[host_index]
                        fields['host'] = prefix[host_index] if re.match(r'(?:node-|gige\d|Interconnect-)', prefix[host_index]) else None
                        fields['component'] = ' '.join(prefix[:host_index] + prefix[host_index + 1:-1]) or 'unknown'
                        fields['event_type'] = prefix[-1]
                    pending = fields
                elif name == 'Hadoop' and pending is not None and pending['parse_ok']:
                    pending['message'] += '\n' + text
                    pending['line_end'] = line_no
                    pending['physical_lines'] += 1
                    pending['raw_length'] += len(text) + 1
                    file_counts['continuation_lines'] += 1
                else:
                    if pending is not None:
                        rows.append(pending)
                    pending = dict(source_file=relative, line_start=line_no, line_end=line_no,
                                   physical_lines=1, parse_ok=False, raw_length=len(text),
                                   message=text, time=None)
                    file_counts['orphan_lines'] += 1
        if pending is not None:
            rows.append(pending)
        for key, value in file_counts.items():
            raw_counts[key] += value
        files.append(dict(source_file=relative, bytes=path.stat().st_size,
                          sha256=_digest(path, 'sha256'), **file_counts))
    df = pd.DataFrame(rows)
    formats = {'Hadoop': '%Y-%m-%d %H:%M:%S,%f', 'OpenStack': '%Y-%m-%d %H:%M:%S.%f',
               'Android': '%Y-%m-%d %H:%M:%S.%f'}
    if name == 'HPC':
        seconds = pd.to_numeric(df.time.where(df.time.str.fullmatch(r'\d{10}', na=False)), errors='coerce')
        df['timestamp'] = pd.to_datetime(seconds, unit='s', utc=True, errors='coerce')
    else:
        dates = '2000-' + df.time if name == 'Android' else df.time
        df['timestamp'] = pd.to_datetime(dates, format=formats[name], errors='coerce')
    df['message_length'] = df.message.str.len()
    for col in ['component', 'level', 'host', 'source_file', 'context_label', 'workload', 'capture']:
        if col in df:
            df[col] = df[col].astype('category')
    raw_counts.update(logical_events=len(df), timestamp_missing=int(df.timestamp.isna().sum()))
    assert int(df.physical_lines.sum()) == total, 'Physical lines lost during parsing'
    manifest = dict(dataset=name, archive_md5=DATASETS[name]['md5'],
                    archive_url=f'https://zenodo.org/api/records/3227177/files/{name}.tar.gz/content',
                    source_url=f'https://github.com/logpai/loghub/tree/master/{name}',
                    scope='full archive; no sampling of events', seed=seed,
                    timestamp_semantics=DATASETS[name]['time'])
    return LogBundle(name, df, pd.DataFrame(files), pd.DataFrame(first),
                     pd.DataFrame(reservoir), dict(raw_counts), labels, manifest)
