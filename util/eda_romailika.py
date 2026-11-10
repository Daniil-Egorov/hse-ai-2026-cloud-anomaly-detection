"""EDA tables and figures. Dataset-wide statistics are exact unless labelled sample."""
from __future__ import annotations

import ipaddress
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from util.loghub_romailika import LogBundle, UUID

ENTITY_PATTERNS = {
    'url': re.compile(r'https?://[^\s<>"\]]+'),
    'uuid': re.compile(r'\b' + UUID + r'\b'),
    'ipv4': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    'port': re.compile(r'(?:\b(?:\d{1,3}\.){3}\d{1,3}:|\bport[=: ]+)(\d{1,5})\b', re.I),
    'path': re.compile(r'(?<![\w:/])/(?:[\w.\-]+/)*[\w.\-]+'),
    'hash': re.compile(r'\b[0-9a-fA-F]{32,64}\b'),
    'device': re.compile(r'\b(?:node-\d+|Interconnect-[\w-]+|gige\d+|HWID=\d+)\b'),
    'session_candidate': re.compile(r'\b(?:application_\d+_\d+|container_[\w]+|attempt_[\w]+|task_[\w]+|req-' + UUID + r'|command \d+)\b'),
}
NORMALIZERS = [(pattern, '<' + name + '>') for name, pattern in ENTITY_PATTERNS.items() if name != 'port']
NORMALIZERS += [
    (re.compile(r'\b0x[0-9a-fA-F]+\b'), '<hex>'),
    (re.compile(r'(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)'), '<num>'),
    (re.compile(r'\s+'), ' '),
]


def normalize_template(message: str) -> str:
    """Deterministic regex baseline, not Drain and not ground-truth EventId."""
    for pattern, token in NORMALIZERS:
        message = pattern.sub(token, message)
    return message.strip()


def frequency(series: pd.Series, top: int = 15) -> pd.DataFrame:
    counts = series.astype('string').fillna('<missing>').value_counts()
    return pd.DataFrame({'count': counts, 'share_pct': (100 * counts / counts.sum()).round(3)}).head(top)


def window_table(df: pd.DataFrame) -> pd.DataFrame:
    """Count empty windows inside each file's observed span, never between runs.

    Quantiles include zeros. Only a sparse vector plus the zero count is needed,
    except for a compact vector constructed for quantiles (HPC < 2M minutes).
    """
    rows = []
    for rule in ['1min', '5min', '1h', '1D']:
        all_counts, expected = [], 0
        for _, group in df.dropna(subset=['timestamp']).groupby('source_file', observed=True):
            bins = group.timestamp.dt.floor(rule)
            counts = bins.value_counts()
            all_counts.extend(counts.tolist())
            expected += int((bins.max() - bins.min()).total_seconds() / pd.Timedelta(rule).total_seconds()) + 1
        occupied = len(all_counts)
        if not occupied:
            continue
        values = np.concatenate([np.array(all_counts), np.zeros(expected - occupied, dtype=int)])
        rows.append(dict(window=rule, windows=expected, empty=expected - occupied,
                         empty_pct=round(100 * (expected - occupied) / expected, 3),
                         mean=round(float(values.mean()), 3), median=float(np.median(values)),
                         p99=float(np.quantile(values, .99)), max=int(values.max())))
    return pd.DataFrame(rows)


def analyze(bundle: LogBundle, output_dir: Path, seed: int = 42) -> dict:
    df = bundle.events
    tables = {}
    tables['files'] = bundle.files
    tables['quality'] = pd.Series(bundle.quality, name='value').rename_axis('metric').to_frame()
    tables['first_100'] = bundle.raw_first
    tables['random_100'] = bundle.raw_random
    tables['unparsed'] = df.loc[~df.parse_ok, ['source_file', 'line_start', 'message']].head(20)
    tables['labels_source'] = bundle.labels
    tables['lengths'] = df[['raw_length', 'message_length', 'physical_lines']].describe(percentiles=[.5, .9, .99]).T
    tables['longest_messages'] = df.nlargest(5, 'message_length')[['source_file', 'line_start', 'message_length', 'message']]
    messages = df.message.value_counts()
    tables['common_messages'] = messages.head(15).rename_axis('message').reset_index(name='count')
    tables['rare_messages'] = messages[messages == 1].head(15).rename_axis('message').reset_index(name='count')
    text_summary = dict(unique_messages=len(messages), singleton_messages=int((messages == 1).sum()),
                        duplicate_logical_messages=int(len(df) - len(messages)),
                        mean_physical_line_chars=round(bundle.quality['physical_characters'] / bundle.quality['physical_lines'], 3))
    total_chars = int((messages.index.str.len().to_numpy() * messages.to_numpy()).sum())
    for name, pattern in [('digits', r'\d'), ('letters', r'[^\W\d_]'), ('punctuation', r'[^\w\s]')]:
        count = int((messages.index.str.count(pattern).to_numpy() * messages.to_numpy()).sum())
        text_summary[name + '_chars'] = count
        text_summary[name + '_share_pct'] = round(100 * count / total_chars, 3) if total_chars else 0
    tables['text_summary'] = pd.Series(text_summary).dropna().rename_axis('metric').to_frame('value')

    mapping = {message: normalize_template(message) for message in messages.index}
    df['template'] = df.message.map(mapping)
    template_counts = df.template.value_counts()
    tables['templates_common'] = template_counts.head(20).rename_axis('template').reset_index(name='count')
    tables['templates_rare'] = template_counts.tail(15).rename_axis('template').reset_index(name='count')
    lifetimes = df.groupby('template', sort=False).agg(count=('message', 'size'),
                                                     first_seen=('timestamp', 'min'), last_seen=('timestamp', 'max'))
    lifetimes['span_seconds'] = (lifetimes.last_seen - lifetimes.first_seen).dt.total_seconds()
    tables['template_lifetime'] = lifetimes.sort_values(['first_seen', 'count']).reset_index()
    tables['template_endpoints'] = pd.concat([
        lifetimes.sort_values('first_seen', ascending=False).head(8).assign(selection='latest first appearance'),
        lifetimes.sort_values('last_seen').head(8).assign(selection='earliest last appearance')
    ]).reset_index()

    entity_rows, entity_examples = [], []
    for name, pattern in ENTITY_PATTERNS.items():
        counter = Counter()
        for message, weight in messages.items():
            for match in pattern.findall(message):
                if name == 'ipv4':
                    try:
                        ipaddress.IPv4Address(match)
                    except ipaddress.AddressValueError:
                        continue
                if name == 'port' and not 0 <= int(match) <= 65535:
                    continue
                counter[match] += int(weight)
        entity_rows.append(dict(entity=name, unique=len(counter), mentions=sum(counter.values())))
        entity_examples.extend(dict(entity=name, value=value, mentions=count) for value, count in counter.most_common(10))
    tables['entities'] = pd.DataFrame(entity_rows)
    tables['entity_examples'] = pd.DataFrame(entity_examples)
    for column in ['session_id', 'request_id', 'instance_id', 'pid', 'thread', 'record_id', 'numeric_flag', 'event_type', 'resource', 'format_variant']:
        if column in df:
            tables[column] = frequency(df[column])

    sample = df.sample(min(100_000, len(df)), random_state=seed)
    words, bigrams = Counter(), Counter()
    references = Counter()
    components = set(df.component.dropna().astype(str).unique()) if 'component' in df else set()
    for source_component, message in zip(sample.component.astype('string').fillna('unknown'), sample.message):
        tokens = re.findall(r'[a-zA-Z][a-zA-Z_]{2,}', message.lower())
        words.update(tokens)
        bigrams.update(' '.join(pair) for pair in zip(tokens, tokens[1:]))
        targets = (set(re.findall(r'\b[A-Za-z_][\w.]+\b', message)) & components) - {source_component}
        references.update((source_component, target) for target in targets)
    for name, counter in [('words_sample', words), ('bigrams_sample', bigrams)]:
        pairs = counter.most_common(15) + sorted(counter.items(), key=lambda pair: (pair[1], pair[0]))[:15]
        tables[name] = pd.DataFrame(pairs, columns=['token', 'count'])
    tables['component_mentions_sample'] = pd.DataFrame(
        [dict(source_component=source, mentioned_component=target, messages=count)
         for (source, target), count in references.most_common(15)],
        columns=['source_component', 'mentioned_component', 'messages'])

    temporal = []
    for file, group in df.groupby('source_file', observed=True, sort=False):
        timestamps = group.timestamp.dropna()
        if timestamps.empty:
            continue
        gaps = timestamps.sort_values().diff().dt.total_seconds()
        order = timestamps.diff().dt.total_seconds()
        temporal.append(dict(source_file=file, start=timestamps.min(), end=timestamps.max(),
                             backwards_steps=int((order < 0).sum()), duplicate_timestamps=int(timestamps.duplicated().sum()),
                             max_gap_seconds=float(gaps.max()) if len(gaps) > 1 else 0,
                             median_positive_gap_seconds=float(gaps[gaps > 0].median()) if (gaps > 0).any() else 0))
    tables['time_by_file'] = pd.DataFrame(temporal)
    tables['largest_gaps'] = tables['time_by_file'].nlargest(15, 'max_gap_seconds')
    tables['windows'] = window_table(df)
    for column in ['level', 'component', 'host', 'capture', 'context_label', 'workload']:
        if column in df:
            tables[column] = frequency(df[column])
    for column in ['level', 'component', 'host']:
        if column in df:
            top = df[column].value_counts().head(8).index
            table = pd.crosstab(df.timestamp.dt.floor('1D'), df[column].where(df[column].isin(top)).astype('string'))
            tables[column + '_daily_share'] = table.div(table.sum(axis=1), axis=0).round(4)
    tables['hour_of_day'] = df.timestamp.dt.hour.value_counts().sort_index().to_frame('events')

    if 'context_label' in df:
        tables['label_comparison'] = df.groupby('context_label', observed=True).agg(
            events=('message', 'size'), mean_length=('message_length', 'mean'), templates=('template', 'nunique'),
            first_seen=('timestamp', 'min'), last_seen=('timestamp', 'max'))
        if 'level' in df:
            tables['level_by_label'] = pd.crosstab(df.context_label, df.level, normalize='index').round(4)
        anomaly = df.context_label.isin(['Machine down', 'Network disconnection', 'Disk full', 'injected VM'])
        normal = df.context_label.isin(['Normal', 'normal capture'])
        anomalous_templates = set(df.loc[anomaly, 'template'])
        normal_templates = set(df.loc[normal, 'template'])
        anomaly_counts = df.loc[anomaly, 'template'].value_counts()
        normal_counts = df.loc[normal, 'template'].value_counts()
        tables['templates_context_exclusive'] = pd.DataFrame(
            [dict(context='failure context only', template=t, count=int(anomaly_counts[t])) for t in sorted(anomalous_templates - normal_templates)] +
            [dict(context='normal context only', template=t, count=int(normal_counts[t])) for t in sorted(normal_templates - anomalous_templates)]
        )
        key = 'session_id' if bundle.name == 'Hadoop' else 'instance_id'
        tables['label_units'] = bundle.labels.groupby('context_label').size().to_frame('labelled_units')
        tables['label_units']['share_pct'] = (100 * tables['label_units'].labelled_units / len(bundle.labels)).round(3)
        if bundle.name == 'Hadoop':
            tables['session_summary'] = df.groupby(['session_id', 'context_label', 'workload'], observed=True).agg(
                events=('message', 'size'), physical_lines=('physical_lines', 'sum'),
                first_seen=('timestamp', 'min'), last_seen=('timestamp', 'max')).reset_index()
        else:
            tables['session_summary'] = df.loc[anomaly].groupby('instance_id').agg(
                events=('message', 'size'), first_seen=('timestamp', 'min'), last_seen=('timestamp', 'max')).reset_index()
        evidence = df.loc[anomaly & df[key].notna()]
        evidence_ranked = evidence.assign(signal=(evidence.level.isin(['ERROR', 'FATAL']).astype(int) * 3 +
            evidence.level.isin(['WARN', 'WARNING']).astype(int) +
            evidence.message.str.contains(r'fail|exception|error|killed|disconnect', case=False, regex=True).astype(int)))
        examples = evidence_ranked.sort_values('signal', ascending=False, kind='stable').drop_duplicates(key)
        if bundle.name == 'Hadoop':
            examples = examples.groupby('context_label', observed=True, sort=True).head(4).sort_values('context_label')
        tables['anomaly_examples'] = examples.head(10)[[key, 'context_label', 'timestamp', 'level', 'message']]
        tables['failure_daily'] = pd.crosstab(df.timestamp.dt.floor('1D'), df.context_label)
        episodes = []
        for (file, session), group in evidence.groupby(['source_file', key], observed=True):
            timestamps = group.timestamp.dropna().sort_values()
            groups = timestamps.diff().gt(pd.Timedelta('1min')).cumsum()
            for _, times in timestamps.groupby(groups):
                episodes.append(dict(source_file=file, session=session, start=times.min(), end=times.max(),
                                     seconds=(times.max() - times.min()).total_seconds(), events=len(times)))
        tables['failure_context_episodes'] = pd.DataFrame(episodes).sort_values('events', ascending=False) if episodes else pd.DataFrame()
    summary = dict(dataset=bundle.name, **bundle.quality,
                   start=str(df.timestamp.min()), end=str(df.timestamp.max()),
                   unique_templates=len(template_counts),
                   top10_template_share_pct=round(100 * template_counts.head(10).sum() / len(df), 3),
                   mean_message_length=round(float(df.message_length.mean()), 3),
                   singleton_messages=int((messages == 1).sum()),
                   files=len(bundle.files), sample_text_events=len(sample),
                   backwards_steps=int(tables['time_by_file'].backwards_steps.sum()),
                   duplicate_timestamps_within_files=int(tables['time_by_file'].duplicate_timestamps.sum()))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f'{name}.csv', index=True, encoding='utf-8-sig')
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'manifest.json').write_text(json.dumps(bundle.manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return dict(summary=summary, tables=tables)


def show_table(table: pd.DataFrame, rows: int = 20) -> None:
    from IPython.display import HTML, display
    shown = table.head(rows).map(lambda value: value[:500] + '…' if isinstance(value, str) and len(value) > 500 else value)
    display(HTML('<div style="max-height:480px;overflow:auto">' +
                 shown.to_html(index=True, escape=True) + '</div>'))


def show_raw(table: pd.DataFrame, title: str) -> None:
    """All 100 rows, untruncated raw text, in a scrollable HTML view."""
    from IPython.display import HTML, display
    import html
    lines = [f"{r.source_file}:{r.line_no}  {r.raw}" for r in table.itertuples()]
    display(HTML('<p>' + html.escape(title) + '</p><pre style="max-height:450px;overflow:auto;white-space:pre">' +
                 html.escape('\n'.join(lines)) + '</pre>'))


def plot_overview(bundle: LogBundle, directory: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    df = bundle.events
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    timed = df.dropna(subset=['timestamp'])
    for file, group in timed.groupby('source_file', observed=True):
        counts = group.groupby(group.timestamp.dt.floor('1h')).size().sort_index()
        axes[0, 0].plot(counts.index, counts.values, lw=.7, alpha=.65, marker='.', markersize=2)
    axes[0, 0].set_title('Events/hour, separately within source files')
    axes[0, 0].tick_params(axis='x', rotation=25)
    axes[0, 1].hist(np.log1p(df.message_length), bins=50, color='#2a788e')
    axes[0, 1].set(title='Message lengths', xlabel='log(1 + characters)', ylabel='Events')
    field = 'level' if 'level' in df else 'event_type'
    df[field].value_counts().head(12).sort_values().plot.barh(ax=axes[1, 0], color='#2a788e')
    axes[1, 0].set_title(field)
    components = df.component.value_counts().head(10).sort_values()
    components.index = ['.'.join(str(label).split('.')[-2:])[-48:] for label in components.index]
    components.plot.barh(ax=axes[1, 1], color='#3b528b')
    axes[1, 1].set_title('Top components (abbreviated; full names in CSV)')
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(MaxNLocator(4)) if ax in [axes[1, 0], axes[1, 1]] else None
        if ax in [axes[1, 0], axes[1, 1]]:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1000:g}k'))
    fig.suptitle(bundle.name + (' (Android year 2000 is synthetic)' if bundle.name == 'Android' else ''))
    fig.savefig(directory / 'overview.png', dpi=130)
    plt.show()
    plt.close(fig)


def plot_dynamics(bundle: LogBundle, analysis: dict, directory: Path) -> None:
    import matplotlib.pyplot as plt
    tables = analysis['tables']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    bundle.events.timestamp.dt.hour.value_counts().sort_index().plot.bar(ax=axes[0], color='#2a788e')
    axes[0].set_title('Events by hour (exposure not normalized)')
    share = tables.get('component_daily_share')
    if share is not None:
        top = share.mean().nlargest(3).index
        shown = share[top].copy()
        shown.columns = ['.'.join(str(label).split('.')[-2:])[-28:] for label in top]
        shown.plot(ax=axes[1], legend=True)
        axes[1].legend(fontsize=7)
    axes[1].set_title('Top 3 components / day (top-8 denominator)')
    if 'failure_daily' in tables:
        tables['failure_daily'].plot(ax=axes[2], legend=True)
        axes[2].set_title('Context labels / day (not row ground truth)')
    else:
        tables['windows'].set_index('window').empty_pct.plot.bar(ax=axes[2], color='#3b528b')
        axes[2].set_title('Empty windows within observed file spans, %')
    fig.savefig(directory / 'dynamics.png', dpi=130)
    plt.show()
    plt.close(fig)
