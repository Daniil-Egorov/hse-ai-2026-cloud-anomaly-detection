"""Загрузка и предобработка датасетов LogHub для EDA-ноутбука.

Датасеты: BGL, Windows, HealthApp, Apache.
Источник данных: https://github.com/logpai/loghub (Zenodo record 3227177).

Модуль сам скачивает архивы с Zenodo и распаковывает их в data/raw/,
если файлы ещё не загружены. Каждая функция загрузки возвращает
pandas.DataFrame с разобранными полями лога.
"""

from __future__ import annotations

import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

ZENODO_BASE = "https://zenodo.org/records/3227177/files"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    "BGL": {
        "archive": "BGL.tar.gz",
        "log_file": "BGL.log",
        "description": "BlueGene/L supercomputer (LLNL)",
    },
    "Windows": {
        "archive": "Windows.tar.gz",
        "log_file": "Windows.log",
        "description": "Windows 7 CBS event logs",
    },
    "HealthApp": {
        "archive": "HealthApp.tar.gz",
        "log_file": "HealthApp.log",
        "description": "Android health application logs",
    },
    "Apache": {
        "archive": "Apache.tar.gz",
        "log_file": "Apache.log",
        "description": "Apache HTTP server error log",
    },
}

_APACHE_RE = re.compile(r"^\[(?P<ts>[^\]]+)\] \[(?P<level>\w+)\] (?P<message>.*)$")
_WINDOWS_HEADER_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},")

_BGL_DATETIME_FMT = "%Y-%m-%d-%H.%M.%S.%f"
_APACHE_DATETIME_FMT = "%a %b %d %H:%M:%S %Y"
_HEALTHAPP_DATETIME_FMT = "%Y%m%d-%H:%M:%S:%f"
_WINDOWS_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"

_TOKEN_PATTERNS = [
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<*>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<*>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<*>"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<*>"),
    (re.compile(r"(?<=\s)(/[^\s,;]+)+"), "<*>"),
]


def _normalize_name(name: str) -> str:
    key = name.strip().lower()
    for canonical in DATASETS:
        if key == canonical.lower():
            return canonical
    raise ValueError(f"Неизвестный датасет {name!r}. Доступны: {', '.join(DATASETS)}")


def _download(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Скачивает файл с индикатором прогресса."""
    from tqdm.auto import tqdm

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, miniters=1) as bar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))


def ensure_dataset(name: str, raw_dir: Path | str | None = None) -> Path:
    """Скачивает и распаковывает датасет, если он ещё не загружен.

    Возвращает путь к распакованному .log-файлу.
    """
    canonical = _normalize_name(name)
    meta = DATASETS[canonical]
    directory = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    directory.mkdir(parents=True, exist_ok=True)

    log_file = directory / meta["log_file"]
    if log_file.exists():
        return log_file

    archive = directory / meta["archive"]
    if not archive.exists():
        _download(f"{ZENODO_BASE}/{meta['archive']}?download=1", archive)

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(directory)
    return log_file


def extract_template(message: str) -> str:
    """Приводит одно сообщение к шаблону события, заменяя переменные части на ``<*>``."""
    template = str(message)
    for pattern, replacement in _TOKEN_PATTERNS:
        template = pattern.sub(replacement, template)
    return template


def extract_templates(series: pd.Series) -> pd.Series:
    """Векторизованная версия :func:`extract_template` для колонки сообщений."""
    result = series.astype(str)
    for pattern, replacement in _TOKEN_PATTERNS:
        result = result.str.replace(pattern, replacement, regex=True)
    return result


def load_bgl(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Разбирает BGL.log в DataFrame.

    Первое поле строки — метка аномалии: ``-`` означает норму, всё остальное — алерт.
    """
    labels: list[str] = []
    unix_ts: list[int] = []
    dates: list[str] = []
    nodes: list[str] = []
    datetimes: list[str] = []
    sources: list[str] = []
    sub_sources: list[str] = []
    severities: list[str] = []
    messages: list[str] = []
    skipped = 0

    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            if nrows is not None and index >= nrows:
                break
            parts = line.rstrip("\n").split(None, 9)
            if len(parts) < 10:
                skipped += 1
                continue
            labels.append(parts[0])
            unix_ts.append(int(parts[1]))
            dates.append(parts[2])
            nodes.append(parts[3])
            datetimes.append(parts[4])
            sources.append(parts[6])
            sub_sources.append(parts[7])
            severities.append(parts[8])
            messages.append(parts[9])

    df = pd.DataFrame(
        {
            "label": labels,
            "unix_ts": unix_ts,
            "date": dates,
            "node": nodes,
            "datetime": datetimes,
            "source": sources,
            "sub_source": sub_sources,
            "severity": severities,
            "message": messages,
        }
    )
    df["timestamp"] = pd.to_datetime(df["datetime"], format=_BGL_DATETIME_FMT, errors="coerce")
    df["is_anomaly"] = df["label"].ne("-")
    if skipped:
        print(f"BGL: пропущено строк с неполной структурой: {skipped}")
    return df


def load_apache(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Разбирает Apache.log (error log) в DataFrame.

    Строки без заголовка ``[время] [уровень]`` являются продолжением
    предыдущего сообщения и присоединяются к нему.
    """
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            if nrows is not None and index >= nrows:
                break
            stripped = line.rstrip("\n")
            match = _APACHE_RE.match(stripped)
            if match:
                rows.append(match.groupdict())
            elif rows:
                rows[-1]["message"] = rows[-1]["message"] + " " + stripped

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["ts"], format=_APACHE_DATETIME_FMT, errors="coerce")
    df = df.rename(columns={"level": "log_level"})
    return df


def load_healthapp(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Разбирает HealthApp.log (формат с разделителем ``|``) в DataFrame."""
    timestamps: list[str] = []
    components: list[str] = []
    thread_ids: list[str] = []
    messages: list[str] = []
    skipped = 0

    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            if nrows is not None and index >= nrows:
                break
            parts = line.rstrip("\n").split("|", 3)
            if len(parts) < 4:
                skipped += 1
                continue
            timestamps.append(parts[0])
            components.append(parts[1])
            thread_ids.append(parts[2])
            messages.append(parts[3])

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "component": components,
            "thread_id": thread_ids,
            "message": messages,
        }
    )
    df["timestamp"] = pd.to_datetime(df["datetime"], format=_HEALTHAPP_DATETIME_FMT, errors="coerce")
    if skipped:
        print(f"HealthApp: пропущено строк с неполной структурой: {skipped}")
    return df


def load_windows(
    path: Path,
    nrows: int | None = None,
    sample_size: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Разбирает Windows.log в DataFrame.

    Полный файл содержит ~114 млн строк (26 ГБ), поэтому по умолчанию
    берётся случайная выборка ``sample_size`` строк методом reservoir
    sampling — равномерно по всему файлу и воспроизводимо при заданном
    ``seed``. Чтобы прочитать только первые ``nrows`` строк, передайте
    ``nrows``.

    Строки-продолжения многострочных записей (без заголовка с временем)
    отбрасываются.
    """
    if nrows is not None:
        lines = _take_first_lines(path, nrows)
    else:
        lines = _reservoir_sample(path, sample_size, seed)

    timestamps: list[str] = []
    levels: list[str] = []
    components: list[str] = []
    messages: list[str] = []
    skipped = 0

    for line in lines:
        if not _WINDOWS_HEADER_RE.match(line):
            skipped += 1
            continue
        timestamp, _, rest = line.partition(",")
        fields = rest.split(None, 2)
        if len(fields) < 2:
            skipped += 1
            continue
        timestamps.append(timestamp.strip())
        levels.append(fields[0])
        components.append(fields[1])
        messages.append(fields[2] if len(fields) > 2 else "")

    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "log_level": levels,
            "component": components,
            "message": messages,
        }
    )
    df["timestamp"] = pd.to_datetime(df["datetime"], format=_WINDOWS_DATETIME_FMT, errors="coerce")
    if skipped:
        print(f"Windows: пропущено строк, не соответствующих формату: {skipped}")
    return df


def _take_first_lines(path: Path, nrows: int) -> Iterable[str]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index >= nrows:
                break
            yield line.rstrip("\n")


def _reservoir_sample(path: Path, sample_size: int, seed: int) -> list[str]:
    """Случайная выборка строк по всему файлу методом reservoir sampling.

    Файл читается один раз бинарными чанками, поэтому выборка работает
    быстро даже для Windows.log (26 ГБ): декодируются только отобранные
    строки. Результат воспроизводим благодаря фиксированному ``seed``.
    """
    random = __import__("random")
    random.seed(seed)
    reservoir: list[bytes] = []
    line_index = 0

    with open(path, "rb") as handle:
        remainder = b""
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            chunk = remainder + chunk
            remainder = b""
            start = 0
            while True:
                newline = chunk.find(b"\n", start)
                if newline == -1:
                    remainder = chunk[start:]
                    break
                if line_index < sample_size:
                    reservoir.append(chunk[start:newline])
                else:
                    position = random.randrange(line_index + 1)
                    if position < sample_size:
                        reservoir[position] = chunk[start:newline]
                line_index += 1
                start = newline + 1

    sampled = [raw.decode("utf-8-sig", errors="replace").rstrip("\r") for raw in reservoir]
    print(f"Windows: случайная выборка {len(sampled)} строк из {line_index}")
    return sampled


def load_loghub(
    name: str,
    nrows: int | None = None,
    raw_dir: Path | str | None = None,
    windows_sample_size: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Единая точка входа: загружает датасет LogHub по имени.

    Возвращает DataFrame с колонками ``timestamp``, ``message`` и
    специфичными для датасета полями.
    """
    canonical = _normalize_name(name)
    path = ensure_dataset(canonical, raw_dir)
    if canonical == "BGL":
        return load_bgl(path, nrows=nrows)
    if canonical == "Apache":
        return load_apache(path, nrows=nrows)
    if canonical == "HealthApp":
        return load_healthapp(path, nrows=nrows)
    if canonical == "Windows":
        return load_windows(path, nrows=nrows, sample_size=windows_sample_size, seed=seed)
    raise ValueError(f"Неизвестный датасет {name!r}")


__all__ = [
    "DATASETS",
    "ensure_dataset",
    "extract_template",
    "extract_templates",
    "load_loghub",
    "load_bgl",
    "load_apache",
    "load_healthapp",
    "load_windows",
]
