#!/usr/bin/env bash
# Скачивает датасет LogHub по имени; URL берётся из LogHub README (ветка master).
#
#   loghub_download.sh --list
#   loghub_download.sh HDFS_v1 ./data
#   loghub_download.sh Spark ./data --extract
set -euo pipefail

README_URL="https://raw.githubusercontent.com/logpai/loghub/master/README.md"

usage() {
    echo "usage: ${0##*/} DATASET [DEST_DIR] [--extract] [--force]" >&2
    echo "       ${0##*/} --list" >&2
    exit 2
}

die() { echo "error: $*" >&2; exit 1; }

dataset="" dest="" extract=0 force=0 list=0
pos=0
while [ $# -gt 0 ]; do
    case "$1" in
        --extract) extract=1 ;;
        --force)   force=1 ;;
        --list)    list=1 ;;
        -h|--help) usage ;;
        -*)        usage ;;
        *)         pos=$((pos + 1))
                   case "$pos" in 1) dataset="$1" ;; 2) dest="$1" ;; *) usage ;; esac ;;
    esac
    shift
done

if [ "$list" -eq 0 ]; then
    [ -n "$dataset" ] || usage
    [[ "$dataset" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid dataset name: $dataset"
fi
dest="${dest:-data}"

if command -v curl >/dev/null 2>&1; then
    tool=curl
elif command -v wget >/dev/null 2>&1; then
    tool=wget
else
    die "neither curl nor wget found"
fi

ca="${NODE_EXTRA_CA_CERTS:-}"

# Параметры функции: URL OUT QUIET [CA_FILE]
dl() {
    local url="$1" out="$2" quiet="$3" cafile="${4:-}"
    if [ "$tool" = curl ]; then
        local -a opts=(-fL --retry 3)
        if [ "$quiet" -eq 1 ]; then opts+=(-sS); fi
        curl "${opts[@]}" ${cafile:+--cacert "$cafile"} -o "$out" "$url"
    else
        local -a opts=(--tries=3)
        if [ "$quiet" -eq 1 ]; then opts+=(-q -nv); fi
        wget "${opts[@]}" ${cafile:+--ca-certificate="$cafile"} -O "$out" "$url"
    fi
}

# Сначала используем системное хранилище доверия; повторяем с дополнительным CA только
# при ошибке сертификата (у curl код 60, у wget код 5), т.к. --cacert заменяет системные корни.
fetch() {
    local rc=0
    dl "$1" "$2" "$3" || rc=$?
    [ "$rc" -eq 0 ] && return 0
    if [ -n "$ca" ] && [ -r "$ca" ] && { [ "$tool" = curl ] && [ "$rc" -eq 60 ] || [ "$tool" = wget ] && [ "$rc" -eq 5 ]; }; then
        dl "$1" "$2" "$3" "$ca"
        return $?
    fi
    return "$rc"
}

if [ "$list" -eq 0 ] && [ "$force" -eq 0 ] && [ -d "$dest" ] && [ -n "$(ls -A "$dest")" ]; then
    echo "skipping: $dest is not empty (use --force to download anyway)" >&2
    echo "$dest"
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fetch "$README_URL" "$tmp/README.md" 1 || die "could not fetch the LogHub README"

if [ "$list" -eq 1 ]; then
    sed -n 's/^| *\[\([^]]*\)\](.*/\1/p' "$tmp/README.md"
    exit 0
fi

row="$(grep -iE "^\| *\[${dataset}\]\(" "$tmp/README.md" | head -1 || true)"
[ -n "$row" ] || die "dataset '$dataset' not found in the README; try --list"

url="$(printf '%s\n' "$row" | sed -n 's/.*\[:link:\](\([^)]*\)).*/\1/p')"
[ -n "$url" ] || die "no download link for '$dataset' in the README"
size="$(printf '%s\n' "$row" | awk -F'|' '{gsub(/^ +| +$/, "", $7); print $7}')"

file="${url##*/}"; file="${file%%\?*}"
[ -n "$file" ] || die "cannot derive a file name from $url"

mkdir -p "$dest"
out="$dest/$file"

echo "downloading $dataset ($size): $url" >&2
part="$dest/.$file.part"
trap 'rm -rf "$tmp" "$part"' EXIT
fetch "$url" "$part" 0 || die "download failed: $url"
[ -s "$part" ] || die "empty response: $url"
mv "$part" "$out"

if [ "$extract" -eq 1 ]; then
    case "$file" in
        *.zip)
            command -v unzip >/dev/null 2>&1 || die "unzip not found"
            unzip -q -o "$out" -d "$dest" ;;
        *.tar|*.tar.gz|*.tgz|*.tar.bz2|*.tar.xz)
            tar -xf "$out" -C "$dest" ;;
        *)  die "don't know how to extract: $file" ;;
    esac
    rm -f "$out"
    echo "$dest"
else
    echo "$out"
fi
