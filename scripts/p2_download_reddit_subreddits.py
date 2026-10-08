"""Download specific subreddit files from the Academic Torrents Pushshift dump.

Parses file indices from the .torrent file itself (not the webpage) to guarantee
that --select-file indices match the actual torrent manifest.

File naming convention in the torrent:
    subreddits25/{subreddit}_comments.zst
    subreddits25/{subreddit}_submissions.zst

Usage:
    uv run python scripts/p2_download_reddit_subreddits.py configs/p2_download_reddit.yaml
    uv run python scripts/p2_download_reddit_subreddits.py configs/p2_download_reddit.yaml --list
    uv run python scripts/p2_download_reddit_subreddits.py configs/p2_download_reddit.yaml --debug

Flags:
    --list           Print matched files and their sizes; do not download.
    --debug          Download only the first matched file (smoke test).
    --repair         Re-hash on-disk files against the torrent and re-fetch only bad
                     pieces (aria2 --check-integrity). Fixes corrupt/truncated .zst
                     caused by --select-file leaving shared boundary pieces unfetched.
    --validate-only  Skip downloading; only validate already-downloaded files.

Every download is followed by a validation pass (zstd magic check + `zstd --long=31 -t`
full-stream integrity test). The script exits non-zero if any file fails.

Dependencies:
    uv add requests pyyaml
    brew install aria2 zstd  (macOS)  |  apt install aria2 zstd  (Linux)
"""

import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path

import requests
import yaml


FILELIST_CACHE = Path("data/raw/reddit_filelist.json")
TORRENT_CACHE = Path("data/raw/reddit.torrent")
TORRENT_URL = "https://academictorrents.com/download/{hash}.torrent"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Path to YAML config file")
    p.add_argument("--list", action="store_true", help="List matched files and exit; do not download")
    p.add_argument("--debug", action="store_true", help="Download only the first matched file")
    p.add_argument("--refresh-filelist", action="store_true", help="Re-parse torrent even if cache exists")
    p.add_argument("--repair", action="store_true",
                   help="Pass --check-integrity to aria2c: re-hash on-disk files against the "
                        "torrent and re-fetch only bad/boundary pieces (fixes corrupt .zst files)")
    p.add_argument("--validate-only", action="store_true",
                   help="Skip download; only validate already-downloaded files and report failures")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = {"output_dir", "subreddits", "file_types", "torrent_hash"}
    missing = required - cfg.keys()
    assert not missing, f"Config missing required fields: {missing}"
    return cfg


def resolve_file_types(file_types) -> list[str]:
    if file_types == "both":
        return ["comments", "submissions"]
    assert file_types in {"comments", "submissions"}, (
        f"file_types must be 'comments', 'submissions', or 'both'; got '{file_types}'"
    )
    return [file_types]


# ---------------------------------------------------------------------------
# Bencode parser (stdlib only — no extra dependency)
# ---------------------------------------------------------------------------

def _bdecode(data: bytes, pos: int = 0):
    if data[pos:pos+1] == b'd':
        pos += 1
        d = {}
        while data[pos:pos+1] != b'e':
            key, pos = _bdecode(data, pos)
            val, pos = _bdecode(data, pos)
            d[key] = val
        return d, pos + 1
    if data[pos:pos+1] == b'l':
        pos += 1
        lst = []
        while data[pos:pos+1] != b'e':
            item, pos = _bdecode(data, pos)
            lst.append(item)
        return lst, pos + 1
    if data[pos:pos+1] == b'i':
        end = data.index(b'e', pos)
        return int(data[pos+1:end]), end + 1
    colon = data.index(b':', pos)
    length = int(data[pos:colon])
    start = colon + 1
    return data[start:start+length], start + length


# ---------------------------------------------------------------------------
# Torrent download + file list
# ---------------------------------------------------------------------------

def get_torrent_file(torrent_hash: str) -> Path:
    if TORRENT_CACHE.exists():
        print(f"Using cached torrent file: {TORRENT_CACHE}")
        return TORRENT_CACHE
    url = TORRENT_URL.format(hash=torrent_hash)
    print(f"Downloading .torrent from {url} ... ", end="", flush=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    TORRENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TORRENT_CACHE.write_bytes(resp.content)
    print(f"saved to {TORRENT_CACHE}")
    return TORRENT_CACHE


def parse_torrent_filelist(torrent_path: Path) -> list[dict]:
    """Return [{index, name, size}] with 1-based indices matching the torrent manifest."""
    raw = torrent_path.read_bytes()
    torrent, _ = _bdecode(raw)
    info = torrent[b"info"]
    files = []
    for i, entry in enumerate(info[b"files"], 1):
        path_parts = [p.decode() for p in entry[b"path"]]
        name = "/".join(path_parts)
        size_bytes = entry[b"length"]
        # Human-readable size for display
        for unit in ("B", "kB", "MB", "GB"):
            if size_bytes < 1024:
                size_str = f"{size_bytes:.2f}{unit}"
                break
            size_bytes /= 1024
        else:
            size_str = f"{size_bytes:.2f}TB"
        files.append({"index": i, "name": name, "size": size_str})
    return files


def get_filelist(torrent_hash: str, refresh: bool = False) -> list[dict]:
    """Return file list from cache, rebuilding from the torrent if needed."""
    if not refresh and FILELIST_CACHE.exists():
        with open(FILELIST_CACHE) as f:
            return json.load(f)
    torrent_path = get_torrent_file(torrent_hash)
    print("Parsing torrent file list... ", end="", flush=True)
    files = parse_torrent_filelist(torrent_path)
    print(f"{len(files)} files found.")
    FILELIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(FILELIST_CACHE, "w") as f:
        json.dump(files, f)
    print(f"File list cached to {FILELIST_CACHE}")
    return files


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_files(files: list[dict], subreddits: set[str], file_types: list[str]) -> list[dict]:
    matched = []
    for entry in files:
        name = Path(entry["name"]).name          # "education_comments.zst"
        for ft in file_types:
            suffix = f"_{ft}.zst"
            if name.endswith(suffix):
                subreddit = name[: -len(suffix)]
                if subreddit.lower() in subreddits:
                    matched.append({**entry, "subreddit": subreddit, "file_type": ft})
    return matched


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_file_list(matched: list[dict], subreddits: set[str], file_types: list[str]):
    if not matched:
        print("No files matched. Check subreddit names and file_types in config.")
        return
    print(f"\nMatched {len(matched)} file(s):\n")
    print(f"  {'idx':>6}  {'size':>10}  name")
    print(f"  {'---':>6}  {'----':>10}  ----")
    for e in matched:
        print(f"  {e['index']:>6}  {e['size']:>10}  {e['name']}")

    matched_names = {e["subreddit"].lower() for e in matched}
    missing = subreddits - matched_names
    if missing:
        print(f"\nWARNING: not found in torrent: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Download via aria2c
# ---------------------------------------------------------------------------

def download(matched: list[dict], torrent_path: Path, output_dir: Path, repair: bool = False):
    assert shutil.which("aria2c"), (
        "aria2c not found. Install it:\n"
        "  macOS: brew install aria2\n"
        "  Linux: apt install aria2"
    )

    select = ",".join(str(e["index"]) for e in matched)

    cmd = [
        "aria2c",
        f"--torrent-file={torrent_path}",
        f"--select-file={select}",
        f"--dir={output_dir}",
        "--seed-time=0",
        "--max-connection-per-server=4",
        "--split=4",
        "--console-log-level=notice",
    ]
    if repair:
        # Re-hash existing files against the torrent's per-piece SHA-1 and re-fetch
        # only mismatching pieces. This repairs the shared boundary pieces that
        # --select-file leaves incomplete (the cause of corrupt/truncated .zst).
        cmd.append("--check-integrity=true")

    print(f"\nDownloading {len(matched)} file(s) to: {output_dir}"
          f"{' (repair / integrity-check mode)' if repair else ''}")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def find_output_file(output_dir: Path, entry: dict) -> Path | None:
    """Locate the downloaded file by basename (aria2 nests it under the torrent root).

    If several copies exist (e.g. stale preallocated files from an older layout), take the
    most recently written one, which is the one aria2 just produced.
    """
    hits = list(output_dir.rglob(Path(entry["name"]).name))
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def validate_zst(path: Path) -> tuple[bool, str]:
    """Check zstd magic bytes, then run a full-stream integrity test (long window)."""
    if path.stat().st_size == 0:
        return False, "empty file"
    with open(path, "rb") as f:
        if f.read(4) != ZSTD_MAGIC:
            return False, "not a zstd file (bad magic — likely missing first piece)"
    # --long=31: Watchful1's Reddit dumps use a large long-distance-matching window.
    r = subprocess.run(
        ["zstd", "--long=31", "-t", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "zstd -t failed"
        return False, tail
    return True, "ok"


def validate_downloads(matched: list[dict], output_dir: Path) -> list[tuple[str, str]]:
    """Validate each matched file; return [(name, reason)] for failures."""
    assert shutil.which("zstd"), "zstd not found. Install it: brew install zstd | apt install zstd"
    print(f"\nValidating {len(matched)} file(s)...")
    bad: list[tuple[str, str]] = []
    for e in matched:
        path = find_output_file(output_dir, e)
        if path is None:
            bad.append((e["name"], "missing on disk"))
            print(f"  [MISSING] {e['name']}")
            continue
        ok, msg = validate_zst(path)
        if not ok:
            bad.append((e["name"], msg))
            print(f"  [FAIL] {e['name']}: {msg}")
    if bad:
        print(f"\n{len(bad)}/{len(matched)} file(s) failed validation.")
        print("Repair them with:  uv run python scripts/p2_download_reddit_subreddits.py "
              "<config> --repair")
    else:
        print(f"All {len(matched)} file(s) passed validation.")
    return bad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    subreddits = {s.lower().removeprefix("r/") for s in cfg["subreddits"]}
    file_types = resolve_file_types(cfg["file_types"])
    output_dir = Path(cfg["output_dir"])

    files = get_filelist(cfg["torrent_hash"], refresh=args.refresh_filelist)
    matched = match_files(files, subreddits, file_types)

    if args.debug:
        if not matched:
            print("No files matched; nothing to download in debug mode.")
            sys.exit(0)
        matched = matched[:1]
        print(f"[debug] Limiting to first matched file: {matched[0]['name']}")

    print_file_list(matched, subreddits, file_types)

    if args.list or not matched:
        return

    if args.validate_only:
        bad = validate_downloads(matched, output_dir)
        sys.exit(1 if bad else 0)

    torrent_path = get_torrent_file(cfg["torrent_hash"])
    download(matched, torrent_path, output_dir, repair=args.repair)

    # Content-level validation after every download; aria2's hash check verifies
    # pieces, this verifies the resulting .zst actually decompresses end-to-end.
    bad = validate_downloads(matched, output_dir)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
