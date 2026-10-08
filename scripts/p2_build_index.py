"""Build FAISS dense index + sparse JSONL from Reddit .zst corpus (Stage 2).

Outputs per subreddit file:
  data/processed/index/track_a/{source}_faiss.index  — FAISS IndexFlatIP (cosine via inner product)
  data/processed/index/track_a/{source}_sparse.jsonl — sparse lexical weights per passage
  data/processed/index/track_a/{source}_meta.jsonl   — passage metadata aligned by passage_idx

Before building, a preflight scan inspects every corpus file (decompressible?,
supported format?, comment counts, per-year distribution, estimated passages)
and caches the result to a manifest JSON. Run the scan on its own with --inspect.

Sampling: comments per channel are capped (config `max_comments_per_source`) and
drawn `stratified_year` — an even quota per calendar year so no era of a long-lived
subreddit dominates. Set the cap to null to use every comment.

Usage:
  uv run python scripts/p2_build_index.py configs/p2_index.yaml --inspect
  uv run python scripts/p2_build_index.py configs/p2_index.yaml
  uv run python scripts/p2_build_index.py configs/p2_index.yaml --debug
  uv run python scripts/p2_build_index.py configs/p2_index.yaml --corpus-dir /data/... --max-comments 50000
"""

import argparse
import collections
import contextlib
import datetime as dt
import json
import math
import os
import pathlib
import random
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import yaml
import zstandard
from tqdm import tqdm

from src.data.chunker import chunk_text

# faiss and the encoder (heavy, GPU-only deps) are imported lazily inside the
# build path so that --inspect can run anywhere without them.

DELETED = {"[deleted]", "[removed]", ""}

# These academic-torrent Reddit dumps are compressed with long-distance matching
# and a large window; the streaming decompressor must allow it or it raises on
# otherwise-valid files. 2**31 matches the CLI `--long=31`.
ZSTD_MAX_WINDOW = 2 ** 31

# Comments sampled per file to estimate the passages-per-comment ratio.
PASSAGE_SAMPLE_N = 5000


# ── device resolution ──────────────────────────────────────────────────────────

def resolve_device(requested: str, require_gpu: bool) -> str:
    """Resolve the compute device and (optionally) refuse to run on CPU.

    Guarantees the run uses a GPU when `require_gpu` is set, rather than letting
    the encoder silently auto-detect and fall back to CPU.
    """
    import torch

    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    req = (requested or "auto").lower()

    if req == "auto":
        dev = "cuda" if cuda else ("mps" if mps else "cpu")
    elif req == "cuda":
        dev = "cuda" if cuda else ("mps" if mps else "cpu")
    elif req == "mps":
        dev = "mps" if mps else "cpu"
    else:
        dev = "cpu"

    print(f"[device] requested={req}  resolved={dev}  (cuda={cuda}, mps={mps})")

    if require_gpu and dev == "cpu":
        sys.exit(
            "require_gpu is true but no GPU is available (cuda=False, mps=False).\n"
            "Run this on the GPU server, or set require_gpu: false / device: cpu "
            "to allow a (very slow) CPU build."
        )
    if dev == "cpu":
        print("[device] WARNING: running bge-m3 on CPU — encoding will be very slow.")
    return dev


# ── zst streaming ────────────────────────────────────────────────────────────

def stream_zst(path: pathlib.Path):
    dctx = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW)
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        buf = b""
        while True:
            raw_chunk = reader.read(1 << 16)
            if not raw_chunk:
                break
            buf += raw_chunk
            lines = buf.split(b"\n")
            buf = lines.pop()
            for line in lines:
                if line.strip():
                    yield json.loads(line)
        if buf.strip():
            yield json.loads(buf)


def _year_of(comment: dict) -> int | None:
    ts = comment.get("created_utc")
    if ts is None or ts == "":
        return None
    try:
        return dt.datetime.fromtimestamp(int(float(ts)), dt.timezone.utc).year
    except (ValueError, OverflowError, OSError):
        return None


# ── preflight inspection / manifest ────────────────────────────────────────────

def scan_file(path: pathlib.Path) -> dict:
    """Inspect one .zst file: decompressibility, format, counts, year histogram.

    Returns a manifest record. `status` is one of ok | corrupt | unsupported.
    """
    rec: dict = {
        "size_bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
        "status": "ok",
        "error": None,
        "total_comments": 0,
        "deleted_comments": 0,
        "nondeleted_comments": 0,
        "unknown_year": 0,
        "year_counts": {},
        "est_passages": 0,
    }
    year_counts: dict[int, int] = collections.defaultdict(int)
    passages_sampled = 0
    comments_sampled = 0

    try:
        for comment in stream_zst(path):
            rec["total_comments"] += 1
            body = comment.get("body", "")
            body = body.strip() if isinstance(body, str) else ""
            if body in DELETED:
                rec["deleted_comments"] += 1
                continue
            rec["nondeleted_comments"] += 1
            yr = _year_of(comment)
            if yr is None:
                rec["unknown_year"] += 1
            else:
                year_counts[yr] += 1
            if comments_sampled < PASSAGE_SAMPLE_N:
                passages_sampled += len(chunk_text(body))
                comments_sampled += 1
    except zstandard.ZstdError as exc:
        rec["status"] = "corrupt"
        rec["error"] = str(exc)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        rec["status"] = "unsupported"
        rec["error"] = str(exc)

    rec["year_counts"] = {str(y): n for y, n in sorted(year_counts.items())}
    if comments_sampled:
        ratio = passages_sampled / comments_sampled
        rec["est_passages"] = int(rec["nondeleted_comments"] * ratio)
    return rec


def load_or_build_manifest(
    zst_files: list[pathlib.Path],
    manifest_path: pathlib.Path,
    rescan: bool = False,
) -> dict:
    """Return the manifest, scanning any file that is new or has changed.

    Cached records are reused when size + mtime match, so re-runs are instant.
    """
    manifest: dict = {"generated_utc": None, "files": {}}
    if manifest_path.exists() and not rescan:
        try:
            manifest = json.loads(manifest_path.read_text())
            manifest.setdefault("files", {})
        except json.JSONDecodeError:
            manifest = {"generated_utc": None, "files": {}}

    changed = False
    for fp in zst_files:
        name = fp.name
        cached = manifest["files"].get(name)
        fresh = (
            cached
            and not rescan
            and cached.get("size_bytes") == fp.stat().st_size
            and cached.get("mtime") == fp.stat().st_mtime
        )
        if fresh:
            continue
        print(f"[scan] {name} …", flush=True)
        manifest["files"][name] = scan_file(fp)
        changed = True

    if changed:
        manifest["generated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[scan] manifest written → {manifest_path}")
    return manifest


def print_report(zst_files: list[pathlib.Path], manifest: dict, cap: int | None) -> None:
    """Print a per-file inspection table and a corpus-wide summary."""
    hdr = (f"{'file':40s} {'size':>8s} {'status':>11s} {'comments':>11s} "
           f"{'kept(nodel)':>11s} {'del%':>5s} {'yrs':>4s} {'est_pass':>11s} {'will_index':>11s}")
    print("\n" + hdr)
    print("-" * len(hdr))

    tot_comments = tot_keep = tot_index = tot_pass = 0
    bad = []
    for fp in zst_files:
        r = manifest["files"].get(fp.name)
        if not r:
            continue
        nondel = r["nondeleted_comments"]
        deleted = r["deleted_comments"]
        total = r["total_comments"]
        nyears = len(r["year_counts"])
        will_index = nondel if cap is None else min(cap, nondel)
        delpct = (100.0 * deleted / total) if total else 0.0

        tot_comments += total
        tot_keep += nondel
        tot_index += will_index
        tot_pass += r["est_passages"]
        if r["status"] != "ok":
            bad.append((fp.name, r["status"], r.get("error", "")))

        print(f"{fp.name:40.40s} {_h(r['size_bytes']):>8s} {r['status']:>11s} "
              f"{total:>11,} {nondel:>11,} {delpct:>4.0f}% {nyears:>4d} "
              f"{r['est_passages']:>11,} {will_index:>11,}")

    print("-" * len(hdr))
    print(f"{'TOTAL':40s} {'':>8s} {'':>11s} {tot_comments:>11,} {tot_keep:>11,} "
          f"{'':>5s} {'':>4s} {tot_pass:>11,} {tot_index:>11,}")
    cap_str = "no cap" if cap is None else f"{cap:,}/channel"
    print(f"\nCap: {cap_str}  →  ~{tot_index:,} comments will be indexed "
          f"(of {tot_keep:,} non-deleted).")
    if bad:
        print(f"\n⚠ {len(bad)} file(s) NOT OK and will be skipped at build time:")
        for name, status, err in bad:
            print(f"    [{status}] {name} — {err}")


def _h(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}G"


# ── stratified sampling plan ────────────────────────────────────────────────────

def sampling_plan(rec: dict, cap: int | None, strategy: str) -> dict:
    """Build a per-year keep-probability plan for one file.

    Returns {"mode": ..., "p_year": {year:int -> p}, "p_default": p}.
    `head` ignores year and keeps the first `cap` non-deleted comments.
    `uniform` keeps each non-deleted comment with p = cap/total.
    `stratified_year` allocates an even quota across years with data.
    """
    nondel = rec["nondeleted_comments"]
    if cap is None or nondel <= cap:
        return {"mode": "all", "p_year": {}, "p_default": 1.0}

    if strategy == "head":
        return {"mode": "head", "p_year": {}, "p_default": 1.0}

    if strategy == "uniform":
        return {"mode": "prob", "p_year": {}, "p_default": cap / nondel}

    # stratified_year
    year_counts = {int(y): n for y, n in rec["year_counts"].items()}
    n_buckets = len(year_counts) + (1 if rec["unknown_year"] else 0)
    if n_buckets == 0:
        return {"mode": "prob", "p_year": {}, "p_default": cap / max(nondel, 1)}
    quota = math.ceil(cap / n_buckets)
    p_year = {y: min(1.0, quota / n) for y, n in year_counts.items() if n > 0}
    p_unknown = min(1.0, quota / rec["unknown_year"]) if rec["unknown_year"] else 0.0
    return {"mode": "prob", "p_year": p_year, "p_default": p_unknown}


# ── main ─────────────────────────────────────────────────────────────────────

def main(
    config_path: str,
    debug: bool = False,
    inspect_only: bool = False,
    rescan: bool = False,
    corpus_dir_override: str | None = None,
    max_comments_override: int | None = None,
    only: list[str] | None = None,
) -> None:
    cfg = yaml.safe_load(pathlib.Path(config_path).read_text())
    required = {"corpus_dir", "output_dir", "encoder_model", "batch_size", "device"}
    missing = required - cfg.keys()
    if missing:
        sys.exit(f"Config missing required fields: {missing}")

    corpus_dir = pathlib.Path(corpus_dir_override or cfg["corpus_dir"])
    output_dir = pathlib.Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_dir.is_dir():
        sys.exit(f"corpus_dir does not exist: {corpus_dir}")

    cap = max_comments_override if max_comments_override is not None else cfg.get("max_comments_per_source")
    if cap is not None:
        cap = int(cap)
    strategy = cfg.get("sampling", "stratified_year")
    seed = int(cfg.get("sampling_seed", 42))
    manifest_path = pathlib.Path(cfg.get("manifest_path", output_dir / "manifest.json"))

    zst_files = sorted(corpus_dir.glob("*_comments.zst"))
    if not zst_files:
        sys.exit(f"No *_comments.zst files found in {corpus_dir}")
    if only:
        wanted = {s.strip().lower() for s in only if s.strip()}
        by_name = {f.stem.split("_comments")[0].lower(): f for f in zst_files}
        unknown = wanted - by_name.keys()
        if unknown:
            sys.exit(f"--only names not found in {corpus_dir}: {sorted(unknown)}")
        zst_files = [by_name[n] for n in sorted(wanted)]
        print(f"[only] {len(zst_files)} of {len(by_name)} sources selected: {sorted(wanted)}")
    if debug:
        zst_files = zst_files[:1]

    # ── preflight: inspect every file and report ──────────────────────────────
    print(f"Preflight scan of {len(zst_files)} file(s) in {corpus_dir} …")
    manifest = load_or_build_manifest(zst_files, manifest_path, rescan=rescan)
    print_report(zst_files, manifest, cap)

    if inspect_only:
        print("\n[inspect] done — no index built (drop --inspect to build).")
        return

    # ── encoder (GPU enforced here, before any heavy work) ────────────────────
    device = resolve_device(cfg["device"], bool(cfg.get("require_gpu", True)))
    print(f"\nLoading encoder: {cfg['encoder_model']} on {device}")
    import faiss  # heavy GPU dep — imported only on the build path
    from src.models.encoder import BGEM3Encoder  # imported late so --inspect needs no FlagEmbedding
    enc = BGEM3Encoder(model_name=cfg["encoder_model"], device=device)

    print(f"Sampling: {strategy}  |  cap: {'none' if cap is None else f'{cap:,}/channel'}  |  seed: {seed}")

    for zst_file in zst_files:
        rec = manifest["files"].get(zst_file.name, {})
        if rec.get("status") != "ok":
            print(f"[skip] {zst_file.name} — status={rec.get('status', 'unknown')} ({rec.get('error', '')})")
            continue

        source_name = zst_file.stem.split("_comments")[0].lower()
        faiss_path = output_dir / f"{source_name}_faiss.index"
        sparse_path = output_dir / f"{source_name}_sparse.jsonl"
        meta_path = output_dir / f"{source_name}_meta.jsonl"

        if faiss_path.exists() and not cfg.get("overwrite", False):
            print(f"[skip] {source_name} — index exists (set overwrite: true to rebuild)")
            continue

        plan = sampling_plan(rec, cap, strategy)
        rng = random.Random(f"{seed}:{source_name}")
        target = rec["nondeleted_comments"] if cap is None else min(cap, rec["nondeleted_comments"])
        print(f"\n[index] {zst_file.name}  (mode={plan['mode']}, target≈{target:,})")

        index = faiss.IndexFlatIP(1024)  # inner product on L2-normalized vecs = cosine
        passage_idx = 0
        kept_comments = 0
        batch_texts: list[str] = []
        batch_meta: list[dict] = []

        def flush() -> None:
            nonlocal passage_idx
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                out = enc.encode(batch_texts, batch_size=cfg["batch_size"])
            index.add(out["dense"].astype(np.float32))
            for i, (sparse, meta) in enumerate(zip(out["sparse"], batch_meta)):
                idx = passage_idx + i
                sparse_fh.write(json.dumps({"passage_idx": idx, "sparse": {k: float(v) for k, v in sparse.items()}}) + "\n")
                meta_fh.write(json.dumps({"passage_idx": idx, **meta}) + "\n")
            passage_idx += len(batch_texts)
            batch_texts.clear()
            batch_meta.clear()

        def keep(comment: dict) -> bool:
            """Sampling decision for one non-deleted comment."""
            if plan["mode"] == "all":
                return True
            if plan["mode"] == "head":
                return kept_comments < target
            yr = _year_of(comment)
            p = plan["p_year"].get(yr, plan["p_default"]) if yr is not None else plan["p_default"]
            return rng.random() < p

        try:
            with sparse_path.open("w") as sparse_fh, meta_path.open("w") as meta_fh:
                for comment in tqdm(stream_zst(zst_file), desc=source_name):
                    body = comment.get("body", "")
                    body = body.strip() if isinstance(body, str) else ""
                    if body in DELETED:
                        continue
                    if not keep(comment):
                        continue
                    kept_comments += 1
                    for chunk in chunk_text(body):
                        batch_texts.append(chunk["text"])
                        batch_meta.append({
                            "source_id": comment.get("id", ""),
                            "topic": source_name,
                            "text": chunk["text"],
                        })
                        if len(batch_texts) >= cfg["batch_size"] * 4:
                            flush()
                    if plan["mode"] == "head" and kept_comments >= target:
                        break
                    if debug and passage_idx >= 2000:
                        break
                if batch_texts:
                    flush()
        except zstandard.ZstdError as exc:
            print(f"  [error] {zst_file.name}: {exc} — skipping")
            for partial in (sparse_path, meta_path):
                partial.unlink(missing_ok=True)
            continue

        faiss.write_index(index, str(faiss_path))
        print(f"  → {kept_comments:,} comments / {passage_idx:,} passages  |  {faiss_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BGE-M3 index for Track A corpus")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--inspect", action="store_true",
                        help="Only scan + report on the corpus files; build nothing")
    parser.add_argument("--rescan", action="store_true",
                        help="Force re-scan of all files, ignoring the cached manifest")
    parser.add_argument("--corpus-dir", metavar="DIR", default=None,
                        help="Override corpus_dir from the config")
    parser.add_argument("--max-comments", type=int, default=None,
                        help="Override max_comments_per_source (per channel)")
    parser.add_argument("--debug", action="store_true",
                        help="Process first .zst file only, stop at 2000 passages")
    parser.add_argument("--only", metavar="SUBS", default=None,
                        help="Comma-separated subreddit names to build (the rest are skipped). "
                             "Use to shard a multi-GPU run; pin each process with CUDA_VISIBLE_DEVICES.")
    args = parser.parse_args()
    main(
        args.config,
        debug=args.debug,
        inspect_only=args.inspect,
        rescan=args.rescan,
        corpus_dir_override=args.corpus_dir,
        max_comments_override=args.max_comments,
        only=args.only.split(",") if args.only else None,
    )
