"""Stage 6 pilot — convergent validation feasibility: test description -> ability assignment.

CPU/local. Assigns each battery test description to one of the candidate abilities under four
representations of an ability and scores the rank of the test's true ability:

  A  label        the CHC technical name                        one vector per ability
  B  definition   the inventory definition                      one vector per ability
  C  seeds        the human-authored retrieval seeds (inventory `seeds`)
  D  spans        on-target descriptor spans (span-only vectors, capped per ability)
  E  profile      the ability's full folk vocabulary as one document: canonical descriptions and
                  conceptual frames (with their member phrases), distinctive lexical terms, social
                  contexts, and every on-target descriptor span (deduplicated), from the assembled
                  fingerprint (config `fingerprints_dir`)

C and D are scored as the mean of the top-k cosines to the ability's members; A, B and E as a
single cosine. Before scoring, each side is mean-centered within its own register (test
descriptions by their mean; each condition's vectors by that condition's mean), since the
folk/expert register gap is largely a constant offset. `zscore: true` additionally
standardizes each ability's scores across the test set (a hubness correction standing in for
the background-text z-normalization of the full design).

Candidate pool: abilities with at least `min_support` descriptor spans, minus
`exclude_abilities`; the same pool is used for every condition. Tests whose primary target is
not in the pool cannot be scored and are listed separately. Dense channel only.

This is a PILOT: its materials are unverified drafts and its numbers are not for reporting.

Usage:
  uv run python scripts/p6_validation_pilot.py configs/p6_validation_wj4.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from p5_repr_ablation import load_rows  # noqa: E402  (same rows as the cached span vectors)
from src.data.inventory import read_inventory  # noqa: E402  taxonomy + project annotations, merged

REQUIRED_KEYS = ("materials_path", "inventory_path", "descriptions_dir", "corpus_dir", "fingerprints_dir", "sep",
                 "span_cache_path", "encoder_model", "device", "batch_size", "top_k",
                 "cap_per_ability", "min_support", "exclude_abilities", "random_seed", "out_dir")
CONDITIONS = ("A_label", "B_definition", "C_seeds", "D_spans", "E_profile")
# paired comparisons reported (x vs y): what the full vocabulary adds over its spans, how it
# compares with the hand-written seeds, and the spans-vs-seeds comparison
COMPARISONS = (("E_profile", "D_spans"), ("E_profile", "C_seeds"), ("D_spans", "C_seeds"))


def unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def center(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return unit(X - mu)


def profile_text(fp: dict, spans: list[str]) -> str:
    """The ability's full folk vocabulary as one plain-text document (no ability name or code)."""
    def phrases(items):
        out = []
        for it in items:
            out += [it["canonical"]] + [m for m in it.get("members", []) if m != it["canonical"]]
        return out
    parts = []
    if fp.get("canonical_descriptions"):
        parts.append("How people describe it: " + "; ".join(phrases(fp["canonical_descriptions"])) + ".")
    if fp.get("conceptual_frame"):
        parts.append("Underlying ideas: " + "; ".join(phrases(fp["conceptual_frame"])) + ".")
    if fp.get("lexical"):
        parts.append("Characteristic words: " + ", ".join(x["term"] for x in fp["lexical"]) + ".")
    ctx = (fp.get("evaluative_framing") or {}).get("social_contexts") or []
    if ctx:
        parts.append("Where it comes up: " + "; ".join(ctx) + ".")
    if spans:
        parts.append("Descriptions: " + "; ".join(spans) + ".")
    return " ".join(parts)


def score_matrix(T: np.ndarray, members: dict[str, np.ndarray], pool: list[str], k: int) -> np.ndarray:
    """S[t, a] = mean of the top-k cosines between test t and ability a's member vectors."""
    S = np.zeros((len(T), len(pool)))
    for j, a in enumerate(pool):
        sims = T @ members[a].T                              # (tests, members)
        kk = min(k, sims.shape[1])
        S[:, j] = np.sort(sims, axis=1)[:, -kk:].mean(axis=1)
    return S


def evaluate(S: np.ndarray, tests: list[dict], pool: list[str], stratum: dict[str, str]) -> dict:
    idx = {a: i for i, a in enumerate(pool)}
    ranks, any_hit1, broad1, within_rr = [], [], [], []
    per_test = []
    for t, row in zip(tests, S):
        order = np.argsort(-row)
        tgt = t["targets"][0]
        r = int(np.where(order == idx[tgt])[0][0]) + 1
        top = pool[order[0]]
        ranks.append(r)
        any_hit1.append(top in t["targets"])
        broad1.append(stratum[top] == stratum[tgt])
        same = [j for j, a in enumerate(pool) if stratum[a] == stratum[tgt]]
        sub = row[same]
        wr = int((sub > row[idx[tgt]]).sum()) + 1
        within_rr.append(1 / wr)
        per_test.append({"id": t["id"], "name": t["name"], "target": tgt, "rank": r,
                         "top": top, "top5": [pool[j] for j in order[:5]],
                         "within_rank": wr, "within_n": len(same)})
    ranks = np.array(ranks)
    return {"acc1": float((ranks == 1).mean()), "acc5": float((ranks <= 5).mean()),
            "mrr": float((1 / ranks).mean()), "any1": float(np.mean(any_hit1)),
            "broad1": float(np.mean(broad1)), "within_mrr": float(np.mean(within_rr)),
            "per_test": per_test}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6 pilot — test -> ability assignment")
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"Config missing required keys: {missing}")
    rng = np.random.default_rng(int(cfg["random_seed"]))
    out_dir = pathlib.Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = read_inventory(cfg["inventory_path"])
    ab = {n["uid"]: n for b in inv["broad_abilities"] for n in b["narrow_abilities"]}
    stratum = {u: u.split("-", 1)[0] for u in ab}
    tests = json.loads(pathlib.Path(cfg["materials_path"]).read_text())["tests"]

    # ---- C members: cached span-only vectors, aligned to load_rows() ----
    rows = load_rows(pathlib.Path(cfg["descriptions_dir"]), pathlib.Path(cfg["corpus_dir"]), cfg["sep"])
    z = np.load(cfg["span_cache_path"])
    if int(z["n_rows"]) != len(rows):
        sys.exit(f"span cache has {int(z['n_rows'])} rows, loader gives {len(rows)}: re-run p5_repr_ablation.py")
    span_vecs = z["span"].astype(np.float64)
    by_uid = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_uid[r["uid"]].append(i)
    support = {u: len(v) for u, v in by_uid.items()}
    pool = sorted(u for u in ab if support.get(u, 0) >= int(cfg["min_support"])
                  and u not in cfg["exclude_abilities"])
    cap = int(cfg["cap_per_ability"])
    c_idx = {u: (by_uid[u] if len(by_uid[u]) <= cap else list(rng.choice(by_uid[u], cap, replace=False)))
             for u in pool}

    # ---- E: one profile document per ability from the assembled fingerprint ----
    fp_dir = pathlib.Path(cfg["fingerprints_dir"])
    profiles = {}
    for u in pool:
        fp = json.loads((fp_dir / f"{u}.json").read_text())
        spans = list(dict.fromkeys(rows[i]["span"] for i in by_uid[u]))   # all on-target spans, deduplicated
        profiles[u] = profile_text(fp, spans)
    (out_dir / "profiles.json").write_text(json.dumps(profiles, indent=2, ensure_ascii=False))

    # ---- encode A, B, C, E and the test descriptions ----
    from src.models.encoder import BGEM3Encoder
    enc = BGEM3Encoder(model_name=cfg["encoder_model"], device=cfg["device"])
    seeds = {u: [s for s in (ab[u].get("seeds") or []) if s.strip()] for u in pool}
    texts = ([t["description"] for t in tests] + [ab[u]["name"] for u in pool]
             + [ab[u]["definition"] for u in pool] + [s for u in pool for s in seeds[u]])
    V = unit(np.asarray(enc.encode(texts, batch_size=int(cfg["batch_size"]))["dense"], dtype=np.float64))
    nT, nP = len(tests), len(pool)
    T, lab, dfn = V[:nT], V[nT:nT + nP], V[nT + nP:nT + 2 * nP]
    sd = V[nT + 2 * nP:]
    # profiles run far past the default 512-token window; BGE-M3 accepts 8,192
    prof_max = int(cfg.get("profile_max_length", 8192))
    prof = unit(np.asarray(enc.encode([profiles[u] for u in pool], batch_size=4, max_length=prof_max)["dense"],
                           dtype=np.float64))
    n_tok = [len(enc.model.tokenizer(profiles[u])["input_ids"]) for u in pool]
    print(f"profile tokens: median {int(np.median(n_tok))}, max {max(n_tok)} (window {prof_max})")
    seed_members, o = {}, 0
    for u in pool:
        seed_members[u] = sd[o:o + len(seeds[u])]
        o += len(seeds[u])

    # ---- register-wise mean-centering ----
    Tc = center(T, T.mean(0))
    all_c = np.vstack([span_vecs[c_idx[u]] for u in pool])
    mu_c = unit(all_c).mean(0)
    reps = {
        "A_label": {u: center(lab[j:j + 1], lab.mean(0)) for j, u in enumerate(pool)},
        "B_definition": {u: center(dfn[j:j + 1], dfn.mean(0)) for j, u in enumerate(pool)},
        "C_seeds": {u: center(seed_members[u], sd.mean(0)) for u in pool},
        "D_spans": {u: center(unit(span_vecs[c_idx[u]]), mu_c) for u in pool},
        "E_profile": {u: center(prof[j:j + 1], prof.mean(0)) for j, u in enumerate(pool)},
    }

    scorable = [t for t in tests if t["targets"][0] in pool]
    unscorable = [t for t in tests if t["targets"][0] not in pool]
    Ts = Tc[[tests.index(t) for t in scorable]]
    chance = {"acc1": 1 / nP, "acc5": 5 / nP,
              "mrr": float(np.mean([1 / r for r in range(1, nP + 1)])),
              "broad1": float(np.mean([sum(stratum[a] == stratum[t["targets"][0]] for a in pool) / nP
                                       for t in scorable]))}
    k = int(cfg["top_k"])
    results, raw_scores = {}, {}
    for zs in (False, True):
        for cond in CONDITIONS:
            S = score_matrix(Ts, reps[cond], pool, k)
            if zs:
                S = (S - S.mean(0)) / (S.std(0) + 1e-9)
            results[f"{cond}{'_z' if zs else ''}"] = evaluate(S, scorable, pool, stratum)
            if not zs:
                raw_scores[cond] = S
    # raw score matrices for p6_validation_report.py (pools, support split, error analysis)
    np.savez_compressed(out_dir / "scores.npz", pool=np.array(pool),
                        test_ids=np.array([t["id"] for t in scorable]),
                        support=np.array([support.get(u, 0) for u in pool]), **raw_scores)

    # paired comparisons over tests (raw scores): x vs y
    from scipy.stats import binomtest
    def compare(x: str, y: str) -> dict:
        rx = np.array([t["rank"] for t in results[x]["per_test"]])
        ry = np.array([t["rank"] for t in results[y]["per_test"]])
        diff = 1 / rx - 1 / ry
        brng = np.random.default_rng(int(cfg["random_seed"]))
        boots = [diff[brng.integers(0, len(diff), len(diff))].mean() for _ in range(int(cfg.get("bootstrap_n", 10000)))]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        b01 = int(((rx == 1) & (ry != 1)).sum()); b10 = int(((rx != 1) & (ry == 1)).sum())
        p = float(binomtest(min(b01, b10), b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0
        return {"mrr_diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "x_better": int((rx < ry).sum()),
                "y_better": int((ry < rx).sum()), "ties": int((rx == ry).sum()),
                "acc1_discordant_x_only": b01, "acc1_discordant_y_only": b10, "mcnemar_exact_p": p}
    comparisons = {f"{x} vs {y}": compare(x, y) for x, y in COMPARISONS}

    # manipulation check: how close each test description sits to its target's definition
    overlap = {t["id"]: float(T[tests.index(t)] @ dfn[pool.index(t["targets"][0])]) for t in scorable}

    summary = {"n_tests": len(tests), "n_scorable": len(scorable), "pool_size": nP,
               "unscorable": [{"id": t["id"], "target": t["targets"][0],
                               "support": support.get(t["targets"][0], 0)} for t in unscorable],
               "chance": chance, "definition_overlap": overlap, "comparisons": comparisons,
               "results": {c: {m: v for m, v in r.items() if m != "per_test"} for c, r in results.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_test.json").write_text(json.dumps({c: r["per_test"] for c, r in results.items()}, indent=2))

    L = [f"# Stage 6 convergent validation ({cfg['materials_path']})\n",
         f"tests {len(tests)} (scorable {len(scorable)}), pool {nP} abilities, top-k {k}, cap {cap}\n",
         f"chance: acc@1 {chance['acc1']:.3f}  acc@5 {chance['acc5']:.3f}  MRR {chance['mrr']:.3f}  broad@1 {chance['broad1']:.3f}\n",
         "| condition | acc@1 | acc@5 | MRR | any-target@1 | broad@1 | within-stratum MRR |",
         "|---|---|---|---|---|---|---|"]
    for c, r in results.items():
        L.append(f"| {c} | {r['acc1']:.3f} | {r['acc5']:.3f} | {r['mrr']:.3f} | {r['any1']:.3f} | "
                 f"{r['broad1']:.3f} | {r['within_mrr']:.3f} |")
    for name, c in comparisons.items():
        L.append(f"\n{name} (MRR, paired bootstrap): diff {c['mrr_diff']:+.3f}, 95% CI "
                 f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]; "
                 f"first better {c['x_better']}, second better {c['y_better']}, ties {c['ties']}; "
                 f"McNemar acc@1 exact p {c['mcnemar_exact_p']:.3f}")
    L.append("\nUnscorable (target below support): " +
             ", ".join(f"{u['id']}->{u['target']} (n={u['support']})" for u in summary["unscorable"]))
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
