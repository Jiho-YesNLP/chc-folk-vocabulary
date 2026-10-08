# Ability-descriptor fingerprint cluster explorer

Interactive d3.js page over the ability-descriptor fingerprints of the Opus 5.5 run over the v10-filtered corpus (the run the manuscript reports). Two-column layout: broad-ability buttons on the left, a UMAP scatter of ability-descriptor spans on the right (color + hull = CHC broad stratum, mirroring `results/track_a_cluster_joint_opus55/scatter.png`). Click a stratum to highlight its points and hull; click any dot or narrow-ability label to open a floating fingerprint panel (definition, discriminator, seeds, distinctive terms, conceptual frames, construal/register/evaluative framing, sample spans).

The live site, [chcfolkvocab.yesnlp.us](https://chcfolkvocab.yesnlp.us), serves this folder as static files; `wrangler.jsonc` at the repository root deploys it as an assets-only Cloudflare Worker (`npx wrangler deploy`).

## Run

`fetch()` needs HTTP (file:// is blocked by CORS), so serve the folder:

```sh
cd webapp
python -m http.server 8000
# open http://localhost:8000/
```

## Data

The page loads three files from `data/joint/`:

- `cluster_layout.json` — projected points + per-stratum hulls + tab20 colors (the only artifact needing the offline embeddings / UMAP). Points are the source-conditioned joint vectors from `p5_embed_joint.py`, the representation the RQ2 analysis measures (71-ability inventory; cross-listed abilities counted once under their parent, none excluded).
- `inventory.json` — the CHC taxonomy (`data/raw/chc_taxonomy.json`) joined with the project annotations (`data/raw/chc_project_annotations.json`), as `src/data/inventory.py` merges them.
- `fingerprints.json` — copy of `data/processed/fingerprints/track_a_opus55/fingerprints.json` (corpus `track_a_v10`, extraction `llm:claude-opus-5.5`).

## Regenerate the data

```sh
uv run python scripts/p5_export_cluster_web_data.py \
    --config configs/p5_cluster_viz.yaml \
    --fingerprints data/processed/fingerprints/track_a_opus55/fingerprints.json --dataset joint
```
