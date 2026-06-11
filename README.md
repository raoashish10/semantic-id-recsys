# recsys-mlops

Offline-to-online recommendation system built on semantic IDs (RQ-VAE) and sequential modeling (SASRec), following the retrieval → ranking pattern used in production discovery systems.

Inspired by Eugene Yan's [Semantic IDs](https://eugeneyan.com/writing/semantic-ids/) and [System Design for Recommendations and Search](https://eugeneyan.com/writing/system-design-for-discovery/).

---

## Architecture

The system follows a 2×2 decomposition: **offline vs. online** environments, and **candidate retrieval vs. ranking** steps.

```
                    CANDIDATE RETRIEVAL              RANKING
               ┌────────────────────────────┬─────────────────────────────┐
               │                            │                             │
               │  • RQ-VAE encodes item     │  • SASRec trained on        │
  OFFLINE      │    embeddings → semantic   │    semantic ID sequences    │
               │    IDs (3–4 tokens/item)   │    → scores next-item       │
               │                            │    predictions              │
               │  • FAISS ANN index built   │                             │
               │    from sentence-          │  • Precomputed recs written │
               │    transformer embeddings  │    to Redis via beam search │
               │    (audit use only — see   │    (SASRec on full history) │
               │    RQ-VAE Auditing below)  │                             │
               ├────────────────────────────┼─────────────────────────────┤
               │                            │                             │
               │  Cache hit → return        │  SASRec beam search over    │
  ONLINE       │  immediately (<1ms)        │  semantic ID space          │
               │                            │                             │
               │  Cache miss → SASRec       │  Beam expands top predicted │
               │  real-time inference       │  code combinations →        │
               │  on session (~20ms)        │  Redis reverse lookup →     │
               │                            │  ranked item list           │
               └────────────────────────────┴─────────────────────────────┘
```

### Recommendation paths

| Path | When | How |
|---|---|---|
| **Precomputed** | Cache hit (`recs:{user_id}` in Redis, TTL 24h) | SASRec beam search on full user history — runs offline, sequential and interaction-order-aware |
| **Real-time** | Cache miss, session ≥ 3 items | Same SASRec beam search on the current session — result written back to Redis in background |
| **Cold start** | Cache miss, session < 3 items | Semantic ID prefix lookup — finds items sharing the (c0, c1) prefix of the most recent viewed item |

The cold-start path uses the RQ-VAE prefix hierarchy instead of the model. If a user has only viewed 1–2 items, SASRec's attention has too little signal to be meaningful. Instead, the system looks up the most common `(c0, c1)` prefix among the session items, then returns a random sample from the `prefix:{c0}:{c1}` Redis set. This means a user who viewed one vitamin-C serum immediately gets recommendations for other face serums in the same subcategory — no session history needed.

The SASRec and precomputed paths run identical inference code (`serving/inference.py`), so warm-user recommendations are consistent regardless of which path served them.

---

## Semantic IDs

Each item is encoded into a 3–4 token tuple by the RQ-VAE. Items with similar content share common prefix tokens, forming a tree:

```
token 0  →  coarse category    (256 possible values)
token 1  →  subcategory        (256 possible values)
token 2  →  fine-grained attr  (256 possible values)
token 3  →  uniqueness token   (added only if collision rate > 5%)
```

This hierarchy gives cold-start generalization for free: a new item sharing a prefix with known items inherits their recommendation signal. SASRec treats the full tuple as a compound token — predicting the next item means predicting all L codes sequentially.

**Example — item cold start:**

```
c0=7  →  "hydrating skincare"
c1=3  →  "face serum"
c2=8  →  "with vitamin C"
```

SASRec learned: _"users who buy hydrating face serums tend to next buy vitamin C face serums."_
A new vitamin C serum launched today gets encoded by the RQ-VAE into `(7, 3, 8, 0)` and gets
that recommendation signal immediately — the model learned the category pattern, not the specific
product. No retraining required.

---

## RQ-VAE Auditing (FAISS)

The FAISS index (`artifacts/ann/index.faiss`) is built from raw sentence-transformer embeddings and is **not used in serving**. Its purpose is to validate that the RQ-VAE learned meaningful quantization:

> If the RQ-VAE is working correctly, items that are nearest neighbors in continuous embedding space should share semantic ID prefixes.

Use `notebooks/audit_rqvae.ipynb` (or `serving/retrieval.py` directly) to query the index and compare prefix overlap between a query item and its nearest neighbors. A high prefix-match rate indicates the discrete codebook tree mirrors the continuous embedding structure.

---

## Offline pipeline

```
download → preprocess → embeddings → rqvae → sasrec → ann → ranking → index → precompute
```

| Step | Output | Where |
|---|---|---|
| `download` | Raw Amazon Reviews 2023 (All_Beauty) | `data/raw/` |
| `preprocess` | `items.parquet`, `sequences.parquet` (5-core filtered) | `data/processed/` |
| `embeddings` | `item_embeddings.npy` via all-MiniLM-L6-v2 | `artifacts/embeddings/` |
| `rqvae` | `semantic_ids.parquet`, `model.pt` | `artifacts/rqvae/` |
| `sasrec` | `model.pt` (best Hit@10 checkpoint) | `artifacts/sasrec/` |
| `ann` | `index.faiss` — audit use only | `artifacts/ann/` |
| `ranking` | `model.pt` — offline artifact, not loaded in serving | `artifacts/ranking/` |
| `index` | Redis: `item:*`, `sid:*`, `feat:*`, `prefix:*` keys | Redis |
| `precompute` | Redis: `recs:{user_id}` keys with 24h TTL | Redis |

---

## Online serving

```
POST /recommend
{
  "user_id": "u123",        // optional — enables cache lookup + write-back
  "session": ["B001", ...], // recently interacted item IDs, ordered
  "top_k": 10
}
```

### Request flow

```
1. Cache check     recs:{user_id} in Redis (TTL 24h)         → return, cache_hit: true
                            │ miss
2. Cold start?     len(session) < 3
   ├─ LLM path     IntentCache hit (Redis, TTL 5m)
   │               └─ or: Ollama llama3.2 (200ms timeout)
   │                        → IntentResult {intent, predicted_prefixes, confidence}
   │                        → intent_based_recommend: prefix candidates + cosine re-rank
   │                        → cold_start_method: "intent"
   │
   └─ Fallback     SRANDMEMBER prefix:{c0}:{c1} (unranked)
                            → cold_start_method: "prefix_fallback"
                            (also used when LLM times out, returns bad JSON, or Ollama is down)
                            │ no results from either
3. SASRec          build_input → beam_recommend → sid:* reverse lookup
4. Write-back      set_user_recs (background task, non-blocking)
```

### Response fields

| Field | Type | Description |
|---|---|---|
| `recommendations` | list | ranked items with `item_id`, `title`, `semantic_id` |
| `session_length` | int | number of session items resolved in the catalog |
| `cache_hit` | bool | true if served from precomputed `recs:{user_id}` |
| `cold_start_method` | str \| null | `"intent"`, `"prefix_fallback"`, or null (warm-user SASRec path) |

### Cold start in detail

When a user has fewer than 3 session items, SASRec's self-attention has too little signal to predict coherently. Instead:

**LLM path** (`COLD_START_LLM_ENABLED=true`, default): The session item titles and their `(c0, c1)` values are sent to a locally-running Ollama instance (model `llama3.2`). The system prompt grounds the LLM in the actual codebook vocabulary — it sees the real `(c0, c1)` values and is instructed to predict prefixes grounded in those values, not arbitrary numbers. The LLM returns a structured `IntentResult` with a natural-language intent description, up to 3 predicted `(c0, c1)` prefixes with weights, and a confidence score. Candidates are then retrieved from the Redis prefix sets and re-ranked by weighted cosine similarity between the intent text embedding and each item's stored `feat:*` embedding (same all-MiniLM-L6-v2 model used offline). Intent results are cached in Redis for 5 minutes under `intent:{fingerprint}`.

**Prefix fallback** (if LLM times out, fails, or `COLD_START_LLM_ENABLED=false`): Falls back to a random sample from the `prefix:{c0}:{c1}` Redis set of the most common prefix in the session. Unranked but always fast and never fails as long as the index step ran.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `COLD_START_LLM_ENABLED` | `true` | Set to `false` to skip Ollama entirely |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `REC_TTL_SECONDS` | `86400` | TTL for precomputed user recs (seconds) |

---

## Running locally

```bash
pip install -e ".[dev]"

# Start Redis + MLflow
make up

# Run the full offline pipeline
make pipeline

# Or step by step
make data && make embeddings && make rqvae && make sasrec
make ann && make ranking
make pipeline --from index   # populate Redis + precompute recs

# Start the serving API
make serve   # http://localhost:8000/docs
```

---

## Project layout

```
recsys-mlops/
├── data/                    # download + preprocess scripts
├── offline/
│   ├── embeddings/          # sentence-transformer inference
│   ├── rqvae/               # RQ-VAE model + training
│   ├── sasrec/              # sequential recommender
│   ├── ann/                 # FAISS index (audit only)
│   ├── ranking/             # MLP ranker (offline artifact)
│   ├── precompute.py        # batch rec precomputation → Redis
│   └── pipeline.py          # Prefect flow wiring all steps
├── serving/
│   ├── api/                 # FastAPI app (SASRec-backed)
│   ├── store/               # Redis client (semantic IDs + feature store)
│   ├── retrieval.py         # FAISS query helper (audit use only)
│   └── inference.py         # SASRec beam search — shared by API + precompute
├── artifacts/               # generated by pipeline, not committed
└── infra/
    └── docker-compose.yml   # Redis + MLflow
```
