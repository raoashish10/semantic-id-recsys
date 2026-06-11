"""Recommendation API routes.

POST /recommend
  Body: {"user_id": "u123", "session": ["item_id_1", ...], "top_k": 10}

  Request flow:
    1. Cache check        — Redis recs:{user_id}              → return immediately (<1ms)
    2. Cold start (LLM)   — session < COLD_START_THRESHOLD    → intent inference + ranked retrieval
       a. Check IntentCache (Redis, 300s TTL) — skip LLM on hit
       b. Call Ollama (200ms timeout) → IntentResult
       c. intent_based_recommend: prefix candidates + cosine re-rank by intent embedding
       d. On any failure (timeout, parse error, no results) → fall through to prefix fallback
    3. Cold start (fallback) — SRANDMEMBER on prefix:{c0}:{c1} Redis set (unranked)
    4. SASRec inference   — session ≥ 3 items (or cold start produced nothing)
    5. Write-back         — result stored in Redis            → background task (non-blocking)

  Response includes:
    cache_hit: bool          — true if served from precomputed recs
    cold_start_method: str   — "intent" | "prefix_fallback" | null (warm-user SASRec path)

  Environment variables:
    COLD_START_LLM_ENABLED  set to "false" to disable LLM path entirely (default: true)
                            useful for environments without Ollama

GET /health
"""

import os
from collections import Counter
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from serving.inference import beam_recommend, build_input
from serving.intent import IntentResult, infer_session_intent
from serving.intent_retrieval import intent_based_recommend
from serving.models import RecommendedItem, RecommendResponse

router = APIRouter()

COLD_START_THRESHOLD = 3
COLD_START_LLM_ENABLED = os.getenv("COLD_START_LLM_ENABLED", "true").lower() not in (
    "false", "0", "no"
)


class RecommendRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Used to look up / write precomputed recs")
    session: list[str] = Field(..., description="Ordered list of recently interacted item IDs")
    top_k: int = Field(default=10, ge=1, le=100)


def _prefix_recommend(session: list[str], store, top_k: int) -> list[RecommendedItem]:
    """Unranked cold-start fallback: random sample from the session's dominant (c0, c1) prefix bucket.

    Used when the LLM path is disabled or fails. No model inference, no embeddings —
    just SRANDMEMBER on a Redis set. Fast and always available as long as the prefix
    index was built during the pipeline's index step.
    """
    prefix_counts: Counter = Counter()
    session_set = set(session)

    for item_id in session:
        codes = store.get_codes(item_id)
        if codes is not None and len(codes) >= 2:
            prefix_counts[(codes[0], codes[1])] += 1

    if not prefix_counts:
        return []

    (c0, c1), _ = prefix_counts.most_common(1)[0]
    candidates = store.get_items_by_prefix(c0, c1, limit=top_k * 5)

    recs: list[RecommendedItem] = []
    for item_id in candidates:
        if item_id in session_set:
            continue
        codes = store.get_codes(item_id)
        if codes is None:
            continue
        recs.append(RecommendedItem(
            item_id=item_id,
            title=store.get_title(item_id),
            semantic_id=codes,
        ))
        if len(recs) >= top_k:
            break

    return recs


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(
    req: RecommendRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> RecommendResponse:
    state = request.app.state.recsys

    # ── 1. Cache check ────────────────────────────────────────────────────────
    if req.user_id:
        cached = state.store.get_user_recs(req.user_id)
        if cached is not None:
            return RecommendResponse(
                recommendations=[RecommendedItem(**r) for r in cached[: req.top_k]],
                session_length=len(req.session),
                cache_hit=True,
            )

    # ── 2 & 3. Cold start: session too short for meaningful SASRec attention ──
    if len(req.session) < COLD_START_THRESHOLD:
        recs: list[RecommendedItem] = []
        cold_start_method = "prefix_fallback"

        if COLD_START_LLM_ENABLED:
            intent_result: IntentResult | None = state.intent_cache.get(req.session)

            if intent_result is None:
                try:
                    item_titles = [state.store.get_title(iid) for iid in req.session]
                    item_codes = [state.store.get_codes(iid) or [] for iid in req.session]
                    intent_result = await infer_session_intent(item_titles, item_codes)
                    state.intent_cache.set(req.session, intent_result)
                except Exception:
                    intent_result = None

            if intent_result is not None:
                recs = intent_based_recommend(intent_result, req.session, state.store, req.top_k)
                if recs:
                    cold_start_method = "intent"

        # Silent fallback — no error surfaced to the caller
        if not recs:
            recs = _prefix_recommend(req.session, state.store, req.top_k)

        if recs:
            if req.user_id:
                background_tasks.add_task(
                    state.store.set_user_recs,
                    req.user_id,
                    [r.model_dump() for r in recs],
                )
            return RecommendResponse(
                recommendations=recs,
                session_length=len(req.session),
                cache_hit=False,
                cold_start_method=cold_start_method,
            )
        # Neither LLM nor prefix produced anything — fall through to SASRec

    # ── 4. Real-time SASRec inference ─────────────────────────────────────────
    inp, n_resolved = build_input(
        req.session, state.store, state.num_levels, state.model.max_len, state.device
    )
    if inp is None:
        raise HTTPException(status_code=422, detail="No session items found in catalog")

    recs = beam_recommend(state.model, state.store, inp, state.num_levels, req.top_k)

    # ── 5. Write back to Redis in background — don't block the response ───────
    if req.user_id and recs:
        background_tasks.add_task(
            state.store.set_user_recs,
            req.user_id,
            [r.model_dump() for r in recs],
        )

    return RecommendResponse(
        recommendations=recs,
        session_length=n_resolved,
        cache_hit=False,
    )


@router.get("/health")
async def health(request: Request) -> dict:
    state = request.app.state.recsys
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "redis": state.store.ping() if state.store else False,
    }
