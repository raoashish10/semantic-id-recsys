"""Shared Pydantic response models used by both the API routes and precompute script."""

from pydantic import BaseModel, Field
from typing import Optional


class RecommendedItem(BaseModel):
    item_id: str
    title: str
    semantic_id: list[int]


class RecommendResponse(BaseModel):
    recommendations: list[RecommendedItem]
    session_length: int
    cache_hit: bool
    cold_start_method: str | None = None
