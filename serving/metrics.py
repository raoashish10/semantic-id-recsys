"""Prometheus metrics for the recommendation serving API.

Metrics
-------
  recsys_requests_total{path_type}       — Counter: total requests by serving path
  recsys_request_latency_seconds{path_type} — Histogram: end-to-end latency by path
  recsys_cache_hit_total                 — Counter: requests served from Redis cache
  recsys_cold_start_total{method}        — Counter: cold-start requests by method
                                           (method = "intent" | "prefix_fallback")

path_type values
  "cache_hit"          — served from precomputed Redis key
  "cold_start"         — session < 3 items
  "warm"               — real-time SASRec inference

Exposed at GET /metrics (text/plain Prometheus format).
"""

from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry(auto_describe=True)

REQUEST_COUNT = Counter(
    "recsys_requests_total",
    "Total recommendation requests by serving path",
    ["path_type"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "recsys_request_latency_seconds",
    "End-to-end request latency by serving path",
    ["path_type"],
    buckets=(0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.5),
    registry=REGISTRY,
)

CACHE_HIT_COUNT = Counter(
    "recsys_cache_hit_total",
    "Requests served from the precomputed Redis cache",
    registry=REGISTRY,
)

COLD_START_COUNT = Counter(
    "recsys_cold_start_total",
    "Cold-start requests by fallback method",
    ["method"],
    registry=REGISTRY,
)
