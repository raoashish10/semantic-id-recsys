"""Locust load test for the RecSys serving API.

Simulates three user archetypes in a realistic traffic mix:
  - CachedUser (60%)  — known user_id with precomputed recs in Redis → cache hit path
  - WarmUser (30%)    — anonymous user with 4-5 session items → SASRec inference path
  - ColdUser (10%)    — new user with 1 session item → cold-start fallback path

Run (headless, 60s ramp, 50 concurrent users):
  locust -f tests/load/locustfile.py \
         --host http://localhost:8000 \
         --headless -u 50 -r 5 -t 60s \
         --html tests/load/report.html

Run (interactive browser UI):
  locust -f tests/load/locustfile.py --host http://localhost:8000
  # then open http://localhost:8089
"""

import random

from locust import HttpUser, between, task

# ── Sample data pulled from a pipeline run (replace with actual IDs) ──────────
# Warm items: must exist in the Redis catalog (run `make index` first)
CATALOG_SAMPLE = [
    "B00OXDWFG2", "B01N7A5AGF", "B010U82WRU", "B08VF373NX",
    "B07YHGZ37K", "B00W10JW1A", "B09SPPL7S4", "B09ZP8TK87",
    "B0B87SH493", "B08ZYKQ3D3", "B07SV4HTMC", "B08NJ5BTWG",
    "B081D8D5YV", "B007QG7G3U", "B08MCT8C27", "B0C4LR3NZT",
    "B08L3J4FB9", "B0722HKRHH", "B004N7B4IS", "B08XMHSM5B",
]

# Known user_ids that have precomputed recs (from `make precompute`)
KNOWN_USERS = [
    "AEJQTTVSBECATLYDADHXZ7OIZQZQ",
    "AED5N26K77SV4ZAJA4VNCNGELXVA",
]


class CachedUser(HttpUser):
    """Simulates a returning user whose recs are precomputed in Redis.
    Expected latency: <5ms (pure Redis GET, no model inference).
    """
    weight = 6
    wait_time = between(0.1, 0.5)

    @task
    def get_cached_recs(self):
        user_id = random.choice(KNOWN_USERS)
        with self.client.post(
            "/recommend",
            json={"user_id": user_id, "session": [], "top_k": 10},
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
            elif not resp.json().get("recommendations"):
                resp.failure("Empty recommendations")
            elif not resp.json().get("cache_hit"):
                resp.failure("Expected cache hit but got miss")


class WarmUser(HttpUser):
    """Simulates a user with 4-5 recent items triggering SASRec inference.
    Expected latency: 20-150ms (depends on beam search + Redis lookups).
    """
    weight = 3
    wait_time = between(0.5, 2.0)

    @task
    def get_warm_recs(self):
        session = random.sample(CATALOG_SAMPLE, k=random.randint(4, 5))
        with self.client.post(
            "/recommend",
            json={"session": session, "top_k": 10},
            catch_response=True,
        ) as resp:
            if resp.status_code not in (200, 422):
                resp.failure(f"Status {resp.status_code}")
            elif resp.status_code == 200:
                body = resp.json()
                if body.get("cache_hit"):
                    resp.success()
                elif body.get("cold_start_method") is not None:
                    resp.failure("Expected warm path, got cold start")


class ColdUser(HttpUser):
    """Simulates a new user with a single item — cold-start prefix fallback.
    Expected latency: 5-30ms (SRANDMEMBER Redis lookup, no inference).
    """
    weight = 1
    wait_time = between(1.0, 3.0)

    @task
    def get_cold_recs(self):
        session = [random.choice(CATALOG_SAMPLE)]
        with self.client.post(
            "/recommend",
            json={"user_id": f"new_user_{random.randint(1, 99999)}", "session": session, "top_k": 10},
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
            else:
                body = resp.json()
                if body.get("cache_hit"):
                    resp.success()
                elif body.get("cold_start_method") not in ("prefix_fallback", "intent", None):
                    resp.failure(f"Unexpected cold_start_method: {body.get('cold_start_method')}")
