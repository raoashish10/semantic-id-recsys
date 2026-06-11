"""Shared application state: SASRec model + item store + intent cache loaded once at startup."""

import json
from pathlib import Path

import torch

from offline.sasrec.model import SASRec
from serving.intent_cache import IntentCache
from serving.store.redis_client import ItemStore

SASREC_WEIGHTS = Path("artifacts/sasrec/model.pt")
SASREC_CFG = Path("artifacts/sasrec/config.json")


class AppState:
    model: SASRec | None = None
    store: ItemStore | None = None
    intent_cache: IntentCache | None = None
    num_codes: int = 256
    num_levels: int = 3
    device: torch.device = torch.device("cpu")

    def load(self) -> None:
        with SASREC_CFG.open() as f:
            cfg = json.load(f)

        self.num_codes = cfg["num_codes"]
        self.num_levels = cfg["num_levels"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SASRec(
            num_codes=cfg["num_codes"],
            num_levels=cfg["num_levels"],
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            max_len=cfg["max_len"],
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(torch.load(SASREC_WEIGHTS, map_location=self.device))
        self.model.eval()

        self.store = ItemStore()
        if not self.store.ping():
            raise RuntimeError("Redis not reachable — is `make up` running?")

        # Reuse the store's Redis connection — no new connection created
        self.intent_cache = IntentCache(self.store.redis_client)
