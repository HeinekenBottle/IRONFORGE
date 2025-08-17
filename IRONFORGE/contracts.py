from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiscoveryResult:
    shard_id: str
    embeddings_path: str
    patterns_path: str
    meta: dict[str, Any]
