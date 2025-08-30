from __future__ import annotations
from typing import Any, Dict, List


class AttentionWeightProcessor:
    def extract_attention_matrices(self, embeddings: Dict[str, Any]) -> Any:
        return embeddings.get("attention_weights")

    def calculate_attention_scores(self, weights: Any) -> List[float]:
        if weights is None:
            return []
        # Placeholder: return mean weight per layer if available
        try:
            return [float(w.mean()) for w in weights]
        except Exception:
            return []

    def identify_key_relationships(self, attention_map: List[float]) -> List[Dict[str, Any]]:
        return [
            {"layer": i, "score": score}
            for i, score in enumerate(attention_map)
            if score >= 0
        ]
