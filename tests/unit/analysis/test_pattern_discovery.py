import importlib.util
from pathlib import Path

from sklearn.datasets import make_blobs

SPEC = importlib.util.spec_from_file_location(
    "pattern_discovery",
    Path(__file__).resolve().parents[3] / "ironforge" / "analysis" / "pattern_discovery.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

discover_patterns = MODULE.discover_patterns


def test_discover_patterns_shapes() -> None:
    X, _ = make_blobs(n_samples=50, n_features=6, centers=3, random_state=0)
    result = discover_patterns(X, n_components=2, cluster_min_size=5)

    assert result.embeddings.shape == (50, 2)
    assert result.labels.shape == (50,)
    assert result.mutual_information.shape == (6, 6)
    assert result.precision.shape == (6, 6)
    clusters = set(result.labels)
    clusters.discard(-1)
    assert len(clusters) >= 1
