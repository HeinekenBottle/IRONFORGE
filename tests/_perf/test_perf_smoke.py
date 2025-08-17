import resource
import time

BUDGET_S = 2.0
BUDGET_MB = 500


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def test_perf_smoke() -> None:
    start = time.time()
    m0 = mem_mb()
    # minimal work: import + tiny noop
    import ironforge  # noqa: F401

    elapsed = time.time() - start
    delta_mb = mem_mb() - m0
    assert elapsed <= BUDGET_S
    assert delta_mb <= BUDGET_MB
