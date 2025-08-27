import importlib
import warnings

import numpy as np

from ironforge.converters.json_to_parquet import FeatureExtractor
from ironforge.data_engine import schemas


def test_feature_dims_without_htf():
    fx = FeatureExtractor(htf_processor=None)
    vec = fx.extract_node_features({}, {}, None)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == schemas.NFEATS_NODE == 45


def test_feature_dims_with_htf_values():
    # Any truthy object enables HTF path; provide expected keys
    fx = FeatureExtractor(htf_processor=object())
    htf = {
        "f45_sv_m15_z": 1.0,
        "f46_sv_h1_z": 2.0,
        "f47_barpos_m15": 0.25,
        "f48_barpos_h1": 0.75,
        "f49_dist_daily_mid": -0.4,
        "f50_htf_regime": 2,
    }
    vec = fx.extract_node_features({}, {}, htf)
    assert len(vec) == 51
    # Check a couple of mapped positions
    assert vec[45] == 1.0 and vec[50] == 2.0


def test_conversion_config_htf_default_off():
    from ironforge.converters.json_to_parquet import ConversionConfig

    cfg = ConversionConfig()
    assert cfg.htf_context_enabled is False


def test_single_source_version():
    from importlib import import_module

    from ironforge import __version__ as pkg_version
    from ironforge.reporting import __version__ as reporting_version

    core_version = import_module("ironforge.__version__").__version__
    assert pkg_version == core_version
    assert reporting_version == core_version


def test_cli_legacy_discovery_emits_deprecation(monkeypatch):
    # Simulate canonical import failing and legacy succeeding
    import ironforge.sdk.cli as cli

    def fake_import_module(name):
        class Dummy:
            def __init__(self, fn_name):
                setattr(self, fn_name, lambda cfg: 1)

        if name == "ironforge.learning.discovery_pipeline":
            raise ImportError
        if name in ("ironforge.learning.tgat_discovery", "ironforge.discovery.runner"):
            return Dummy("run_discovery")
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        cli.cmd_discover(object())
        assert any(isinstance(rec.message, DeprecationWarning) for rec in w)

