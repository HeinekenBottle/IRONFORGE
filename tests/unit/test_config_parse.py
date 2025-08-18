from ironforge.sdk.config import load_cfg


def test_load_cfg(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        """
paths:
  shards_dir: data/shards/test
  out_dir: runs/test
loader:
  fanouts: [5, 5]
  batch_size: 1024
confluence:
  threshold: 70
        """
    )
    cfg = load_cfg(str(cfg_file))
    assert cfg.paths.shards_dir == "data/shards/test"
    assert cfg.loader.fanouts == (5, 5)
    assert cfg.confluence.threshold == 70
