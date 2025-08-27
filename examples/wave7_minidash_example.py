from ironforge.api import load_config
from ironforge.sdk.cli import cmd_report

if __name__ == "__main__":
    cfg = load_config("configs/dev.yml")
    cmd_report(cfg)

