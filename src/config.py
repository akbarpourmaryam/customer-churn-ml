"""YAML configuration loading with project-relative path resolution."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"


def load_config(path=DEFAULT_CONFIG_PATH):
    """Load training configuration and resolve configured paths."""

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("data_path", "artifacts_dir", "reports_dir", "mlruns_dir"):
        value = Path(config["paths"][key])
        config["paths"][key] = value if value.is_absolute() else PROJECT_ROOT / value
    return config
