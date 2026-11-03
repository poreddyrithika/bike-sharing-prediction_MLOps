import json
import os
from pathlib import Path
from typing import Any

import joblib
import yaml


class ModelLoader:
    """Loads the model + matching preprocessor from the deployment bundle (deployment/)."""

    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path
        self.deployment_dir = Path(os.getenv("DEPLOYMENT_DIR", self.project_root / "deployment"))
        self.config: dict[str, Any] = {}
        self.model = None
        self.preprocessor = None
        self.model_uri = None
        self.model_name = None
        self.model_version = None
        self.run_id = None
        self.metadata: dict[str, Any] = {}

    def load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def load_deployment_bundle(self) -> None:
        model_path = self.deployment_dir / "model.pkl"
        preprocessor_path = self.deployment_dir / "preprocessor.pkl"
        metadata_path = self.deployment_dir / "model_metadata.json"
        for path in (model_path, preprocessor_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"Deployment artifact not found: {path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.model_name = self.metadata.get("model_name")
        self.model_version = self.metadata.get("model_version")
        self.run_id = self.metadata.get("run_id")
        self.model_uri = str(model_path)

        trained_with = (self.metadata.get("runtime") or {}).get("scikit_learn")
        try:
            import sklearn
            if trained_with and trained_with != sklearn.__version__:
                print(f"[WARN] Model trained with scikit-learn {trained_with} but {sklearn.__version__} is installed.")
        except ImportError:
            pass

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print(f"[SUCCESS] Loaded {self.model_name} v{self.model_version} (run {self.run_id})")

    def load(self) -> None:
        print("[INFO] Initializing ModelLoader...")
        self.load_config()
        self.load_deployment_bundle()

    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None


if __name__ == "__main__":
    loader = ModelLoader()
    loader.load()
    print("Ready:", loader.is_ready(), "| version:", loader.model_version)
