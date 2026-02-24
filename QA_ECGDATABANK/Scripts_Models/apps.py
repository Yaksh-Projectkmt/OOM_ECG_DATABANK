# Scripts_Models/apps.py
import os
import importlib
from pathlib import Path
from django.apps import AppConfig


class Script_ModelsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Scripts_Models'

    def ready(self):
        if os.environ.get("RUN_MAIN") != "true":
            return

        base_dir = Path(__file__).resolve().parent

        self.load_all_python_scripts(base_dir)
        self.load_all_models(base_dir)

        print("Scripts_Models fully preloaded")

    # ----------------------------
    # LOAD PYTHON SCRIPTS
    # ----------------------------
    def load_all_python_scripts(self, base_dir):
        folders = [
            base_dir / "Scripts",
        ]

        EXCLUDE_DIRS = {
            "__pycache__",
            "detectron2",
            "data",
            "Models",
        }

        for folder in folders:
            for root, dirs, files in os.walk(folder):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

                for file in files:
                    if not file.endswith(".py") or file.startswith("__"):
                        continue

                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(base_dir)
                    module_path = ".".join(rel_path.with_suffix("").parts)

                    try:
                        importlib.import_module(f"{self.name}.{module_path}")
                        print(f"Loaded: {module_path}")
                    except Exception as e:
                        print(f"Skipped {module_path}: {e}")

    # ----------------------------
    # LOAD MODELS
    # ----------------------------
    def load_all_models(self, base_dir):
        from .model_registry import load_all_models

        model_dir = base_dir / "Model"
        load_all_models(model_dir)
