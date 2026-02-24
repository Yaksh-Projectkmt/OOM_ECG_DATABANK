from pathlib import Path
import tensorflow as tf
import torch
from collections import OrderedDict

MODEL_STORE = {}

def load_all_models(model_dir):
    for file in Path(model_dir).glob("*"):
        try:
            # ----------------------------
            # TFLITE MODELS
            # ----------------------------
            if file.suffix == ".tflite":
                interpreter = tf.lite.Interpreter(model_path=str(file))
                interpreter.allocate_tensors()
                MODEL_STORE[file.name] = interpreter
                print(f"Loaded TFLite: {file.name}")

            # ----------------------------
            # PYTORCH MODELS
            # ----------------------------
            elif file.suffix == ".pth":
                obj = torch.load(file, map_location="cpu")

                # FULL MODEL
                if hasattr(obj, "eval"):
                    obj.eval()
                    MODEL_STORE[file.name] = obj
                    print(f"Loaded Torch model: {file.name}")

                # STATE_DICT ONLY
                elif isinstance(obj, OrderedDict):
                    MODEL_STORE[file.name] = obj
                    print(f"Loaded Torch state_dict: {file.name}")

                else:
                    print(f"Unknown torch object: {file.name}")

        except Exception as e:
            print(f"Failed to load {file.name}: {e}")
