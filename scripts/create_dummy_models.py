#!/usr/bin/env python3
"""
Generate dummy models for testing on Linux/GPU environment.
Run: python3 scripts/create_dummy_models.py
"""
import os
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def create_onnx_model():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    print("Creating dummy ONNX model...")
    X = np.array([[1,2],[3,4],[5,6],[7,8]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression().fit(X, y)

    initial_type = [("float_input", FloatTensorType([None, 2]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)

    path = MODELS_DIR / "echo.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ {path} created!")

def create_pytorch_model():
    import torch

    print("Creating dummy PyTorch model...")

    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    model = SimpleModel()
    scripted = torch.jit.script(model)
    path = MODELS_DIR / "classifier.pt"
    scripted.save(str(path))
    print(f"✅ {path} created!")

if __name__ == "__main__":
    create_onnx_model()
    create_pytorch_model()
    print("\n🎉 All dummy models ready!")
    print(f"   Location: {MODELS_DIR}")