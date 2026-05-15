import os
import numpy as np
os.makedirs("models", exist_ok=True)

# ── Dummy ONNX model ──────────────────────────────
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

print("Creating dummy ONNX model...")
X = np.array([[1,2],[3,4],[5,6],[7,8]], dtype=np.float32)
y = np.array([0, 0, 1, 1])
clf = LogisticRegression().fit(X, y)

initial_type = [("float_input", FloatTensorType([None, 2]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)

with open("models/echo.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("✅ models/echo.onnx created!")

# ── Dummy TorchScript model ───────────────────────
import torch

print("Creating dummy PyTorch model...")

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2.0

model = SimpleModel()
scripted = torch.jit.script(model)
scripted.save("models/classifier.pt")
print("✅ models/classifier.pt created!")

print("\n🎉 All dummy models ready!")