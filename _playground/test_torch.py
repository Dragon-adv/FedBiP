#!/usr/bin/env python
import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("SUCCESS: Torch is working!")
except ImportError as e:
    print("ERROR: Torch not found:", e)

