try:
    import torch
except ImportError:
    import sys

    print("pytorch required for these utilities")
    sys.exit(1)
