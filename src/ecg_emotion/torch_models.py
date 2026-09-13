"""Optional compact neural models.

PyTorch is deliberately optional so the feature baselines and unit tests remain usable on a clean
CPU-only Python environment.
"""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - exercised through the public factory
    torch = None


def build_tiny_cnn(num_classes: int = 4):
    """Build a small 1D CNN suitable for CPU experiments."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install the optional dependency with "
            '`pip install -e ".[deep-learning]"`'
        )
    return TinyCNN1D(num_classes=num_classes)


if torch is not None:

    class TinyCNN1D(torch.nn.Module):
        """Compact temporal convolutional classifier for fixed-length ECG windows."""

        def __init__(self, num_classes: int = 4) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
                torch.nn.BatchNorm1d(32),
                torch.nn.SiLU(),
                torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
                torch.nn.BatchNorm1d(64),
                torch.nn.SiLU(),
                torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm1d(128),
                torch.nn.SiLU(),
                torch.nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(128, 64),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(64, num_classes),
            )

        def forward(self, inputs):
            return self.classifier(self.encoder(inputs))

else:

    class TinyCNN1D:  # pragma: no cover - only used to provide a clear error
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Install the optional deep-learning dependency to use TinyCNN1D")

