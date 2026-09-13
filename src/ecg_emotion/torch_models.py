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

    class StatisticsPooling(torch.nn.Module):
        """Pool temporal feature maps with both mean and standard deviation."""

        def forward(self, inputs):
            mean = inputs.mean(dim=-1)
            std = inputs.std(dim=-1, unbiased=False)
            return torch.cat((mean, std), dim=1)

    class TinyCNN1D(torch.nn.Module):
        """Compact subject-robust temporal classifier for fixed-length ECG windows."""

        def __init__(self, num_classes: int = 4) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7, bias=False),
                torch.nn.GroupNorm(4, 16),
                torch.nn.SiLU(),
                torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                torch.nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4, bias=False),
                torch.nn.GroupNorm(8, 32),
                torch.nn.SiLU(),
                torch.nn.Conv1d(32, 32, kernel_size=7, padding=3, groups=32, bias=False),
                torch.nn.Conv1d(32, 64, kernel_size=1, stride=2, bias=False),
                torch.nn.GroupNorm(8, 64),
                torch.nn.SiLU(),
                torch.nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2, bias=False),
                torch.nn.GroupNorm(8, 96),
                torch.nn.SiLU(),
            )
            self.pool = StatisticsPooling()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(192, 96),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=0.25),
                torch.nn.Linear(96, num_classes),
            )

        def forward(self, inputs):
            return self.classifier(self.pool(self.encoder(inputs)))

else:

    class TinyCNN1D:  # pragma: no cover - only used to provide a clear error
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Install the optional deep-learning dependency to use TinyCNN1D")
