import pytest

from ecg_emotion.torch_models import build_tiny_cnn


def test_tiny_cnn_has_clear_optional_dependency_error() -> None:
    try:
        import torch as torch_module
    except ImportError:
        with pytest.raises(RuntimeError, match="PyTorch"):
            build_tiny_cnn()
    else:
        model = build_tiny_cnn(num_classes=4)
        output = model(torch_module.randn(2, 1, 512))
        assert output.shape == (2, 4)
        assert sum(parameter.numel() for parameter in model.parameters()) < 100_000
