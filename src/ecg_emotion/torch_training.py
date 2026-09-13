"""CPU-friendly training loop for the optional Tiny 1D-CNN."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_npz, split_by_subject
from .evaluation import classification_metrics
from .preprocessing import normalize_windows
from .torch_models import build_tiny_cnn


def train_tiny_cnn(
    data_path: str | Path,
    output_dir: str | Path,
    sample_rate: float = 256.0,
    batch_size: int = 128,
    max_epochs: int = 80,
    early_stopping_patience: int = 10,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    num_threads: int = 4,
) -> dict[str, Any]:
    """Train and save a TinyCNN1D using deterministic subject-level splits."""

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError(
            'Install PyTorch with `pip install -e ".[deep-learning]"` before using this command.'
        ) from error

    torch.manual_seed(random_seed)
    torch.set_num_threads(max(1, num_threads))
    started = time.perf_counter()

    dataset = load_npz(data_path)
    split_indexes = split_by_subject(dataset, random_seed=random_seed)
    signals = normalize_windows(dataset.signals)
    num_classes = int(np.max(dataset.labels)) + 1

    def tensor_dataset(indexes: np.ndarray) -> TensorDataset:
        signal_tensor = torch.from_numpy(signals[indexes]).unsqueeze(1)
        label_tensor = torch.from_numpy(dataset.labels[indexes]).long()
        return TensorDataset(signal_tensor, label_tensor)

    train_loader = DataLoader(
        tensor_dataset(split_indexes["train"]),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        tensor_dataset(split_indexes["validation"]), batch_size=batch_size, shuffle=False
    )
    model = build_tiny_cnn(num_classes=num_classes)
    class_counts = np.bincount(dataset.labels[split_indexes["train"]], minlength=num_classes)
    weights = len(split_indexes["train"]) / np.maximum(class_counts, 1)
    class_weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_state: dict[str, Any] | None = None
    best_validation_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch_signals, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_signals)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_labels)
            sample_count += len(batch_labels)

        validation_metrics = _evaluate(model, validation_loader, num_classes)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(sample_count, 1),
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(epoch_record)

        if validation_metrics["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    split_metrics = {
        split_name: _evaluate(
            model,
            DataLoader(tensor_dataset(indexes), batch_size=batch_size),
            num_classes,
        )
        for split_name, indexes in split_indexes.items()
    }
    split_subjects = {
        split_name: sorted({str(subject) for subject in dataset.subjects[indexes]})
        for split_name, indexes in split_indexes.items()
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": num_classes,
            "signal_length": dataset.signal_length,
            "sample_rate": sample_rate,
        },
        output_dir / "tiny_cnn.pt",
    )
    result = {
        "model_name": "tiny-cnn-1d",
        "data_path": str(Path(data_path)),
        "num_samples": dataset.num_samples,
        "num_subjects": dataset.num_subjects,
        "signal_length": dataset.signal_length,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "random_seed": random_seed,
        "epochs_completed": len(history),
        "split_subjects": split_subjects,
        "metrics": split_metrics,
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 4),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _evaluate(model, loader, num_classes: int) -> dict[str, Any]:
    import torch

    model.eval()
    true_labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for signals, labels in loader:
            logits = model(signals)
            true_labels.extend(labels.numpy().tolist())
            predictions.extend(torch.argmax(logits, dim=1).numpy().tolist())
    return classification_metrics(
        np.asarray(true_labels),
        np.asarray(predictions),
        labels=list(range(num_classes)),
    )
