"""CPU-friendly training loop for the optional Tiny 1D-CNN."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data import load_npz, split_by_subject
from .evaluation import classification_metrics
from .preprocessing import normalize_windows
from .torch_models import build_tiny_cnn, build_tiny_lstm


def cross_validate_tiny_cnn(
    data_path: str | Path,
    output_dir: str | Path,
    sample_rate: float = 256.0,
    batch_size: int = 128,
    max_epochs: int = 30,
    early_stopping_patience: int = 6,
    learning_rate: float = 5e-4,
    num_folds: int = 5,
    random_seed: int = 42,
    num_threads: int = 4,
    use_augmentation: bool = True,
    model_factory=build_tiny_cnn,
    model_name: str = "tiny-cnn-1d-v2",
) -> dict[str, Any]:
    """Evaluate TinyCNN with nested subject-level validation folds.

    The outer folds estimate performance on unseen participants.  A validation
    participant subset is drawn only from each outer training partition for
    early stopping, so the outer test subjects never influence optimization.
    """

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError(
            'Install PyTorch with `pip install -e "[deep-learning]"` before using this command.'
        ) from error

    started = time.perf_counter()
    torch.set_num_threads(max(1, num_threads))
    dataset = load_npz(data_path)
    if dataset.num_subjects < num_folds + 1:
        raise ValueError("cross-validation needs more subjects than outer folds")

    signals = normalize_windows(dataset.signals)
    labels = sorted(np.unique(dataset.labels).astype(int).tolist())
    num_classes = len(labels)
    splitter = StratifiedGroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_seed,
    )
    fold_results: list[dict[str, Any]] = []
    subject_results: list[dict[str, Any]] = []

    for fold_index, (outer_train, test_indexes) in enumerate(
        splitter.split(signals, dataset.labels, groups=dataset.subjects),
        start=1,
    ):
        train_indexes, validation_indexes = _split_validation_subjects(
            outer_train,
            dataset.subjects,
            random_seed + fold_index,
        )
        fold_started = time.perf_counter()
        model, history = _fit_tiny_cnn_fold(
            signals=signals,
            labels_array=dataset.labels,
            train_indexes=train_indexes,
            validation_indexes=validation_indexes,
            num_classes=num_classes,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            learning_rate=learning_rate,
            random_seed=random_seed + fold_index,
            use_augmentation=use_augmentation,
            model_factory=model_factory,
        )

        def loader(indexes: np.ndarray) -> DataLoader:
            return DataLoader(
                TensorDataset(
                    torch.from_numpy(signals[indexes]).unsqueeze(1),
                    torch.from_numpy(dataset.labels[indexes]).long(),
                ),
                batch_size=batch_size,
                shuffle=False,
            )

        split_metrics = {
            split_name: _evaluate(model, loader(indexes), num_classes)
            for split_name, indexes in {
                "train": train_indexes,
                "validation": validation_indexes,
                "test": test_indexes,
            }.items()
        }
        test_subjects = sorted({str(subject) for subject in dataset.subjects[test_indexes]})
        fold_results.append(
            {
                "fold": fold_index,
                "train_subjects": sorted(
                    {str(subject) for subject in dataset.subjects[train_indexes]}
                ),
                "validation_subjects": sorted(
                    {str(subject) for subject in dataset.subjects[validation_indexes]}
                ),
                "test_subjects": test_subjects,
                "epochs_completed": len(history),
                "metrics": split_metrics["test"],
                "train_metrics": split_metrics["train"],
                "validation_metrics": split_metrics["validation"],
                "training_seconds": round(time.perf_counter() - fold_started, 4),
            }
        )
        for subject in test_subjects:
            subject_indexes = test_indexes[dataset.subjects[test_indexes] == subject]
            subject_results.append(
                {
                    "fold": fold_index,
                    "subject": subject,
                    "metrics": _evaluate(model, loader(subject_indexes), num_classes),
                }
            )

    scalar_metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    aggregate = {
        metric: {
            "mean": float(np.mean([fold["metrics"][metric] for fold in fold_results])),
            "std": float(np.std([fold["metrics"][metric] for fold in fold_results])),
        }
        for metric in scalar_metrics
    }
    result = {
        "model_name": model_name,
        "data_path": str(Path(data_path)),
        "num_samples": dataset.num_samples,
        "num_subjects": dataset.num_subjects,
        "num_folds": num_folds,
        "split_strategy": "StratifiedGroupKFold with inner validation subjects",
        "signal_length": dataset.signal_length,
        "sample_rate": sample_rate,
        "parameter_count": sum(
            parameter.numel() for parameter in model_factory(num_classes).parameters()
        ),
        "random_seed": random_seed,
        "augmentation": use_augmentation,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "learning_rate": learning_rate,
        "folds": fold_results,
        "per_subject": subject_results,
        "aggregate": aggregate,
        "evaluation_seconds": round(time.perf_counter() - started, 4),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cross_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def cross_validate_tiny_lstm(
    data_path: str | Path,
    output_dir: str | Path,
    sample_rate: float = 256.0,
    batch_size: int = 128,
    max_epochs: int = 30,
    early_stopping_patience: int = 6,
    learning_rate: float = 5e-4,
    num_folds: int = 5,
    random_seed: int = 42,
    num_threads: int = 4,
    use_augmentation: bool = True,
) -> dict[str, Any]:
    """Evaluate the compact convolutional LSTM under grouped cross-validation."""

    return cross_validate_tiny_cnn(
        data_path=data_path,
        output_dir=output_dir,
        sample_rate=sample_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        num_folds=num_folds,
        random_seed=random_seed,
        num_threads=num_threads,
        use_augmentation=use_augmentation,
        model_factory=build_tiny_lstm,
        model_name="tiny-lstm-1d-v1",
    )


def train_tiny_cnn(
    data_path: str | Path,
    output_dir: str | Path,
    sample_rate: float = 256.0,
    batch_size: int = 128,
    max_epochs: int = 80,
    early_stopping_patience: int = 10,
    learning_rate: float = 5e-4,
    random_seed: int = 42,
    num_threads: int = 4,
    use_augmentation: bool = True,
    model_factory=build_tiny_cnn,
    model_name: str = "tiny-cnn-1d-v2",
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
    model = model_factory(num_classes=num_classes)
    class_counts = np.bincount(dataset.labels[split_indexes["train"]], minlength=num_classes)
    weights = len(split_indexes["train"]) / np.maximum(class_counts, 1)
    class_weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    best_state: dict[str, Any] | None = None
    best_validation_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch_signals, batch_labels in train_loader:
            if use_augmentation:
                batch_signals = _augment_batch(batch_signals)
            optimizer.zero_grad()
            logits = model(batch_signals)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_labels)
            sample_count += len(batch_labels)

        validation_metrics = _evaluate(model, validation_loader, num_classes)
        scheduler.step(validation_metrics["macro_f1"])
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(sample_count, 1),
            "validation_macro_f1": validation_metrics["macro_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
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
        "model_name": model_name,
        "data_path": str(Path(data_path)),
        "num_samples": dataset.num_samples,
        "num_subjects": dataset.num_subjects,
        "signal_length": dataset.signal_length,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "random_seed": random_seed,
        "augmentation": use_augmentation,
        "epochs_completed": len(history),
        "split_subjects": split_subjects,
        "metrics": split_metrics,
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 4),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_tiny_lstm(
    data_path: str | Path,
    output_dir: str | Path,
    sample_rate: float = 256.0,
    batch_size: int = 128,
    max_epochs: int = 80,
    early_stopping_patience: int = 10,
    learning_rate: float = 5e-4,
    random_seed: int = 42,
    num_threads: int = 4,
    use_augmentation: bool = True,
) -> dict[str, Any]:
    """Train and save the compact convolutional LSTM on a locked split."""

    return train_tiny_cnn(
        data_path=data_path,
        output_dir=output_dir,
        sample_rate=sample_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        random_seed=random_seed,
        num_threads=num_threads,
        use_augmentation=use_augmentation,
        model_factory=build_tiny_lstm,
        model_name="tiny-lstm-1d-v1",
    )


def _augment_batch(signals):
    """Apply inexpensive label-preserving augmentation to normalized ECG windows."""

    import torch

    gains = torch.empty((len(signals), 1, 1)).uniform_(0.9, 1.1)
    augmented = signals * gains + 0.015 * torch.randn_like(signals)
    max_shift = max(1, signals.shape[-1] // 20)
    shift = int(torch.randint(-max_shift, max_shift + 1, size=(1,)).item())
    return torch.roll(augmented, shifts=shift, dims=-1)


def _split_validation_subjects(
    indexes: np.ndarray,
    subjects: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out validation participants from one outer training partition."""

    unique_subjects = np.unique(subjects[indexes])
    if len(unique_subjects) < 3:
        raise ValueError("each outer training partition needs at least three subjects")
    shuffled = unique_subjects.copy()
    np.random.default_rng(random_seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * 0.2)))
    validation_count = min(validation_count, len(shuffled) - 2)
    validation_subjects = shuffled[:validation_count]
    validation_mask = np.isin(subjects[indexes], validation_subjects)
    return indexes[~validation_mask], indexes[validation_mask]


def _fit_tiny_cnn_fold(
    signals: np.ndarray,
    labels_array: np.ndarray,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    num_classes: int,
    batch_size: int,
    max_epochs: int,
    early_stopping_patience: int,
    learning_rate: float,
    random_seed: int,
    use_augmentation: bool,
    model_factory=build_tiny_cnn,
):
    """Fit one CNN fold and restore its validation-selected checkpoint."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_seed)

    def make_loader(indexes: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(signals[indexes]).unsqueeze(1),
            torch.from_numpy(labels_array[indexes]).long(),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_indexes, shuffle=True)
    validation_loader = make_loader(validation_indexes, shuffle=False)
    model = model_factory(num_classes=num_classes)
    class_counts = np.bincount(labels_array[train_indexes], minlength=num_classes)
    weights = len(train_indexes) / np.maximum(class_counts, 1)
    class_weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )
    best_state = None
    best_validation_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch_signals, batch_labels in train_loader:
            if use_augmentation:
                batch_signals = _augment_batch(batch_signals)
            optimizer.zero_grad()
            loss = loss_function(model(batch_signals), batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_labels)
            sample_count += len(batch_labels)

        validation_metrics = _evaluate(model, validation_loader, num_classes)
        scheduler.step(validation_metrics["macro_f1"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / max(sample_count, 1),
                "validation_macro_f1": validation_metrics["macro_f1"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
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
    return model, history


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
