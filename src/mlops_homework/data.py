from dataclasses import dataclass

import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class WineData:
    x_train: Tensor
    x_test: Tensor
    y_train: Tensor
    y_test: Tensor
    train_indices: Tensor
    test_indices: Tensor
    mean: Tensor
    scale: Tensor


def prepare_wine(test_size: float = 0.2, seed: int = 42) -> WineData:
    """Split Wine data and fit normalization on training samples only."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    dataset = load_wine()
    features = torch.tensor(dataset.data, dtype=torch.float64)
    labels = torch.tensor(dataset.target, dtype=torch.int64)

    train_ids, test_ids = train_test_split(
        list(range(len(labels))),
        test_size=test_size,
        random_state=seed,
        stratify=dataset.target,
    )

    train_indices = torch.tensor(train_ids, dtype=torch.int64)
    test_indices = torch.tensor(test_ids, dtype=torch.int64)

    raw_train = features[train_indices]
    mean = raw_train.mean(dim=0)
    scale = raw_train.std(dim=0, correction=0)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    return WineData(
        x_train=((raw_train - mean) / scale).float(),
        x_test=((features[test_indices] - mean) / scale).float(),
        y_train=labels[train_indices],
        y_test=labels[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        mean=mean,
        scale=scale,
    )


def make_loader(
    features: Tensor,
    labels: Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Create reproducible batches without dropping the final batch."""
    if features.ndim != 2 or features.shape[1] != 13:
        raise ValueError("features must have shape (N, 13)")
    if labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("labels must have shape (N,) matching features")
    if len(features) == 0:
        raise ValueError("Dataset must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")
    if labels.dtype != torch.int64:
        raise ValueError("labels must have dtype int64")
    if ((labels < 0) | (labels > 2)).any():
        raise ValueError("labels must be class indices 0, 1 or 2")

    dataset = TensorDataset(features.float(), labels)
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
