from itertools import islice

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
) -> float:
    """Train for one epoch or a limited number of batches."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    batches = iter(loader)
    if max_batches is not None:
        batches = islice(batches, max_batches)

    for features, labels in batches:
        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss).item():
            raise ValueError("Training loss must be finite")

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader must not be empty")

    return total_loss / total_samples


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """Calculate mean loss and accuracy without updating weights."""
    was_training = model.training
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    try:
        with torch.no_grad():
            for features, labels in loader:
                logits = model(features)
                loss = criterion(logits, labels)

                if not torch.isfinite(loss).item():
                    raise ValueError("Evaluation loss must be finite")

                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += labels.shape[0]
    finally:
        model.train(was_training)

    if total_samples == 0:
        raise ValueError("Evaluation loader must not be empty")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
