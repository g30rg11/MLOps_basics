import math

import pytest
import torch

from mlops_homework.data import make_loader, prepare_wine
from mlops_homework.model import WineClassifier
from mlops_homework.training import evaluate, train_epoch


@pytest.mark.smoke
def test_training_updates_weights() -> None:
    torch.manual_seed(42)
    data = prepare_wine()
    loader = make_loader(
        data.x_train,
        data.y_train,
        batch_size=16,
        shuffle=False,
    )
    model = WineClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_epoch(model, loader, optimizer, max_batches=2)

    assert math.isfinite(loss)
    assert loss >= 0
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.integration
def test_training_and_evaluation_pipeline() -> None:
    torch.manual_seed(42)
    data = prepare_wine()
    train_loader = make_loader(data.x_train, data.y_train, batch_size=32)
    test_loader = make_loader(
        data.x_test,
        data.y_test,
        batch_size=16,
        shuffle=False,
    )
    model = WineClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loss = train_epoch(model, train_loader, optimizer)
    before_evaluation = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = evaluate(model, test_loader)

    assert math.isfinite(train_loss)
    assert train_loss >= 0
    assert math.isfinite(metrics["loss"])
    assert metrics["loss"] >= 0
    assert 0 <= metrics["accuracy"] <= 1

    with torch.no_grad():
        logits = model(data.x_test)
        expected_loss = torch.nn.functional.cross_entropy(logits, data.y_test).item()
        expected_accuracy = (logits.argmax(dim=1) == data.y_test).float().mean().item()

    assert metrics["loss"] == pytest.approx(expected_loss, rel=1e-5)
    assert metrics["accuracy"] == pytest.approx(expected_accuracy, abs=1e-7)

    for before, after in zip(before_evaluation, model.parameters(), strict=True):
        torch.testing.assert_close(before, after.detach(), rtol=0, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("max_batches", [0, -1])
def test_training_rejects_invalid_batch_limit(max_batches: int) -> None:
    data = prepare_wine()
    loader = make_loader(data.x_train, data.y_train)
    model = WineClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="max_batches must be positive"):
        train_epoch(model, loader, optimizer, max_batches=max_batches)


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_empty_loader_is_rejected(operation: str) -> None:
    dataset = torch.utils.data.TensorDataset(
        torch.empty(0, 13),
        torch.empty(0, dtype=torch.int64),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    model = WineClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="loader must not be empty"):
        if operation == "train":
            train_epoch(model, loader, optimizer)
        else:
            evaluate(model, loader)


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_nonfinite_loss_is_rejected(operation: str) -> None:
    dataset = torch.utils.data.TensorDataset(
        torch.full((4, 13), float("nan")),
        torch.zeros(4, dtype=torch.int64),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    model = WineClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.train()

    with pytest.raises(ValueError, match="loss must be finite"):
        if operation == "train":
            train_epoch(model, loader, optimizer)
        else:
            evaluate(model, loader)

    for old, new in zip(before, model.parameters(), strict=True):
        torch.testing.assert_close(old, new.detach(), rtol=0, atol=0)

    assert model.training
