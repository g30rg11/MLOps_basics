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
