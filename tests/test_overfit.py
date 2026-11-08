import pytest
import torch

from mlops_homework.data import make_loader, prepare_wine
from mlops_homework.model import WineClassifier
from mlops_homework.training import evaluate, train_epoch


@pytest.mark.slow
def test_model_overfits_small_batch() -> None:
    torch.manual_seed(42)
    data = prepare_wine()

    # Take four training samples from each class.
    indices = torch.cat(
        [torch.where(data.y_train == label)[0][:4] for label in range(3)]
    )
    features = data.x_train[indices]
    labels = data.y_train[indices]

    loader = make_loader(
        features,
        labels,
        batch_size=12,
        shuffle=False,
    )
    model = WineClassifier(hidden_features=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    initial = evaluate(model, loader)

    for _ in range(200):
        train_epoch(model, loader, optimizer)

    final = evaluate(model, loader)

    assert final["loss"] < initial["loss"]
    assert final["loss"] < 0.05
    assert final["accuracy"] == pytest.approx(1.0)
