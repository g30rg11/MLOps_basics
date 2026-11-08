import pytest
import torch
from sklearn.datasets import load_wine

from mlops_homework.data import make_loader, prepare_wine


@pytest.mark.unit
def test_split_and_scaling() -> None:
    data = prepare_wine(test_size=0.2, seed=42)

    train_ids = set(data.train_indices.tolist())
    test_ids = set(data.test_indices.tolist())

    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(range(178))
    assert len(train_ids) == 142
    assert len(test_ids) == 36

    assert data.x_train.shape == (142, 13)
    assert data.x_test.shape == (36, 13)
    assert set(data.y_train.tolist()) == {0, 1, 2}
    assert set(data.y_test.tolist()) == {0, 1, 2}

    raw = torch.tensor(load_wine().data, dtype=torch.float64)
    raw_train = raw[data.train_indices]
    expected_mean = raw_train.mean(dim=0)
    expected_scale = raw_train.std(dim=0, correction=0)

    torch.testing.assert_close(data.mean, expected_mean)
    torch.testing.assert_close(data.scale, expected_scale)

    expected_train = ((raw_train - expected_mean) / expected_scale).float()
    expected_test = ((raw[data.test_indices] - expected_mean) / expected_scale).float()

    torch.testing.assert_close(data.x_train, expected_train)
    torch.testing.assert_close(data.x_test, expected_test)


@pytest.mark.unit
def test_loader_batches_and_types() -> None:
    data = prepare_wine(test_size=0.2, seed=42)
    loader = make_loader(
        data.x_train,
        data.y_train,
        batch_size=32,
        shuffle=False,
    )

    batches = list(loader)

    assert [x.shape[0] for x, _ in batches] == [32, 32, 32, 32, 14]

    for features, labels in batches:
        assert features.shape == (len(labels), 13)
        assert features.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert torch.isfinite(features).all()

    all_features = torch.cat([x for x, _ in batches])
    all_labels = torch.cat([y for _, y in batches])

    torch.testing.assert_close(all_features, data.x_train)
    torch.testing.assert_close(all_labels, data.y_train)
