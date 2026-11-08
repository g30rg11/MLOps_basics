import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_homework.model import WineClassifier


@pytest.mark.property
@settings(max_examples=20, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=32),
    hidden_features=st.integers(min_value=1, max_value=32),
    value=st.floats(
        min_value=-10,
        max_value=10,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)
def test_model_shape_and_finite_outputs(
    batch_size: int,
    hidden_features: int,
    value: float,
) -> None:
    torch.manual_seed(42)
    model = WineClassifier(hidden_features=hidden_features)
    inputs = torch.full((batch_size, 13), value, dtype=torch.float32)

    logits = model(inputs)

    assert logits.shape == (batch_size, 3)
    assert torch.isfinite(logits).all()


@pytest.mark.property
@settings(max_examples=20, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=32),
    label=st.integers(min_value=0, max_value=2),
)
def test_classification_loss_is_finite_and_nonnegative(
    batch_size: int,
    label: int,
) -> None:
    torch.manual_seed(42)
    model = WineClassifier()
    inputs = torch.randn(batch_size, 13)
    labels = torch.full((batch_size,), label, dtype=torch.int64)

    loss = torch.nn.functional.cross_entropy(model(inputs), labels)

    assert torch.isfinite(loss).item()
    assert loss.item() >= 0


@pytest.mark.metamorphic
def test_permutation_reorders_predictions() -> None:
    torch.manual_seed(42)
    model = WineClassifier()
    model.eval()
    inputs = torch.randn(8, 13)
    permutation = torch.tensor([7, 0, 4, 2, 6, 1, 5, 3])

    with torch.no_grad():
        original = model(inputs)
        permuted = model(inputs[permutation])

    torch.testing.assert_close(permuted, original[permutation])


@pytest.mark.metamorphic
def test_batch_partition_preserves_predictions() -> None:
    torch.manual_seed(42)
    model = WineClassifier()
    model.eval()
    inputs = torch.randn(8, 13)

    with torch.no_grad():
        whole_batch = model(inputs)
        separate_batches = torch.cat(
            [model(inputs[:3]), model(inputs[3:])],
            dim=0,
        )

    torch.testing.assert_close(
        whole_batch,
        separate_batches,
        rtol=1e-5,
        atol=1e-6,
    )
