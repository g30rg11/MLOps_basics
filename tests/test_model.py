import pytest
import torch

from mlops_homework.model import DenseBlock, WineClassifier


@pytest.mark.unit
def test_dense_block_shape_and_gradients() -> None:
    torch.manual_seed(42)
    block = DenseBlock(in_features=13, out_features=16)
    inputs = torch.randn(4, 13, requires_grad=True)

    outputs = block(inputs)
    outputs.square().mean().backward()

    assert outputs.shape == (4, 16)
    assert torch.isfinite(outputs).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()

    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [1, 8, 17])
def test_classifier_output(batch_size: int) -> None:
    torch.manual_seed(42)
    model = WineClassifier()
    inputs = torch.randn(batch_size, 13)

    logits = model(inputs)

    assert logits.shape == (batch_size, 3)
    assert torch.isfinite(logits).all()
