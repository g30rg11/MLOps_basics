from torch import Tensor, nn


class DenseBlock(nn.Module):
    """Fully connected layer followed by ReLU."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return self.activation(self.linear(inputs))


class WineClassifier(nn.Module):
    """Classify Wine samples using 13 input features."""

    def __init__(self, hidden_features: int = 16) -> None:
        super().__init__()
        self.block = DenseBlock(13, hidden_features)
        self.output = nn.Linear(hidden_features, 3)

    def forward(self, inputs: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return self.output(self.block(inputs))
