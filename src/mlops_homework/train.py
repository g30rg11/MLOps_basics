import argparse
import json
from pathlib import Path

import torch

from mlops_homework.data import make_loader, prepare_wine
from mlops_homework.model import WineClassifier
from mlops_homework.training import evaluate, train_epoch


def run_training(
    epochs: int,
    output_dir: Path,
    seed: int = 42,
    hidden_features: int = 16,
) -> dict[str, float | int]:
    """Train on Wine and save the model, preprocessing and metrics."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    data = prepare_wine(seed=seed)
    train_loader = make_loader(
        data.x_train,
        data.y_train,
        batch_size=32,
        shuffle=True,
        seed=seed,
    )
    test_loader = make_loader(
        data.x_test,
        data.y_test,
        batch_size=32,
        shuffle=False,
    )

    model = WineClassifier(hidden_features=hidden_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loss = 0.0
    for _ in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer)

    evaluation = evaluate(model, test_loader)
    metrics: dict[str, float | int] = {
        "epochs": epochs,
        "seed": seed,
        "train_samples": len(data.y_train),
        "test_samples": len(data.y_test),
        "train_loss": train_loss,
        "loss": evaluation["loss"],
        "accuracy": evaluation["accuracy"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "hidden_features": hidden_features,
        "mean": data.mean,
        "scale": data.scale,
    }
    torch.save(checkpoint, output_dir / "model.pt")

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small Wine classifier on CPU."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    metrics = run_training(
        epochs=args.epochs,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
