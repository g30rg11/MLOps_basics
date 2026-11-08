import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from sklearn.datasets import load_wine

from mlops_homework.model import WineClassifier


@pytest.mark.e2e
def test_training_cli_saves_usable_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlops_homework.train",
            "--epochs",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    checkpoint_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"

    assert checkpoint_path.is_file()
    assert metrics_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics["epochs"] == 3
    assert metrics["train_samples"] == 142
    assert metrics["test_samples"] == 36
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["loss"] < float("inf")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model = WineClassifier(hidden_features=checkpoint["hidden_features"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    raw = torch.tensor(
        load_wine().data[:5],
        dtype=torch.float64,
    )
    inputs = ((raw - checkpoint["mean"]) / checkpoint["scale"]).float()

    with torch.no_grad():
        logits = model(inputs)
        predictions = logits.argmax(dim=1)

    assert logits.shape == (5, 3)
    assert torch.isfinite(logits).all()
    assert predictions.shape == (5,)
    assert ((predictions >= 0) & (predictions <= 2)).all()
