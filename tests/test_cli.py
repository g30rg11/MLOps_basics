import sys
from pathlib import Path

import pytest

from mlops_homework.train import main, run_training


@pytest.mark.unit
@pytest.mark.parametrize("epochs", [0, -1])
def test_run_training_rejects_invalid_epochs(
    epochs: int,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="epochs must be positive"):
        run_training(epochs=epochs, output_dir=output_dir)

    assert not output_dir.exists()


@pytest.mark.unit
def test_cli_rejects_invalid_epochs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--epochs",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "--epochs must be positive" in capsys.readouterr().err
    assert not output_dir.exists()
