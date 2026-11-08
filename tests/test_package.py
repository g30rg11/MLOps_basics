from unittest.mock import Mock

import pytest

import mlops_homework
from mlops_homework import train


@pytest.mark.unit
def test_entrypoint_calls_training(monkeypatch: pytest.MonkeyPatch) -> None:
    training_main = Mock()
    monkeypatch.setattr(train, "main", training_main)

    mlops_homework.main()

    training_main.assert_called_once_with()
