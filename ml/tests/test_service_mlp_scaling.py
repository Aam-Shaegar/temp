"""Regression tests for highly concentrated normal TCP age features."""
import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from netwatch_ml.models.detector import fit_transform_parameters, transform
from netwatch_ml.training import service_detector


def test_mlp_scales_tiny_normal_iqr_against_training_only_lifecycle_values():
    healthy = np.asarray([[.00021, 1.], [.00022, 2.], [.00023, 3.], [.00024, 4.]])
    train = np.vstack([healthy, [1000., np.nan]])
    healthy_params = fit_transform_parameters(healthy)
    assert np.max(np.abs(transform(train, healthy_params))) > 1000

    params = service_detector.bound_mlp_train_scale(train, healthy_params)
    scores = transform(train, params)

    assert params["median"] == healthy_params["median"]
    assert params["scale"][0] > healthy_params["scale"][0]
    assert params["scale"][1] == healthy_params["scale"][1]
    assert np.max(np.abs(scores)) <= service_detector.MLP_MAX_TRAIN_DEVIATION + 1e-10
    assert scores[-1, 3] == 1


def test_unconverged_mlp_is_not_published(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"feature_set_version": "traffic-v2"}))
    destination = tmp_path / "bundle"
    train = [([0.], "normal"), ([1000.], "anomaly")]
    cal = [([0.], "normal")]

    class UnconvergedMLP:
        def __init__(self, **_):
            pass

        def fit(self, x, y):
            warnings.warn("failed", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(service_detector, "load_partitions", lambda *args: dict(train=train, calibration=cal))
    monkeypatch.setattr(service_detector, "train_scaler", lambda rows, *args: fit_transform_parameters(np.array(rows)))
    monkeypatch.setattr(service_detector, "feature_rows", lambda rows, *args: np.array(rows))
    monkeypatch.setattr(service_detector, "MLPClassifier", UnconvergedMLP)

    with pytest.raises(ValueError, match="MLP did not converge"):
        service_detector.fit(manifest, destination, "mlp")

    assert not destination.exists()
