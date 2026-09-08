import numpy as np
from src.tracker.kalman_filter import KalmanFilter8D


def test_kalman_filter_init_and_predict():
    kf = KalmanFilter8D()
    measurement = np.array([100.0, 200.0, 1.2, 50.0], dtype=np.float32)
    mean, cov = kf.initiate(measurement)

    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    assert np.allclose(mean[:4], measurement)

    # Predict
    pred_mean, pred_cov = kf.predict(mean, cov)
    assert pred_mean.shape == (8,)
    assert pred_cov.shape == (8, 8)


def test_kalman_filter_update():
    kf = KalmanFilter8D()
    measurement1 = np.array([100.0, 200.0, 1.2, 50.0], dtype=np.float32)
    mean, cov = kf.initiate(measurement1)

    pred_mean, pred_cov = kf.predict(mean, cov)
    measurement2 = np.array([105.0, 202.0, 1.2, 50.0], dtype=np.float32)
    up_mean, up_cov = kf.update(pred_mean, pred_cov, measurement2)

    assert up_mean[0] > 100.0  # Moving in +x direction
