import numpy as np

from bgnet import BGNet, BGNetConfig
from bgnet.runtime import resolve_input_array


CH_NAMES = ["Fp1", "Fp2", "F3"]


class DummyRaw:
    def __init__(self, data: np.ndarray, ch_names: list[str], sfreq: float) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self.ch_names = list(ch_names)
        self.info = {"sfreq": float(sfreq)}

    def copy(self) -> "DummyRaw":
        return DummyRaw(self._data.copy(), list(self.ch_names), float(self.info["sfreq"]))

    def load_data(self) -> "DummyRaw":
        return self

    def get_data(self) -> np.ndarray:
        return self._data

    def resample(self, sfreq: float, npad: str = "auto") -> "DummyRaw":
        _ = npad
        old_sfreq = float(self.info["sfreq"])
        new_sfreq = float(sfreq)
        old_times = np.arange(self._data.shape[1], dtype=np.float32) / old_sfreq
        new_len = max(1, int(round(self._data.shape[1] * new_sfreq / old_sfreq)))
        new_times = np.arange(new_len, dtype=np.float32) / new_sfreq
        self._data = np.stack(
            [np.interp(new_times, old_times, row).astype(np.float32) for row in self._data],
            axis=0,
        )
        self.info["sfreq"] = new_sfreq
        return self


def test_resolve_input_array_zero_fills_missing_channels():
    config = BGNetConfig.from_preset("clinical", n_outputs=2, ch_names=CH_NAMES, sfreq=256)
    x = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    out = resolve_input_array(x, config=config, ch_names=["Fp1", "F3"])
    assert out.shape == (1, 3, 3)
    np.testing.assert_allclose(out[0, 0].numpy(), x[0])
    np.testing.assert_allclose(out[0, 1].numpy(), np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(out[0, 2].numpy(), x[1])


def test_predict_raw_full_resamples_and_returns_window_metadata():
    model = BGNet.from_preset("clinical", n_outputs=2, ch_names=CH_NAMES, sfreq=256).eval()
    raw = DummyRaw(np.random.randn(3, 600).astype(np.float32), CH_NAMES, sfreq=128)
    result = model.predict_raw_full(raw, window_seconds=1.0, stride_seconds=0.5)
    assert result.probabilities.shape == (2,)
    assert result.window_probabilities.ndim == 2
    assert result.window_probabilities.shape[0] == result.window_start_seconds.shape[0]
    assert result.window_start_seconds.shape == result.window_stop_seconds.shape
    assert int(result.prediction) in {0, 1}
