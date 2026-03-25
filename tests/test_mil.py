import numpy as np
import torch

from bgnet import BGNetConfig, BGNetMILModel


CH_NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]


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


def _model() -> BGNetMILModel:
    return BGNetMILModel.from_config(
        BGNetConfig.from_preset("clinical", n_outputs=2, ch_names=CH_NAMES, sfreq=256),
        label_map={"normal": 0, "abnormal": 1},
    ).eval()


def test_mil_save_pretrained_roundtrip(tmp_path):
    model = _model()
    bundle = model.save_pretrained(tmp_path / "bundle")
    reloaded = BGNetMILModel.from_pretrained(bundle).eval()
    x = torch.randn(7, len(CH_NAMES), 2560)
    bag_slices = torch.tensor([[0, 7]], dtype=torch.long)
    torch.testing.assert_close(
        torch.as_tensor(model.predict_bag_proba(x, bag_slices=bag_slices)),
        torch.as_tensor(reloaded.predict_bag_proba(x, bag_slices=bag_slices)),
    )


def test_mil_predict_raw_full_shapes():
    model = _model()
    raw = DummyRaw(np.random.randn(len(CH_NAMES), 1200).astype(np.float32), CH_NAMES, sfreq=128)
    result = model.predict_raw_full(raw, window_seconds=1.0, stride_seconds=0.5)
    assert result.probabilities.shape == (2,)
    assert result.attention_weights.ndim == 1
    assert result.attention_weights.shape[0] == result.window_start_seconds.shape[0]
    assert abs(float(result.attention_weights.sum()) - 1.0) < 1e-4
