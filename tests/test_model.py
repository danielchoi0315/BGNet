from pathlib import Path

import numpy as np
import torch

from bgnet import BGNet


CH_NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
GOLDEN = np.load(Path(__file__).resolve().parent / "data" / "golden_forward.npz")


def _model():
    torch.manual_seed(0)
    return BGNet.from_preset("clinical", n_outputs=2, ch_names=CH_NAMES, sfreq=256)


def test_forward_golden_shapes_and_logits():
    model = _model().eval()
    x = torch.as_tensor(GOLDEN["x"])
    logits = model(x)
    assert logits.shape == (2, 2)
    torch.testing.assert_close(logits, torch.as_tensor(GOLDEN["logits"]))


def test_state_dict_roundtrip_is_identity(tmp_path):
    model = _model().eval()
    x = torch.randn(2, len(CH_NAMES), 2560)
    y1 = model(x)
    path = tmp_path / "ckpt"
    model.save_pretrained(path)
    reloaded = BGNet.from_pretrained(path).eval()
    y2 = reloaded(x)
    torch.testing.assert_close(y1, y2)


def test_deterministic_eval_smoke():
    torch.manual_seed(0)
    model_a = _model().eval()
    torch.manual_seed(0)
    model_b = _model().eval()
    x = torch.randn(1, len(CH_NAMES), 2560)
    torch.testing.assert_close(model_a(x), model_b(x))


def test_one_step_train_smoke_deterministic():
    torch.manual_seed(0)
    model = _model().train()
    x = torch.randn(4, len(CH_NAMES), 2560)
    y = torch.tensor([0, 1, 0, 1])
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optim.step()
    assert torch.isfinite(loss)
