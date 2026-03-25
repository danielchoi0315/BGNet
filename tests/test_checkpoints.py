import pytest
import torch

from bgnet import BGNetConfig, convert_research_checkpoint


def test_convert_research_checkpoint_rejects_mil_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "mil.pt"
    torch.save(
        {
            "encoder_state": {"dummy": torch.tensor([1.0])},
            "mil_head_state": {"head": torch.tensor([2.0])},
        },
        checkpoint_path,
    )
    config = BGNetConfig.from_preset("clinical", n_outputs=2, ch_names=["Fp1", "Fp2"], sfreq=256)
    with pytest.raises(ValueError, match="mil_head_state"):
        convert_research_checkpoint(checkpoint_path, tmp_path / "out", config=config)
