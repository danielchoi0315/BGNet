import pytest
import torch

from bgnet import BGNet, BGNetConfig, BGNetMILModel, convert_research_checkpoint, convert_research_mil_checkpoint


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


def test_convert_research_checkpoint_roundtrip_loadable(tmp_path):
    ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
    model = BGNet.from_preset("clinical", n_outputs=2, ch_names=ch_names, sfreq=256).eval()
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save({"encoder_state": model.core.state_dict()}, checkpoint_path)
    config = BGNetConfig.from_preset("clinical", n_outputs=2, ch_names=ch_names, sfreq=256)
    bundle = convert_research_checkpoint(checkpoint_path, tmp_path / "bundle", config=config)
    reloaded = BGNet.from_pretrained(bundle).eval()
    x = torch.randn(2, len(ch_names), 2560)
    torch.testing.assert_close(model(x), reloaded(x))


def test_convert_research_mil_checkpoint_roundtrip_loadable(tmp_path):
    ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
    model = BGNetMILModel.from_config(
        BGNetConfig.from_preset("clinical", n_outputs=2, ch_names=ch_names, sfreq=256),
        label_map={"normal": 0, "abnormal": 1},
    ).eval()
    checkpoint_path = tmp_path / "mil.pt"
    torch.save(
        {
            "encoder_state": model.encoder.core.state_dict(),
            "mil_head_state": model.mil_head.state_dict(),
            "epoch": 7,
        },
        checkpoint_path,
    )
    bundle = convert_research_mil_checkpoint(
        checkpoint_path,
        tmp_path / "mil_bundle",
        config=model.config,
        label_map={"normal": 0, "abnormal": 1},
    )
    reloaded = BGNetMILModel.from_pretrained(bundle).eval()
    x = torch.randn(6, len(ch_names), 2560)
    bag_slices = torch.tensor([[0, 6]], dtype=torch.long)
    orig = model.predict_bag_proba(x, bag_slices=bag_slices)
    new = reloaded.predict_bag_proba(x, bag_slices=bag_slices)
    assert reloaded.label_map == {"normal": 0, "abnormal": 1}
    torch.testing.assert_close(torch.as_tensor(orig), torch.as_tensor(new))
