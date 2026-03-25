from pathlib import Path
import sys

import pytest
import torch

from bgnet import BGNet
from bgnet.runtime import sensor_geometry_tensors


RESEARCH_SRC = Path(__file__).resolve().parents[2] / "eeg_native_moe" / "src"
if RESEARCH_SRC.exists():
    sys.path.insert(0, str(RESEARCH_SRC.resolve()))


@pytest.mark.skipif(not RESEARCH_SRC.exists(), reason="Research repo not available.")
def test_parity_with_research_repo_forward():
    from sst_eeg.geometry import build_default_source_layout, normalized_source_adjacency
    from sst_eeg.model_background_first import BackgroundFirstSourceFieldEEG

    ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
    public = BGNet.from_preset("clinical", n_outputs=2, ch_names=ch_names, sfreq=256).eval()

    layout = build_default_source_layout()
    adjacency = normalized_source_adjacency(layout.positions, sigma=0.10, self_weight=1.5)
    research = BackgroundFirstSourceFieldEEG(
        source_positions=layout.positions,
        source_names=layout.names,
        graph_adjacency=adjacency,
        n_sensor_channels=len(ch_names),
        n_classes=2,
        time_window_size=250,
        time_window_stride=125,
        sample_rate_hz=256,
        d_model=128,
        osc_depth=3,
        n_heads=4,
        dropout=0.1,
        sigma=0.05,
        low_rank=8,
        use_pair_expert=False,
        use_event_expert=True,
        use_artifact_expert=True,
    ).eval()

    research.load_state_dict(public.core.state_dict())
    x = torch.randn(2, len(ch_names), 2560)
    sensor_pos, sensor_mask = sensor_geometry_tensors(public.config, batch_size=2, device=torch.device("cpu"))
    public_logits = public(x)
    research_logits = research(x, sensor_pos=sensor_pos, sensor_mask=sensor_mask).logits
    torch.testing.assert_close(public_logits, research_logits)
