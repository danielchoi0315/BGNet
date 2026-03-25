from __future__ import annotations

from typing import Sequence

from torch import nn

from .config import BGNetConfig
from .model import BGNet, BGNetOutput

try:
    from braindecode.models.base import EEGModuleMixin
except Exception as exc:  # pragma: no cover
    EEGModuleMixin = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


if EEGModuleMixin is None:  # pragma: no cover
    class BraindecodeBGNet(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Braindecode support requires `pip install bgnet-eeg[braindecode]`.") from _IMPORT_ERROR
else:
    class BraindecodeBGNet(EEGModuleMixin, nn.Module):
        def __init__(
            self,
            n_outputs: int,
            n_chans: int,
            *,
            n_times: int | None = None,
            sfreq: float | None = None,
            input_window_seconds: float | None = None,
            chs_info=None,
            channel_names: Sequence[str] | None = None,
            preset: str = "clinical",
            **kwargs,
        ) -> None:
            if channel_names is None:
                if chs_info is None:
                    raise ValueError("Provide channel_names or chs_info for BraindecodeBGNet.")
                channel_names = [str(info["ch_name"]) for info in chs_info]
            super().__init__(
                n_outputs=n_outputs,
                n_chans=n_chans,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
                chs_info=chs_info,
            )
            config = BGNetConfig.from_preset(
                preset,
                n_outputs=n_outputs,
                ch_names=channel_names,
                sfreq=float(sfreq or 250.0),
                **kwargs,
            )
            self.bgnet = BGNet(config)

        def forward(self, x):
            return self.bgnet(x)

        def forward_full(self, x) -> BGNetOutput:
            return self.bgnet.forward_full(x)
