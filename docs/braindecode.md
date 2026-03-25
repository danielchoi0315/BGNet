# Braindecode Integration

Install the extra:

```bash
pip install "bgnet-eeg[braindecode]"
```

Then use:

```python
from bgnet.braindecode import BraindecodeBGNet

model = BraindecodeBGNet(
    n_outputs=2,
    n_chans=len(ch_names),
    sfreq=256,
    channel_names=ch_names,
    preset="clinical",
)
```

