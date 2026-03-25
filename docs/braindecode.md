# Braindecode Integration

Install the extra:

```bash
pip install "bgnet-eeg[braindecode] @ git+https://github.com/danielchoi0315/BGNet.git"
```

Then use:

```python
from braindecode import EEGClassifier
from bgnet.braindecode import BraindecodeBGNet

module = BraindecodeBGNet(
    n_outputs=2,
    n_chans=len(ch_names),
    sfreq=256,
    channel_names=ch_names,
    preset="clinical",
)

clf = EEGClassifier(
    module,
    criterion="cross_entropy",
    optimizer__lr=1e-3,
    train_split=None,
)
```

If you already have `mne.Info`, prefer passing `chs_info=raw.info["chs"]` so the Braindecode module
inherits the exact EEG channel metadata. BGNet keeps its own checkpoint/config bundle as the source
of truth for model-specific parameters; the Braindecode wrapper is the benchmark-facing adapter, not
the serialization format.
