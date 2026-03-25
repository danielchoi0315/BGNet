# Quickstart

Install:

```bash
pip install git+https://github.com/danielchoi0315/BGNet.git
```

```python
from bgnet import BGNet
import numpy as np

x = np.random.randn(1, 19, 2560).astype("float32")
ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

model = BGNet.from_preset("clinical", n_outputs=2, ch_names=ch_names, sfreq=256)
probs = model.predict_proba(x)
pred = model.predict(x)
```

From MNE info:

```python
import mne
from bgnet import BGNet

info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types="eeg")
model = BGNet.from_mne_info(info, preset="clinical", n_outputs=2)
```

Clinical raw inference:

```python
raw = mne.io.read_raw_edf("sample.edf", preload=True)
result = model.predict_raw_full(raw, window_seconds=10.0, stride_seconds=5.0)
print(result.prediction, result.probabilities)
```
