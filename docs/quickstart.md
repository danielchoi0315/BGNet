# Quickstart

```python
from bgnet import BGNet
import numpy as np

x = np.random.randn(1, 19, 2560).astype("float32")
ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

model = BGNet.from_preset("clinical", n_outputs=2, ch_names=ch_names, sfreq=256)
probs = model.predict_proba(x)
pred = model.predict(x)
```

