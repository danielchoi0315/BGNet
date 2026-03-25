# Fine-Tuning

Install the training extra:

```bash
pip install "bgnet-eeg[train] @ git+https://github.com/danielchoi0315/BGNet.git"
```

```python
from bgnet import BGNetClassifier

clf = BGNetClassifier.from_preset(
    "clinical",
    n_outputs=2,
    ch_names=ch_names,
    sfreq=256,
)
clf.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=5)
metrics = clf.evaluate(X_test, y_test)
```
