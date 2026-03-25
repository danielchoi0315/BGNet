# BGNet

`BGNet` is a public, pip-installable packaging of the Background-First EEG model.

It keeps the core model math consistent with the research implementation while giving
users a simpler surface:

- `BGNet.from_preset(...)`
- `BGNet.from_pretrained(...)`
- `predict(...)`
- `predict_proba(...)`
- `save_pretrained(...)`

## Why BGNet

BGNet is aimed at the regime where generic EEG architectures often struggle most:

- heterogeneous EEG cohorts
- low-label training
- clinically messy channel sets

The public package keeps the original model math intact, but strips away the research-repo friction.

## Install

```bash
pip install bgnet-eeg
```

Optional extras:

```bash
pip install "bgnet-eeg[train]"
pip install "bgnet-eeg[braindecode]"
pip install "bgnet-eeg[dev]"
```

## Quickstart

```python
from bgnet import BGNet
import numpy as np

x = np.random.randn(1, 19, 2560).astype("float32")
ch_names = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "O2",
]

model = BGNet.from_preset(
    "clinical",
    n_outputs=2,
    ch_names=ch_names,
    sfreq=256,
)

probs = model.predict_proba(x)
pred = model.predict(x)
print(pred, probs)
```

## Fine-Tuning

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
print(metrics)
```

## Current Benchmark Snapshot

Closed results from the current internal benchmark package:

| Task | BGNet | Comparator | Result |
| --- | --- | --- | --- |
| TUAB full-label | 79.77 BA / 0.8788 AUROC | Transformer 77.38 / 0.8698 | BGNet better |
| TUAB 25% | 79.78 / 0.8731 | Transformer 76.10 / 0.8397 | BGNet better |
| TUAB 10% | 72.43 / 0.7981 | EEGNet 66.83 / 0.7129 | BGNet better |
| TUSZ full-label | 74.30 / 0.8231 | Transformer 55.67 / 0.7619 | BGNet better |

These numbers are here to position the model. The full benchmark and artifact story remains in the research repository.

## When To Use BGNet vs EEGNet

Use `BGNet` when:

- you want a physiology-first EEG model
- you care about heterogeneous or low-label regimes
- you want a stronger inductive bias than a generic temporal CNN or transformer

Use `EEGNet` when:

- you need a tiny baseline fast
- you want the lightest possible default model
- you are benchmarking a small task and care more about simplicity than representational structure

## Research Checkpoints

BGNet can convert research checkpoints into a public checkpoint directory:

```python
from bgnet import BGNetConfig, convert_research_checkpoint

config = BGNetConfig.from_preset(
    "clinical",
    n_outputs=2,
    ch_names=ch_names,
    sfreq=256,
)
convert_research_checkpoint("best_model.pt", "./bgnet-tuab", config=config)
```

## Scope

This repository is intentionally small. It ships:

- the public BGNet model
- frozen geometry defaults
- checkpoint IO
- a small classifier wrapper
- an optional Braindecode adapter

## Limitations

- The first public release supports frozen `standard_1005` geometry only.
- Pretrained registry entries are not populated yet; `from_pretrained(...)` currently targets local checkpoint bundles.
- The public package does not include the full clinical benchmark orchestration stack.

It does not ship:

- cluster orchestration
- TUAB/TUSZ download pipelines
- internal training caches
- benchmark-specific launcher scripts

## Docs

- [Quickstart](docs/quickstart.md)
- [Pretrained Checkpoints](docs/pretrained.md)
- [Fine-Tuning](docs/fine_tuning.md)
- [Braindecode Integration](docs/braindecode.md)
- [Reproducing Paper Results](docs/reproduce_paper.md)
- [FAQ](docs/faq.md)
