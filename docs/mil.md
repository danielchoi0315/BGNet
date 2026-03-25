# MIL Clinical Bundles

BGNet now supports a public record-level MIL bundle format for clinical models such as `TUAB` and
`TUSZ`.

Bundle contents:

- `encoder.safetensors`
- `mil_head.safetensors`
- `config.json`
- `channels.json`
- `label_map.json`
- `mil_metadata.json`
- optional `split_manifest.json`

Convert a research MIL checkpoint:

```python
from bgnet import BGNetConfig, convert_research_mil_checkpoint

config = BGNetConfig.from_preset(
    "clinical",
    n_outputs=2,
    ch_names=ch_names,
    sfreq=256,
)

convert_research_mil_checkpoint(
    "best_model.pt",
    "./bgnet-tuab-mil",
    config=config,
    label_map={"normal": 0, "abnormal": 1},
    split_manifest_path="split_manifest.json",
)
```

Load the bundle:

```python
from bgnet import BGNetMILModel

model = BGNetMILModel.from_pretrained("./bgnet-tuab-mil")
```

Clinical raw-record inference:

```python
import mne

raw = mne.io.read_raw_edf("sample.edf", preload=True)
result = model.predict_raw_full(raw, window_seconds=10.0, stride_seconds=5.0)
print(result.prediction, result.probabilities, result.attention_weights.shape)
```

Notes:

- The MIL head math matches the research attention-pooling head.
- The public MIL bundle is inference-oriented; it is not a full training-resume artifact.
- `label_map.json` should reflect the exported task labels exactly.
