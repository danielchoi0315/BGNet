# Pretrained Checkpoints

The package ships a checkpoint registry API, but the initial public release does not bundle large weights inside git.

Use:

```python
from bgnet import BGNet

model = BGNet.from_pretrained("/path/to/local/checkpoint_dir")
```

Each checkpoint directory contains:

- `model.safetensors`
- `config.json`
- `metadata.json`

