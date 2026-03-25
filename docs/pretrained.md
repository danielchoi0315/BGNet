# Pretrained Checkpoints

The package ships a checkpoint registry API and a cache directory helper, but the initial public
release does not bundle large weights inside git and does not yet publish named checkpoints.

Use:

```python
from bgnet import BGNet, checkpoint_cache_dir

model = BGNet.from_pretrained("/path/to/local/checkpoint_dir")
print(checkpoint_cache_dir())
```

Each checkpoint directory contains:

- `model.safetensors`
- `config.json`
- `metadata.json`
- `channels.json`

Registry-backed checkpoints can resolve from:

- `local_path`
- Hugging Face bundles via `hf_repo_id`
- downloadable zip bundles via `url`

Window / encoder checkpoints from the research repo can be converted with
`convert_research_checkpoint(...)`. Record-level MIL checkpoints are intentionally not accepted by
that converter because they require an extra head bundle beyond the core BGNet encoder.

For clinical record-level checkpoints, use `convert_research_mil_checkpoint(...)` and
`BGNetMILModel.from_pretrained(...)`.
