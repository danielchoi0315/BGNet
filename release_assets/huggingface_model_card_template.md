---
library_name: bgnet-eeg
tags:
  - eeg
  - clinical-eeg
  - bgnet
  - pytorch
license: mit
---

# BGNet Model Card

## Model

- Name: `bgnet/<checkpoint-name>`
- Bundle type: `encoder` or `mil`
- BGNet version: `0.1.x`
- Task: `<task>`
- Labels: `<label map>`

## Intended Use

- Primary use: `<screening / feature extraction / research benchmarking>`
- Input montage: `<channel list or canonical montage>`
- Sampling rate: `<sfreq>`

## Bundle Contents

- `config.json`
- `channels.json`
- `metadata.json` or `mil_metadata.json`
- `model.safetensors` or `encoder.safetensors` + `mil_head.safetensors`
- optional `label_map.json`
- optional `split_manifest.json`

## Training Data

- Dataset: `<dataset>`
- Split protocol: `<official / patient-disjoint / record-disjoint>`
- Notes: `<known caveats>`

## Metrics

- Balanced accuracy: `<value>`
- AUROC: `<value>`
- Comparator: `<value>`

## Known Limitations

- `<missing channels policy>`
- `<domain / montage caveats>`
- `<not for diagnosis without validation>`
