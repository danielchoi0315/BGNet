# Contributing

## Scope

This repository is for the public BGNet model package.

Please keep contributions focused on:

- public API quality
- checkpointing
- documentation
- examples
- tests
- integration wrappers

Please do not add cluster orchestration, benchmark-specific node scripts, or internal experiment dumps here.

## Development

```bash
pip install -e .[dev]
python -m pytest
python -m build
```

## Parity Rule

BGNet is intended to preserve the core Background-First EEG math. If a change affects numerical behavior, document it explicitly and add a regression test.

