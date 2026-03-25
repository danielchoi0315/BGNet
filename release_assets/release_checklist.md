# BGNet Release Checklist

## PyPI

1. Create PyPI project `bgnet-eeg`
2. Add trusted publisher:
   - owner: `danielchoi0315`
   - repo: `BGNet`
   - workflow: `release.yml`
3. Push a release tag:
   - `git tag v0.1.0`
   - `git push origin v0.1.0`

## Named Checkpoints

1. Build local bundle:
   - encoder: `convert_research_checkpoint(...)`
   - clinical MIL: `convert_research_mil_checkpoint(...)`
2. Upload bundle folder to Hugging Face
3. Copy `release_assets/checkpoints.template.json` into `src/bgnet/data/checkpoints.json`
4. Replace placeholder repo IDs with the real HF repos
5. Commit and tag a patch release

## Local Handoff

1. Save built wheel/sdist
2. Save public checkpoint bundles
3. Save the exact model card used for each released checkpoint
4. Save the registry JSON that matches the release
