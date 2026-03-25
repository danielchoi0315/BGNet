# Reproduce Paper Results

This public repository packages the BGNet model itself.

It does not mirror the full cluster orchestration and cache-building system used in the research repo.

Recommended workflow:

1. Convert a research checkpoint with `convert_research_checkpoint(...)`
2. Load it with `BGNet.from_pretrained(...)`
3. Reuse the original benchmark manifests and evaluation code from the research repository for exact paper replication

