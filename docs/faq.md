# FAQ

## Does BGNet change the original research math?

No. The public package vendors the core math-bearing modules and wraps them with a simpler public API.

## Why is there no giant benchmark pipeline in this repo?

Because this package is meant to be a reusable model library, not a cluster-ops dump.

## Why does channel order matter?

The geometry and source mapping depend on channel identity and order. BGNet stores the training channel order in its config and can reorder inputs when channel names are provided.

## What happens if channels are missing?

The public runtime zero-fills missing channels by default when names are provided. That keeps the
model math intact while making public inference usable on real clinical recordings with partial
montages. If you want strict behavior instead, call the model with `on_missing="error"`.

## Can I convert the clinical MIL checkpoints from the research repo?

Not directly in `v0.1`. The public converter accepts encoder / window checkpoints only and rejects
MIL checkpoints with a separate `mil_head_state` so it cannot silently produce an incomplete bundle.
