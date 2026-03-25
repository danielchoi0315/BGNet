# FAQ

## Does BGNet change the original research math?

No. The public package vendors the core math-bearing modules and wraps them with a simpler public API.

## Why is there no giant benchmark pipeline in this repo?

Because this package is meant to be a reusable model library, not a cluster-ops dump.

## Why does channel order matter?

The geometry and source mapping depend on channel identity and order. BGNet stores the training channel order in its config and can reorder inputs when channel names are provided.

