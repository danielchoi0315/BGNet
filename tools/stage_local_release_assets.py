from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage BGNet release assets into a local handoff directory.")
    parser.add_argument("--repo-root", default=".", help="Path to the BGNet repo root.")
    parser.add_argument("--output-dir", required=True, help="Directory to receive the staged release assets.")
    parser.add_argument(
        "--bundle-dir",
        action="append",
        default=[],
        help="Optional checkpoint bundle directory to copy into the staged handoff.",
    )
    return parser.parse_args()


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def iter_bundle_sources(src: Path) -> list[Path]:
    if not src.is_dir():
        return []
    if (src / "config.json").exists():
        return [src]
    return sorted(
        child
        for child in src.iterdir()
        if child.is_dir() and (child / "config.json").exists()
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in ("dist", "release_assets"):
        src = repo_root / name
        if src.exists():
            dst = output_dir / name
            copy_tree(src, dst)
            copied.append(str(dst))

    docs_dst = output_dir / "docs"
    docs_dst.mkdir(exist_ok=True)
    for name in ("README.md", "pyproject.toml"):
        src = repo_root / name
        if src.exists():
            shutil.copy2(src, docs_dst / name)
            copied.append(str(docs_dst / name))

    bundles_dst = output_dir / "checkpoint_bundles"
    bundle_paths = []
    if bundles_dst.exists():
        shutil.rmtree(bundles_dst)
    for raw_bundle in args.bundle_dir:
        src = Path(raw_bundle).resolve()
        if not src.exists():
            continue
        bundles_dst.mkdir(exist_ok=True)
        for bundle_src in iter_bundle_sources(src):
            dst = bundles_dst / bundle_src.name
            copy_tree(bundle_src, dst)
            copied.append(str(dst))
            bundle_paths.append(str(dst))

    manifest = {
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "copied_paths": copied,
        "bundles": bundle_paths,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
