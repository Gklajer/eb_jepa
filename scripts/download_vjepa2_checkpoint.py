#!/usr/bin/env python3
"""Download and verify an official Meta V-JEPA 2 checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_ID = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_LOCAL_DIR = Path("downloads") / "hf" / "facebook" / "vjepa2-vitl-fpc64-256"
REQUIRED_FILES = (
    "config.json",
    "video_preprocessor_config.json",
    "model.safetensors",
)
DOWNLOAD_PATTERNS = (
    ".gitattributes",
    "README.md",
    "config.json",
    "video_preprocessor_config.json",
    "model.safetensors",
)


def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def _find_remote_file(model_info, filename: str):
    for sibling in model_info.siblings:
        if sibling.rfilename == filename:
            return sibling
    return None


def _require_remote_files(model_info, filenames: Iterable[str]) -> None:
    missing = [
        filename
        for filename in filenames
        if _find_remote_file(model_info, filename) is None
    ]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required file(s) on Hugging Face: {joined}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_local_snapshot(
    local_dir: Path,
    model_info,
    min_weights_bytes: int,
) -> None:
    """Verify that the downloaded snapshot contains usable checkpoint files."""
    missing = [name for name in REQUIRED_FILES if not (local_dir / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required downloaded file(s): {joined}")

    _load_json(local_dir / "config.json")
    _load_json(local_dir / "video_preprocessor_config.json")

    weights_path = local_dir / "model.safetensors"
    local_weights_size = weights_path.stat().st_size
    if local_weights_size < min_weights_bytes:
        raise RuntimeError(
            "Downloaded model.safetensors is smaller than expected: "
            f"{_format_bytes(local_weights_size)} < {_format_bytes(min_weights_bytes)}"
        )

    remote_weights = (
        _find_remote_file(model_info, "model.safetensors")
        if model_info is not None
        else None
    )
    remote_weights_size = getattr(remote_weights, "size", None)
    if remote_weights_size and local_weights_size != remote_weights_size:
        raise RuntimeError(
            "Downloaded model.safetensors size does not match Hugging Face metadata: "
            f"{_format_bytes(local_weights_size)} != {_format_bytes(remote_weights_size)}"
        )


def verify_transformers_load(local_dir: Path) -> None:
    """Optionally verify that Transformers can instantiate the processor and model."""
    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for --load-transformers. Install the current "
            "V-JEPA 2-compatible version with:\n"
            "  pip install -U git+https://github.com/huggingface/transformers"
        ) from exc

    processor = AutoVideoProcessor.from_pretrained(local_dir, local_files_only=True)
    model = AutoModel.from_pretrained(local_dir, local_files_only=True)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Transformers processor: {processor.__class__.__name__}")
    print(f"Transformers model: {model.__class__.__name__}")
    print(f"Model parameters: {num_params:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download facebook/vjepa2-vitl-fpc64-256 from Hugging Face and "
            "verify that the required checkpoint files are present."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repo to download. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face revision, branch, or commit. Default: main",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Destination directory. Default: {DEFAULT_LOCAL_DIR}",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only verify Hugging Face metadata; do not download model files.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not hit the network; verify files that are already present locally.",
    )
    parser.add_argument(
        "--load-transformers",
        action="store_true",
        help="After download, instantiate AutoVideoProcessor and AutoModel locally.",
    )
    parser.add_argument(
        "--min-weights-bytes",
        type=int,
        default=1_000_000_000,
        help="Minimum acceptable size for model.safetensors. Default: 1GB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.local_files_only:
            if args.metadata_only:
                raise RuntimeError(
                    "--metadata-only cannot be combined with --local-files-only"
                )

            local_dir = args.local_dir.expanduser().resolve()
            print(f"Verifying local snapshot only: {local_dir}")
            verify_local_snapshot(
                local_dir,
                model_info=None,
                min_weights_bytes=args.min_weights_bytes,
            )

            if args.load_transformers:
                verify_transformers_load(local_dir)

            print(f"Local snapshot verified: {local_dir}")
            return 0

        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required. Install project dependencies or run:\n"
                "  pip install huggingface-hub"
            ) from exc

        api = HfApi()
        model_info = api.model_info(
            repo_id=args.repo_id,
            revision=args.revision,
            files_metadata=True,
        )
        _require_remote_files(model_info, REQUIRED_FILES)

        print(f"Hugging Face repo: {args.repo_id}@{args.revision}")
        for filename in REQUIRED_FILES:
            remote_file = _find_remote_file(model_info, filename)
            print(
                f"Remote file OK: {filename} "
                f"({_format_bytes(getattr(remote_file, 'size', None))})"
            )

        if args.metadata_only:
            print("Metadata verification complete; no files downloaded.")
            return 0

        local_dir = args.local_dir.expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = Path(
            snapshot_download(
                repo_id=args.repo_id,
                revision=args.revision,
                local_dir=local_dir,
                allow_patterns=list(DOWNLOAD_PATTERNS),
                local_files_only=args.local_files_only,
            )
        )
        verify_local_snapshot(snapshot_path, model_info, args.min_weights_bytes)

        if args.load_transformers:
            verify_transformers_load(snapshot_path)

        print(f"Downloaded snapshot verified: {snapshot_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
