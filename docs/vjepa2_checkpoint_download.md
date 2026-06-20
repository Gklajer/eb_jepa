# V-JEPA 2 Checkpoint Download

Use this helper to fetch the lightweight official Meta V-JEPA 2 starting point:

```bash
python scripts/download_vjepa2_checkpoint.py \
  --repo-id facebook/vjepa2-vitl-fpc64-256
```

By default the checkpoint is written to:

```text
downloads/hf/facebook/vjepa2-vitl-fpc64-256/
```

The `downloads/` directory is already ignored by Git, so the 1.3 GB
`model.safetensors` file is not committed.

The script verifies that Hugging Face exposes the required files, downloads only
the model card, config files, and `model.safetensors`, then checks that:

- `config.json` is present and valid JSON
- `video_preprocessor_config.json` is present and valid JSON
- `model.safetensors` is present
- the local weights file is at least 1 GB
- the local weights file size matches Hugging Face metadata when available

For a metadata-only check that does not download weights:

```bash
python scripts/download_vjepa2_checkpoint.py --metadata-only
```

To re-check a snapshot that is already on disk without using the network:

```bash
python scripts/download_vjepa2_checkpoint.py --local-files-only
```

For a stronger local load check, install a V-JEPA 2-compatible Transformers
build and pass `--load-transformers`:

```bash
pip install -U git+https://github.com/huggingface/transformers
python scripts/download_vjepa2_checkpoint.py --load-transformers
```

The official Hugging Face model card uses `AutoModel` and
`AutoVideoProcessor` for `facebook/vjepa2-vitl-fpc64-256`.
