# Hakka tone-mark restoration

Restoring Pha̍k-fa-sṳ tone marks in Hakka Wikipedia text with a from-scratch BiLSTM-CRF.

## Setup

```bash
uv sync
```

On macOS this installs PyTorch with MPS (Apple GPU) support; on Linux/Windows it installs the CUDA 12.8 build for NVIDIA GPUs.

## Data check

Download `20231101.hak/train-00000-of-00001.parquet` from
[wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), then:

```bash
uv run python check_hakwiki.py path/to/train-00000-of-00001.parquet
```
