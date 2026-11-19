# Project 1: NumPy Image Encoder

## Goal

This project implements the forward pass of the supplied OpenCLIP ViT-B/32 image encoder using NumPy only. The implementation converts preprocessed images with shape `[B, 3, 224, 224]` into normalized image features with shape `[B, 512]`.

## Environment

- Python 3.9+
- NumPy
- Pillow

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Implementation

The six assessed parts are implemented in `image_encoder.py`:

1. `patchify`: converts each image into 49 ordered 32x32 RGB patches.
2. `split_heads`: converts `[B, N, 768]` into `[B, 12, N, 64]`.
3. `MultiHeadAttention`: computes scaled dot-product attention, merges heads, and applies the output projection.
4. `TransformerBlock`: applies both LayerNorms, the attention residual, GELU MLP, and the second residual.
5. `embed`: applies patch projection, prepends CLS, and adds positional embeddings.
6. `ImageEncoder.__call__`: selects CLS, applies post-Norm, projects to 512 dimensions, and normalizes each output row.

## Reproduce the results

Run the staged checks from this directory:

```bash
python check.py --stage all
```

The check compares preprocessing, patch tokens, input tokens, the first block, final tokens, final output, and a reordered batch against the supplied independent FP32 references.

To save output tokens for the ten supplied images:

```bash
python run.py --output output_tokens.npz
```

The output archive contains `ids` and `output_tokens`. The supplied weight and reference files are course assets and are not modified by the implementation.

## Validation result

All requested stages pass with finite `float32` values and the required tolerances. The final output has shape `(10, 512)` and each row has unit length.

## GitHub

The repository containing `image_encoder.py`, this README, and the report is:

```text
GitHub URL: https://github.com/Auga0123/project1-numpy-image-encoder
```

