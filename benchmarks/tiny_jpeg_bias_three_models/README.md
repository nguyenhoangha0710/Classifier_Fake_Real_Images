# Tiny-GenImage JPEG-bias evaluation

Test-only benchmark for three checkpoints trained on the original Tiny-GenImage:

- CLIP ViT-B/32 linear probe;
- NPR-ResNet18;
- original Full AIDE.

The notebook audits the Tiny validation set and evaluates raw references plus
PNG/JPEG transformations without retraining or overwriting source images.

## Files to upload to Kaggle

Upload the notebook:

```text
test_tiny_jpeg_bias_three_models_kaggle.ipynb
```

Create one private Kaggle Dataset from the local `checkpoints/` directory:

```text
checkpoints/
|-- aide_original_full_trainable.pt
|-- clip_linear_head.pt
|-- npr_resnet18_from_scratch.pt
`-- checkpoint_manifest.json
```

Also attach `yangsangtai/tiny-genimage` to the notebook. The notebook searches
`/kaggle/input` recursively, verifies all three SHA-256 hashes, and refuses to
run with a wrong checkpoint.

Full AIDE additionally needs its frozen OpenCLIP ConvNeXt-XXLarge backbone.
Either enable Internet so `open_clip_torch` downloads it, or attach exactly one
file named `open_clip_pytorch_model.bin` and leave
`CONFIG["aide_semantic_checkpoint"] = None` for automatic discovery.

## Recommended first run

Set `CONFIG["smoke"] = True`. This evaluates 100 real and 100 fake images per
generator. After the outputs look correct, set it back to `False` for all 7,000
Tiny test images.

The result archive is written to:

```text
/kaggle/working/tiny_jpeg_bias_three_models_results.zip
```

Partial prediction CSVs are saved after every 25 batches. Re-running the
inference cell in the same Kaggle session resumes completed samples.

## Rebuild the notebook

After editing any runtime, regenerate the self-contained notebook locally:

```powershell
.\.venv12\Scripts\python.exe `
  benchmarks/tiny_jpeg_bias_three_models/build_kaggle_notebook.py
```
