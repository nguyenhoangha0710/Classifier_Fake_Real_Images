"""Download one pinned CLIP snapshot before launching training workers."""
import argparse
import json
from pathlib import Path


def download_clip(cache_dir, revision='main'):
    from huggingface_hub import HfApi, snapshot_download

    repo = 'openai/clip-vit-large-patch14'
    info = HfApi().model_info(repo_id=repo, revision=revision)
    files = {item.rfilename for item in info.siblings}
    # Avoid downloading duplicate PyTorch and safetensors copies of the same model.
    weights = next((name for name in ('model.safetensors', 'pytorch_model.bin') if name in files), None)
    if weights is None or 'config.json' not in files:
        raise RuntimeError('Expected config.json and unsharded CLIP model weights')
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(repo_id=repo, revision=info.sha,
                                    cache_dir=str(cache), allow_patterns=['config.json', weights]))
    config = json.loads((snapshot / 'config.json').read_text())
    vision = config.get('vision_config', config)
    if (vision.get('hidden_size'), vision.get('num_hidden_layers'),
            vision.get('patch_size'), vision.get('image_size')) != (1024, 24, 14, 224):
        raise ValueError('Downloaded checkpoint is not CLIP ViT-L/14 at 224 resolution')
    provenance = {'repo_id': repo, 'requested_revision': revision, 'resolved_commit': info.sha,
                  'weights_file': weights, 'snapshot_path': str(snapshot)}
    (cache / 'clip_download.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2), flush=True)
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    download_clip(args.cache_dir, args.revision)
