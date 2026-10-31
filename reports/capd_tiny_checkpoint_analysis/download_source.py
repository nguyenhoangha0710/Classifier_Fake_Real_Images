"""Download only small CAPD logs, configs and evaluation tables from Kaggle."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

HANDLE = 'hadinho0710/checkpoin-capd-train-tiny'
API = 'https://www.kaggle.com/api/v1/datasets/'
ROOT = Path(__file__).resolve().parent / 'source'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'CAPD-checkpoint-audit/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata = fetch(API + 'view/' + HANDLE)
    version = json.loads(metadata)['currentVersionNumber']
    (ROOT / 'dataset_metadata.json').write_bytes(metadata)
    page = json.loads(fetch(API + 'list/' + HANDLE + '?pageSize=200'))
    files = page['datasetFiles']
    (ROOT / 'dataset_file_inventory.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    skip_names = {'best.pt', 'last_state.pt', 'epoch_001.pt', 'epoch_002.pt', 'epoch_003.pt',
                  'epoch_004.pt', 'epoch_005.pt', 'smaps_native.npz'}
    skip_suffixes = ('_manifest.csv', 'reused_manifest_index.csv', 'validation_processing_plan.csv',
                     'validation_predictions_epoch001.csv', 'validation_predictions_epoch002.csv',
                     'validation_predictions_epoch003.csv', 'validation_predictions_epoch004.csv',
                     'validation_predictions_epoch005.csv')
    selected = [item['name'] for item in files
                if Path(item['name']).name not in skip_names
                and not item['name'].endswith(skip_suffixes)
                and (item['name'].endswith(('.json', '.csv', '.txt', '.py')))]

    def download(name):
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = fetch(API + 'download/' + HANDLE + '/' + urllib.parse.quote(name, safe='')
                     + '?datasetVersionNumber=' + str(version))
        target.write_bytes(data)
        return {'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        downloaded = list(executor.map(download, selected))
    provenance = {'dataset': HANDLE, 'version': version, 'listed_files': len(files), 'downloads': downloaded}
    (ROOT / 'download_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
