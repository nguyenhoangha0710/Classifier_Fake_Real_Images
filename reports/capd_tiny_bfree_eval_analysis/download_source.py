import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

HANDLE = 'hadinho0710/eval-capd-test-tiny-bfree'
API = 'https://www.kaggle.com/api/v1/datasets/'
ROOT = Path(__file__).resolve().parent / 'source'

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(API + 'view/' + HANDLE, timeout=90)
    response.raise_for_status()
    metadata = response.json()
    (ROOT / 'dataset_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    version = metadata['currentVersionNumber']
    files, token = [], None
    while True:
        params = {'pageSize': 200}
        if token:
            params['pageToken'] = token
        response = session.get(API + 'list/' + HANDLE, params=params, timeout=90)
        response.raise_for_status()
        page = response.json()
        files.extend(page['datasetFiles'])
        token = page.get('nextPageToken')
        if not token:
            break
    (ROOT / 'dataset_file_inventory.json').write_text(json.dumps(files, indent=2), encoding='utf8')
    selected = [f['name'] for f in files if
                (f['name'].endswith(('.json', '.txt', '.py')) or
                 ('metrics' in f['name'] and f['name'].endswith('.csv')))]
    def download(name):
        response = requests.get(API + 'download/' + HANDLE + '/' + requests.utils.quote(name, safe=''),
                                params={'datasetVersionNumber': version}, timeout=120)
        response.raise_for_status()
        data = response.content
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        downloads = list(pool.map(download, selected))
    provenance = {'dataset': HANDLE, 'version': version, 'listed_files': len(files), 'downloads': downloads}
    (ROOT / 'download_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf8')
    print(json.dumps(provenance, indent=2))

if __name__ == '__main__':
    main()
