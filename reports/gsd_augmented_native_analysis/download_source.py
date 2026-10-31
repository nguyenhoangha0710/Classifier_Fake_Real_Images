"""Download only public result tables/metadata, never the 1.2GB checkpoint."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request

HANDLE = 'hoangha071005/evaluate-augmented-tiny-gsd-new'
ROOT = Path(__file__).resolve().parent / 'source'
API = 'https://www.kaggle.com/api/v1/datasets/'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent':'GSD-result-audit/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata_bytes = fetch(API + 'view/' + HANDLE)
    metadata = json.loads(metadata_bytes)
    version = metadata['currentVersionNumber']
    (ROOT/'dataset_metadata.json').write_bytes(metadata_bytes)
    files, token = [], ''
    while True:
        query = urllib.parse.urlencode(dict(pageSize=1000, pageToken=token))
        page = json.loads(fetch(API + 'list/' + HANDLE + '?' + query))
        files.extend(page['datasetFiles'])
        token = page.get('nextPageToken')
        if not token:
            break
    (ROOT/'dataset_file_inventory.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    selected = [f['name'] for f in files if '/progress/' not in f['name'] and
                f['name'].startswith(('gsd_native/', 'gsd_smoke/')) and
                (f['name'].endswith('.json') or f['name'].endswith('/metrics.csv') or
                 f['name'].endswith('/predictions_native.csv'))]
    selected.append('gsd_native_results/gsd_native/predictions_native.csv')
    def download(name):
        target = ROOT / name
        Path(os.path.abspath(target)).relative_to(Path(os.path.abspath(ROOT)))
        target.parent.mkdir(parents=True, exist_ok=True)
        url = API + 'download/' + HANDLE + '/' + urllib.parse.quote(name, safe='')
        data = fetch(url + '?datasetVersionNumber=' + str(version))
        target.write_bytes(data)
        return dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        downloaded = list(executor.map(download, selected))
    provenance = dict(dataset=HANDLE, version=version, downloads=downloaded,
                      listed_files=len(files), listed_bytes=sum(f['totalBytes'] for f in files))
    (ROOT/'download_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(dict(version=version, listed_files=len(files), downloaded=downloaded), indent=2))


if __name__ == '__main__':
    main()
