"""Retrieve public evaluation CSV/JSON only; skip checkpoints and duplicated progress."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request

HANDLE = 'hoangha071005/gsd-test-nine-cases'
ROOT = Path(__file__).resolve().parent/'source'
API = 'https://www.kaggle.com/api/v1/datasets/'
RUN = 'gsd_uploaded_best_variants_v1'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent':'GSD-nine-case-audit/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    raw_metadata = fetch(API+'view/'+HANDLE)
    version = json.loads(raw_metadata)['currentVersionNumber']
    (ROOT/'dataset_metadata.json').write_bytes(raw_metadata)
    files, token = [], ''
    while True:
        page = json.loads(fetch(API+'list/'+HANDLE+'?'+urllib.parse.urlencode(dict(pageSize=1000,pageToken=token))))
        files.extend(page['datasetFiles'])
        token = page.get('nextPageToken')
        if not token:
            break
    (ROOT/'dataset_file_inventory.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    wanted = {'predictions.csv','metrics.csv','paired_summary.csv','macro_all.csv','delta_from_native.csv'}
    selected = [item['name'] for item in files if item['name'].startswith(RUN+'/') and
        '/progress/' not in item['name'] and
        (item['name'].endswith('.json') or Path(item['name']).name in wanted)]
    def download(name):
        target = ROOT/name
        Path(os.path.abspath(target)).relative_to(Path(os.path.abspath(ROOT)))
        target.parent.mkdir(parents=True,exist_ok=True)
        data = fetch(API+'download/'+HANDLE+'/'+urllib.parse.quote(name,safe='')+
                     '?datasetVersionNumber='+str(version))
        target.write_bytes(data)
        return dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        downloaded = list(executor.map(download,selected))
    provenance = dict(dataset=HANDLE,version=version,listed_files=len(files),downloads=downloaded)
    (ROOT/'download_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2))


if __name__ == '__main__':
    main()
