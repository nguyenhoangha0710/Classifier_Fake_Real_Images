import concurrent.futures
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
TARGETS = ['hoangha071005/evaluation-twins-bfree', 'hoangha071005/fit-gmm-checkpoint-200k',
           'hoangha071005/chech-point-gmm-spectral']
SMALL_FILES = {
    'hadinho0710/checkpoint-gsd-sdv5-train': ['gsd_raw_all7_crop560_v1/config.json'],
    'hoangha071005/check-point-gsd-train-bfee-samecat-ori': [
        'gsd_bfree_samecat_origbg_v1/config.json', 'gsd_bfree_samecat_origbg_v1/split_summary.json',
        'gsd_bfree_samecat_origbg_v1/selected_checkpoint.json', 'gsd_bfree_samecat_origbg_v1/run_state.json'],
    'hoangha071005/checkpoints-clip-npr-aide': ['checkpoints/checkpoint_manifest.json'],
}

def inventory(handle):
    files, token, pages = [], None, 0
    for _ in range(20):
        params = {'pageSize': 200}
        if token:
            params['pageToken'] = token
        r = requests.get('https://www.kaggle.com/api/v1/datasets/list/' + handle, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        files.extend(d['datasetFiles'])
        token = d.get('nextPageToken')
        pages += 1
        if not token:
            break
    record = {'handle': handle, 'pages': pages, 'complete': not bool(token), 'files': files}
    (ROOT / (handle.replace('/', '__') + '__inventory.json')).write_text(json.dumps(record,indent=2),encoding='utf8')
    selected = [f['name'] for f in files if any(t in f['name'].lower() for t in
                ['manifest', 'threshold', 'metrics', 'summary', 'gmm', 'config']) and
                not any(t in f['name'].lower() for t in ['embedding','per_generator/','_shard','\.py'])]
    return handle, len(files), not bool(token), selected

def download(item):
    handle,name=item
    r=requests.get('https://www.kaggle.com/api/v1/datasets/download/'+handle+'/'+requests.utils.quote(name,safe=''),
                   params={'datasetVersionNumber': 2 if handle=='hadinho0710/checkpoint-gsd-sdv5-train' else 1},timeout=60)
    r.raise_for_status()
    p=ROOT/'small_remote_files'/handle/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(r.content)
    return handle,name,len(r.content)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for handle,n,complete,names in pool.map(inventory,TARGETS):
            print(handle,'files',n,'complete',complete)
            for name in names:
                if not name.endswith(('.py','.ipynb')): print(' ',name)
        for out in pool.map(download,[(h,n) for h,ns in SMALL_FILES.items() for n in ns]):print(out)
