"""Download small metadata and predictions from a pinned Kaggle evaluation version."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent/'source'
HANDLE='hoangha071005/gsd-train-bfree-test-tiny'
RUN='gsd_bfree_samecat_origbg_best_tiny9_v1/'
API='https://www.kaggle.com/api/v1/datasets/'

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    r=requests.get(API+'view/'+HANDLE,timeout=60);r.raise_for_status();metadata=r.json()
    version=metadata['currentVersionNumber']
    (ROOT/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    files=[];token=None
    while True:
        params={'pageSize':200}
        if token:params['pageToken']=token
        r=requests.get(API+'list/'+HANDLE,params=params,timeout=60);r.raise_for_status();page=r.json()
        files.extend(page['datasetFiles']);token=page.get('nextPageToken')
        if not token:break
    (ROOT/'dataset_file_inventory.json').write_text(json.dumps(files,indent=2),encoding='utf8')
    wanted={'metrics.csv','macro_all.csv','predictions.csv','delta_from_native.csv','paired_summary.csv','test_manifest.csv'}
    selected=[f['name'] for f in files if f['name'].startswith(RUN) and '/progress/' not in f['name']
              and (f['name'].endswith('.json') or Path(f['name']).name in wanted)]
    def get(name):
        r=requests.get(API+'download/'+HANDLE+'/'+requests.utils.quote(name,safe=''),params={'datasetVersionNumber':version},timeout=120)
        r.raise_for_status();p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
        return dict(name=name,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:downloads=list(pool.map(get,selected))
    result=dict(dataset=HANDLE,version=version,listed_files=len(files),downloads=downloads)
    (ROOT/'download_provenance.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
