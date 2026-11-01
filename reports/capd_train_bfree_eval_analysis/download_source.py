"""Download CAPD BFree-trained evaluation tables, not weights or patch maps."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent/'source'
HANDLE='hadinho0710/eval-train-capd-bfee-test-tiny-and-bfree'
API='https://www.kaggle.com/api/v1/datasets/'

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    r=requests.get(API+'view/'+HANDLE,timeout=60);r.raise_for_status();meta=r.json();v=meta['currentVersionNumber']
    (ROOT/'dataset_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    files=[];token=None
    while True:
        params={'pageSize':200}
        if token:params['pageToken']=token
        r=requests.get(API+'list/'+HANDLE,params=params,timeout=60);r.raise_for_status();d=r.json()
        files+=d['datasetFiles'];token=d.get('nextPageToken')
        if not token:break
    (ROOT/'dataset_file_inventory.json').write_text(json.dumps(files,indent=2),encoding='utf8')
    selected=[]
    for f in files:
        n=f['name'];base=Path(n).name
        if n.endswith(('.json','.txt','.py')) or (base.endswith('.csv') and ('metrics' in base or
            (n.startswith('capd_eval_tiny/') and base.startswith('predictions_')) or base=='predictions_bfree.csv' or
            (n.startswith('capd_eval_bfree/') and base in ['train_order.csv','validation_ids.csv']))):
            if '/progress/' not in n:selected.append(n)
    def get(n):
        r=requests.get(API+'download/'+HANDLE+'/'+requests.utils.quote(n,safe=''),params={'datasetVersionNumber':v},timeout=180);r.raise_for_status()
        p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
        return dict(name=n,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:downloads=list(pool.map(get,selected))
    out=dict(dataset=HANDLE,version=v,listed_files=len(files),downloads=downloads)
    (ROOT/'download_provenance.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print('version',v,'listed',len(files),'downloaded',len(downloads))
    for f in files:
        n=f['name']
        if not any(s in n for s in ['part_','smaps_']) and any(s in n for s in ['metrics','prediction','completion','selected_checkpoint','status','order.csv']):print(n,f['totalBytes'])

if __name__=='__main__':main()
