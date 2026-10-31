"""Read-only public Kaggle metadata/file checks for URLs found in local artifacts."""
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
PIXEL = {'awsaf49/coco-2017-dataset', 'baopv051/b-free-training-data', 'yangsangtai/tiny-genimage'}

def check(handle):
    record = {'handle': handle, 'url': 'https://www.kaggle.com/datasets/' + handle,
              'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        r = requests.get('https://www.kaggle.com/api/v1/datasets/view/' + handle, timeout=45)
        record['http_status'] = r.status_code
        if r.status_code != 200:
            record['error_preview'] = r.text[:250]
            return record
        d = r.json()
        record['metadata'] = d
        if handle not in PIXEL:
            r = requests.get('https://www.kaggle.com/api/v1/datasets/list/' + handle,
                             params={'pageSize': 200}, timeout=45)
            record['list_http_status'] = r.status_code
            if r.status_code == 200:
                page = r.json()
                record['files_first_page'] = page.get('datasetFiles', [])
                record['more_files'] = page.get('hasNextPageToken', False)
        return record
    except Exception as exc:
        record['error'] = str(exc)
        return record

def main():
    evidence = json.loads((ROOT / 'local_url_evidence.json').read_text(encoding='utf8'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(check, sorted(evidence)))
    (ROOT / 'remote_url_checks.json').write_text(json.dumps(records, indent=2), encoding='utf8')
    for r in records:
        d = r.get('metadata', {})
        print(r['handle'], r.get('http_status', r.get('error')), 'v'+str(d.get('currentVersionNumber','?')),
              'listed='+str(len(r.get('files_first_page',[]))), 'more='+str(r.get('more_files')))

if __name__ == '__main__':
    main()
