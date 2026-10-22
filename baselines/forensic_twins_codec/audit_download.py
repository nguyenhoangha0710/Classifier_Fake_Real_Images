"""Read-only Kaggle artifact download for the user-requested training audit."""
from pathlib import Path
from urllib.parse import quote
import requests
import io
import zipfile

OUT=Path(__file__).resolve().parent/'artifacts/training_audit_20261020'
OUT.mkdir(parents=True,exist_ok=True)
BASE='https://www.kaggle.com/api/v1/datasets/download/hoangha071005/checkpoint-encoder-twins-codec-200k/'
for name in ['config.json','run_metadata.json','logs/epoch_metrics.csv','logs/train_steps.csv','encoder_final.pt','encoder_final.pt.sha256']:
    target=OUT/Path(name).name
    response=requests.get(BASE+quote('forensic_twins_codec/'+name,safe=''),timeout=90)
    response.raise_for_status()
    data=response.content
    if data[:2]==b'PK':
        try:
            archive=zipfile.ZipFile(io.BytesIO(data))
            if any(x.endswith('.csv') or x.endswith('.json') for x in archive.namelist()):
                data=archive.read(next(x for x in archive.namelist() if x.endswith(Path(name).name)))
        except zipfile.BadZipFile: pass
    target.write_bytes(data)
    print(name,len(data),response.headers.get('Content-Type'),flush=True)
