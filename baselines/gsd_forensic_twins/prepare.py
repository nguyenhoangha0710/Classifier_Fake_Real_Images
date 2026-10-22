import argparse
import json
from pathlib import Path
from data import prepare_manifest, sha256
from progress import Progress


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--pixel-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--purpose',choices=['ssl','density'],required=True)
    p.add_argument('--audit',choices=['metadata','header','full'],default='metadata')
    p.add_argument('--log-every',type=int,default=1000)
    p.add_argument('--reuse',action='store_true',help='Reuse only if parent/output hashes and policy match')
    args=p.parse_args()
    provenance=args.output.with_suffix('.provenance.json')
    with Progress('prepare') as progress:
        progress.stage(f'parent={args.parent}; audit={args.audit}')
        if args.output.exists():
            if not args.reuse or not provenance.is_file(): raise FileExistsError(args.output)
            saved=json.loads(provenance.read_text())
            expected={'parent_sha256':sha256(args.parent),'derived_sha256':sha256(args.output),
                      'audit':args.audit,'pair':args.purpose=='ssl','pixel_root':str(args.pixel_root.resolve())}
            if any(saved.get(k)!=v for k,v in expected.items()): raise ValueError('Prepared manifest identity mismatch')
            progress.stage(f'Reusing verified eligible manifest: {args.output}; rows={saved["eligible_count"]}')
            return
        result=prepare_manifest(args.parent,args.pixel_root,args.output,args.purpose=='ssl',args.audit,args.log_every,progress)
        provenance.write_text(json.dumps(result,indent=2))
        progress.stage(f'COMPLETE: {args.output}; eligible={result["eligible_count"]}; '
                       f'rejected={len(result["rejected_ids"])}; seconds={result["seconds"]:.1f}')


if __name__=='__main__': main()
