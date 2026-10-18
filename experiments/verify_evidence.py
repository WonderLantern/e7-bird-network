"""Verify only procedural physical evidence; no device/session buffers are read."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert not out.exists() and ROOT not in out.parents and a.runtime.resolve() not in out.parents
    index=json.loads((ROOT/'evidence-index.json').read_text());rows=[]
    assert hashlib.sha256((a.runtime/'fixtures/workspace-mask.npy').read_bytes()).hexdigest()==index['workspace_mask_sha256']
    for row in index['cases']:
        d=ROOT/'synthetic-evidence'/row['name'];target=a.runtime/'fixtures'/row['name']
        for name,pin in row['files'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==pin,name
        assert (d/'input-chw-u8.bin').read_bytes()==(target/'input-chw-u8.bin').read_bytes()
        raw=np.load(d/'expected.npz',allow_pickle=False);fixture=np.load(target/'expected.npz',allow_pickle=False)
        differences={k:int((raw[k]!=fixture[k]).sum()) for k in ['fine','coarse','defined_workspace']};assert not any(differences.values())
        rows.append(dict(name=row['name'],differences=differences,passed=True))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(passed=True,cases=rows),indent=2),encoding='utf8');print(json.dumps(dict(passed=True,cases=len(rows))))
if __name__=='__main__':main()
