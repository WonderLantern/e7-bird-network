"""Offline bitwise regression against captured physical inputs and outputs."""
from pathlib import Path
import argparse,json,hashlib,time,platform,sys
import numpy as np
from network import E7BirdNetwork,decode_heads
from core import torch,rounded
from arithmetic_oracle import oracle_add,oracle_mul
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arithmetic(model):
    rng=np.random.default_rng(20261019);a=rng.integers(0,65536,60000,dtype=np.uint16);b=rng.integers(0,65536,len(a),dtype=np.uint16)
    valid=((a&0x7c00)!=0x7c00)&((b&0x7c00)!=0x7c00);a=a[valid];b=b[valid];finite=np.arange(0x7c00,dtype=np.uint16)
    a=np.concatenate([a,finite,finite|32768]);b=np.concatenate([b,finite|32768,finite])
    x=torch.from_numpy(a.view(np.float16).astype(np.float32));y=torch.from_numpy(b.view(np.float16).astype(np.float32))
    actual=[model.zero_add(x,y).numpy().astype(np.float16).view(np.uint16),rounded(x*y,'rtz',True).numpy().astype(np.float16).view(np.uint16)]
    counts={}
    for kind,got,oracle in zip(['add','multiply'],actual,[oracle_add,oracle_mul]):
        want=np.array([oracle(int(i),int(j)) for i,j in zip(a,b)],np.uint16);different=int((got!=want).sum());counts[kind]=dict(checks=len(a),differences=different)
        if different:raise RuntimeError(f'Integer arithmetic mismatch: {kind}: {different}')
    return counts
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',default='all');p.add_argument('--threads',type=int,default=4);p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-arithmetic',action='store_true');p.add_argument('--heads-only',action='store_true');a=p.parse_args()
    out=a.output.resolve()
    if out==ROOT or ROOT in out.parents:p.error('Output must be outside the package')
    if out.exists():p.error('Output directory exists; choose a fresh one')
    index=json.loads((ROOT/'fixtures/index.json').read_text());cases=[c for c in index['cases'] if a.case=='all' or c['name']==a.case]
    if not cases:p.error('Unknown case; use all or a name from fixtures/index.json')
    out.mkdir(parents=True,exist_ok=False);report=dict(passed=False,python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,cases=[],scope='Network and optional raw candidate decoder only; no camera, ISP, AF or hardware connection required')
    def save():(out/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    save();net=E7BirdNetwork(a.threads)
    try:
        if not a.skip_arithmetic:report['integer_arithmetic']=arithmetic(net.model);save()
        if sha(ROOT/'fixtures/workspace-mask.npy')!=index['workspace_mask_sha256']:raise RuntimeError('Workspace-mask checksum mismatch')
        mask=np.load(ROOT/'fixtures/workspace-mask.npy')
        for c in cases:
            started=time.monotonic();d=ROOT/'fixtures'/c['name']
            for name,pin in c['files'].items():
                if sha(d/name)!=pin:raise RuntimeError(f'Fixture checksum mismatch: {c["name"]}/{name}')
            raw=(d/'input-chw-u8.bin').read_bytes();got=net.infer_planar(raw,include_workspace=True);want=np.load(d/'expected.npz')
            metrics={k:int((got['heads'][k]!=want[k]).sum()) for k in ['fine','coarse']};metrics['defined_workspace']=int((got['workspace'][mask]!=want['defined_workspace']).sum())
            row=dict(case=c['name'],input_sha256=got['input_sha256'],head_halves=11616,defined_workspace_halves=int(mask.sum()),differences=metrics,seconds=time.monotonic()-started)
            if not a.heads_only:
                decoded,_=decode_heads(got['heads']);row['decoder_equal']=decoded['records']==c['expected_records'];row['records']=decoded['records']
            row['passed']=all(v==0 for v in metrics.values()) and row.get('decoder_equal',True);report['cases'].append(row);save();print(json.dumps(row),flush=True)
            if not row['passed']:raise RuntimeError(f'Physical regression mismatch: {c["name"]}')
        report['passed']=True;report['total_head_halves']=11616*len(cases);report['total_defined_workspace_halves']=int(mask.sum())*len(cases);save()
    except BaseException as exc:report['error']=repr(exc);save();raise
    print(json.dumps(dict(passed=True,cases=len(cases),output=str(out))),flush=True)
if __name__=='__main__':main()
