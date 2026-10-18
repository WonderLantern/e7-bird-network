"""Change one arithmetic behavior at a time against a frozen physical fixture."""
from pathlib import Path
import argparse,sys,json,time
import numpy as np
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--case',default='random');p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--variants',nargs='+',default=['all']);a=p.parse_args()
    root=a.runtime.resolve();out=a.output.resolve();assert not out.exists();assert root not in out.parents and Path(__file__).resolve().parent not in out.parents
    sys.path.insert(0,str(root));from core import PhysicalStream,POLICY,torch,rounded,add_variant
    class Variant(PhysicalStream):
        def add(self,x,y,g):
            if g=='ieee':z=rounded(x.double()+y.double(),'rtz',self.policy['add_ftz'])
            else:z=add_variant(x,y,g,'magnitude',self.policy['add_ftz'])
            x,y=torch.broadcast_tensors(x,y);sign=torch.where(x.abs()>y.abs(),x,y)
            return torch.where(z==0,torch.copysign(torch.zeros_like(z),sign),z)
        def add_carry(self,x,y):return self.add(x,y,self.policy['carry_guard'])
        def dot(self,w,x):
            z=rounded(w[None,:,:,None,None]*x[:,None,:,:,:],self.policy.get('product_rounding','rtz'),self.policy['product_ftz'])
            if self.policy.get('tree_permutation')=='bit_reverse':z=z[:,:,[0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]]
            while z.shape[2]>1:z=self.add(z[:,:,0::2],z[:,:,1::2],self.policy['tree_guard'])
            return z[:,:,0]
    assert 1<=a.threads<=32;torch.set_num_threads(a.threads);model=Variant()
    raw=(root/'fixtures'/a.case/'input-chw-u8.bin').read_bytes();want=np.load(root/'fixtures'/a.case/'expected.npz');mask=np.load(root/'fixtures/workspace-mask.npy');u=np.frombuffer(raw,np.uint8).reshape(1,3,256,384).astype(np.float32)
    jobs=[('control',{}),('input_divide255',{'input_divisor':255}),('tree_guard3',{'tree_guard':3}),('tree_guard5',{'tree_guard':5}),('carry_guard3',{'carry_guard':3}),('carry_guard5',{'carry_guard':5}),('carry_ieee_rtz',{'carry_guard':'ieee'}),('product_gradual',{'product_ftz':False}),('adder_gradual',{'add_ftz':False}),('bias_last',{'bias_position':'last'}),('tree_butterfly',{'tree_permutation':'bit_reverse'}),('product_rne',{'product_rounding':'rne'})]
    selected={n for n,_ in jobs} if a.variants==['all'] else set(a.variants);assert selected<={n for n,_ in jobs};jobs=[j for j in jobs if j[0] in selected];assert jobs
    out.mkdir(parents=True);report=dict(finished=False,capture_id=a.case,results=[],meaning='Control must match. Alternatives measure sensitivity on this fixture; rejection is not a proof over every possible input.')
    for name,changes in jobs:
        started=time.monotonic();model.policy=dict(POLICY,**changes);holder={}
        def trace(batch,layer,bank,x,y,z,banks,work):holder['work']=work
        with torch.inference_mode():heads=model(torch.from_numpy(u/model.policy.get('input_divisor',256)),trace)
        got={k:h[0].numpy().astype('<f2').view('<u2') for k,h in zip(['fine','coarse'],heads)}
        work=holder['work'][0,:len(mask)].numpy().astype('<f2').view('<u2');diff={k:int((got[k]!=want[k]).sum()) for k in got};diff['defined_workspace']=int((work[mask]!=want['defined_workspace']).sum())
        row=dict(name=name,changes=changes,differences=diff,seconds=time.monotonic()-started);report['results'].append(row)
        if name=='control':assert not any(diff.values()),row
        np.savez_compressed(out/(name+'.npz'),**got,defined_workspace=work[mask]);report['finished']=len(report['results'])==len(jobs)
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
