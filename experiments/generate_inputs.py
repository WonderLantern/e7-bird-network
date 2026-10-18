"""Regenerate both complete public physical input tensors from procedures."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert not out.exists() and a.runtime.resolve() not in out.parents
    c,y,x=np.indices((3,256,384));generated=dict(stripe=((x+3*y+71*c)&255).astype(np.uint8).tobytes(),random=np.random.default_rng(20261018).integers(0,256,size=(3,256,384),dtype=np.uint8).tobytes())
    out.mkdir(parents=True);rows=[]
    for name,raw in generated.items():
        expected=(a.runtime/'fixtures'/name/'input-chw-u8.bin').read_bytes();assert raw==expected,name
        (out/(name+'-chw-u8.bin')).write_bytes(raw);rows.append(dict(name=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),exact=True))
    (out/'report.json').write_text(json.dumps(dict(passed=True,cases=rows),indent=2),encoding='utf8');print(json.dumps(dict(passed=True,synthetic_inputs=2)))
if __name__=='__main__':main()
