"""Original resource744 network with the experimentally matched E7 profile."""
from pathlib import Path
import hashlib
import numpy as np
from core import PhysicalStream,torch

class E7BirdNetwork:
    width=384
    height=256
    def __init__(self,threads=4):
        if not 1<=threads<=32:raise ValueError('threads must be1..32')
        torch.set_num_threads(threads)
        self.model=PhysicalStream()

    def infer_planar(self,raw,include_workspace=False):
        if not isinstance(raw,(bytes,bytearray,memoryview)) or len(raw)!=294912:
            raise ValueError('Expected294912 bytes: u8 planar[3,256,384]')
        image=np.frombuffer(raw,np.uint8).reshape(1,3,256,384).astype(np.float32)/256
        holder={}
        def trace(batch,layer,bank,x,y,z,banks,work):holder['work']=work
        with torch.inference_mode():
            outputs=self.model(torch.from_numpy(image),trace if include_workspace else None)
        heads={name:h[0].numpy().astype('<f2').view('<u2').copy() for name,h in zip(('fine','coarse'),outputs)}
        if not all(np.isfinite(a.view('<f2')).all() for a in heads.values()):raise RuntimeError('Nonfinite network output')
        result=dict(heads=heads,input_sha256=hashlib.sha256(raw).hexdigest())
        if include_workspace:
            with np.errstate(over='ignore',invalid='ignore'):
                # The host model reserves a larger scratch canvas. The
                # original744 request exposes only0x60000 bytes of work RAM.
                result['workspace']=holder['work'][0,:0x60000//2].numpy().astype('<f2').view('<u2').copy()
        return result

def decode_heads(heads):
    from consumer import Consumer,pack_heads
    buffer=pack_heads(heads['fine'].view('<f2'),heads['coarse'].view('<f2'))
    result=Consumer().consume(buffer)
    if result['status']!=0:raise RuntimeError(f'Decoder failed: {result}')
    return result,buffer
