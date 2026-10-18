"""Original resource744 replay, matched on three Nikon ZR C1.10 inputs.
This is a finite-input behavioral profile, not vendor RTL or a Z9 hardware proof.
Only CPU torch operations are used. The weights and command order are unchanged.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent

def half(x, ftz=False):
    y = x.to(torch.float16).to(torch.float32)
    if ftz:
        y = torch.where(y.abs() < 2.**-14, torch.copysign(torch.zeros_like(y), y), y)
    return y


def rounded(x,mode='rne',ftz=False):
    # Keep direct double->half conversion explicit; Torch CPU can mediate via f32.
    h=torch.from_numpy(x.detach().numpy().astype(np.float16)) if x.dtype==torch.float64 else x.to(torch.float16)
    if mode=='rtz':h=torch.where(h.float().abs()>x.abs(),(h.view(torch.int16)-1).view(torch.float16),h)
    elif mode=='rmm':
        a=h.abs();up=(a.view(torch.int16)+1).view(torch.float16)
        mid=(a.double()+up.double())*.5
        increment=(x.abs().double()==mid)&torch.isfinite(mid)
        h=torch.where(increment,torch.copysign(up,h),h)
    else:assert mode=='rne'
    y=h.float()
    if ftz:y=torch.where(y.abs()<2.**-14,torch.copysign(torch.zeros_like(y),y),y)
    return y


def add_variant(a,b,g,mode,ftz=False):
    aa,bb=torch.broadcast_tensors(a.half().view(torch.int16).int()&65535,b.half().view(torch.int16).int()&65535)
    ea=((aa>>10)&31).clamp_min(1);eb=((bb>>10)&31).clamp_min(1);e=torch.maximum(ea,eb)
    def align(bits,delta):
        m=((bits&1023)|torch.where((bits&0x7c00)!=0,1024,0))<<g;neg=(bits&32768)!=0
        if mode=='signed_floor':return torch.where(neg,-m,m)>>delta
        v=m>>delta;lost=(m&((1<<delta)-1))!=0
        if mode=='jam':v=v|lost.int()
        elif mode=='nearest':v=(m+torch.where(delta>0,1<<(delta-1).clamp_min(0),0))>>delta
        else:assert mode=='magnitude'
        return torch.where(neg,-v,v)
    n=align(aa,e-ea)+align(bb,e-eb);value=torch.ldexp(n.float(),e-25-g)
    value=torch.where((n==0)&((aa&bb&32768)!=0),-torch.zeros_like(value),value)
    return rounded(value,'rtz',ftz)


def apply_pcnn_candidate(x,lanes,mode='gradual'):
    assert mode in ('gradual','negative_pre_round_ftz','negative_post_round_ftz')
    y=x.clone()
    for i,(kind,sign,shift) in enumerate(lanes):
        if kind==0:continue
        if kind==1:y[:,i]=torch.clamp_min(x[:,i],0)
        elif kind==2:
            z=x[:,i]*((-1. if sign else 1.)*2.**-shift)
            if mode=='negative_pre_round_ftz':
                z=torch.where(z.abs()<2.**-14,torch.copysign(torch.zeros_like(z),z),z)
            elif mode=='negative_post_round_ftz':
                z=z.half().float()
                z=torch.where(z.abs()<2.**-14,torch.copysign(torch.zeros_like(z),z),z)
            y[:,i]=torch.where(x[:,i]<0,z,x[:,i])
        else:raise ValueError(kind)
    return y


POLICY={'input_half': True, 'bank_half': True, 'work_half': True, 'activation_mode': 'negative_pre_round_ftz', 'dot': 'half_tree', 'acc_half': True, 'rounding': 'rtz', 'bias_position': 'first', 'tree_guard': 4, 'carry_guard': 4, 'tree_align': 'magnitude', 'carry_align': 'magnitude', 'product_ftz': True, 'add_ftz': True, 'zero_tie': 'last'}
round_half=rounded
class PhysicalStream:
    def __init__(self):
        manifest=json.loads((ROOT/'manifest.json').read_text())
        for name,pin in manifest['files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
        data=json.loads((ROOT/'commands.json').read_text());raw=(ROOT/'resource744.body.bin').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==data['body_sha256']
        self.events=data['events'];self.audit=data['audit'];self.policy=dict(POLICY)
        self.activations={int(k):v for k,v in data['activations'].items()}
        for p,kind,v in self.events:
            if kind!='matrix':continue
            b=v['block'];offset=int(b['data_body_offset'],16)
            v['weight']=torch.from_numpy(np.frombuffer(raw,'<f2',256,offset).astype(np.float32).reshape(16,16))
            offset=b['bias_body_offset']
            v['bias']=torch.from_numpy(np.frombuffer(raw,'<f2',16,int(offset,16)).astype(np.float32)) if offset else None

    @staticmethod
    def shape(hw,edge):
        h,w,stride=hw
        return ((h+bool(edge&1)+bool(edge&4)+stride-1)//stride,
                (w+bool(edge&16)+bool(edge&64)+stride-1)//stride)

    def zero_add(self,a,b):
        z=add_variant(a,b,4,'magnitude',True)
        # Sign-magnitude adders retain the larger-magnitude operand's sign on
        # cancellation/underflow; tie selection is tested independently.
        aa,bb=torch.broadcast_tensors(a,b)
        choose=(aa.abs()>=bb.abs()) if self.policy['zero_tie']=='first' else (aa.abs()>bb.abs())
        sign=torch.where(choose,aa,bb)
        return torch.where(z==0,torch.copysign(torch.zeros_like(z),sign),z)

    def add_carry(self,a,b):return self.zero_add(a,b)

    def dot(self,w,x):
        z=rounded(w[None,:,:,None,None]*x[:,None,:,:,:],'rtz',True)
        while z.shape[2]>1:z=self.zero_add(z[:,:,0::2],z[:,:,1::2])
        return z[:,:,0]

    def __call__(self, pixels, trace=None):
        assert tuple(pixels.shape[1:]) == (3, 256, 384)
        assert not self.audit['failures']
        pol = self.policy
        n = len(pixels)
        banks = torch.zeros(2, n, 16, 256, 256)
        work = torch.zeros(n, 1048576 // 2)
        head = torch.zeros(n, 25344 // 2)
        input_flat = pixels.reshape(n, -1)
        if pol.get('input_half'):
            input_flat = half(input_flat, pol.get('ftz', False))
        edge = 0
        hw = None
        acc = None
        pending_bias = None
        for p, kind, v in self.events:
            if kind == 'matrix':
                b = v['block']
                flags = int(b['flags'], 16)
                address = int(b['address'], 16)
                srcbank = flags >> 8 & 1
                dstbank = 1 - srcbank
                h, w = self.shape(hw, edge)
                stride = hw[2]
                sx = (address >> 8 & 255) - bool(edge & 16)
                sy = (address & 255) - bool(edge & 1)
                ys = sy + torch.arange(h) * stride
                xs = sx + torch.arange(w) * stride
                ry = torch.arange(h) * stride - int(bool(edge & 1))
                rx = torch.arange(w) * stride - int(bool(edge & 16))
                valid_y = (ys >= 0) & (ys < 256) & (ry >= 0) & (ry < hw[0])
                valid_x = (xs >= 0) & (xs < 256) & (rx >= 0) & (rx < hw[1])
                x = banks[srcbank][:, :, ys.clamp(0, 255)[:, None], xs.clamp(0, 255)[None, :]].clone()
                x *= valid_y[None, None, :, None] * valid_x[None, None, None, :]
                part = self.dot(v['weight'], x)
                if pol.get('rounding') == 'rtz':
                    part = part.double()
                if pol.get('packet_half'):
                    part = half(part, pol.get('ftz', False))
                if flags & 512:
                    assert acc is None and pending_bias is None
                    if pol.get('bias_position', 'first') == 'last':
                        pending_bias = v['bias'][None, :, None, None]
                        acc = part
                    else:
                        acc = self.add_carry(v['bias'][None, :, None, None], part)
                else:
                    assert acc is not None and acc.shape == part.shape
                    acc = self.add_carry(acc, part)
                postpone = bool(flags & 1024) and pol.get('final_round', 'before_activation') == 'after_activation'
                if pol.get('acc_half') and (not postpone):
                    acc = round_half(acc, pol.get('rounding', 'rne'), pol.get('ftz', False))
                if flags & 1024:
                    if pending_bias is not None:
                        if pol.get('rounding') == 'rtz':
                            acc = acc.double()
                        acc = self.add_carry(acc, pending_bias)
                        pending_bias = None
                        if pol.get('acc_half') and (not postpone):
                            acc = round_half(acc, pol.get('rounding', 'rne'), pol.get('ftz', False))
                    acc = apply_pcnn_candidate(acc, self.activations[v['batch']], pol.get('activation_mode', 'gradual'))
                    if pol.get('bank_half'):
                        acc = round_half(acc, pol.get('rounding', 'rne'), pol.get('ftz', False))
                    ox = address >> 24 & 255
                    oy = address >> 16 & 255
                    assert oy + h <= 256 and ox + w <= 256
                    banks[dstbank, :, :, oy:oy + h, ox:ox + w] = acc
                    if trace is not None:
                        trace(v['batch'], v['layer'], dstbank, ox, oy, acc, banks, work)
                    acc = None
                continue
            words = v['words']
            op = words[0]
            major = op >> 8
            if major == 76:
                hw = (op & 255, words[1] & 4095, 2 if words[1] & 4096 else 1)
            elif major == 112:
                edge = op & 255
            elif major == 68:
                pass
            elif major in (32, 34, 48, 49, 50, 51):
                h, w, _ = hw
                addr = words[1] << 16 | words[2]
                pitch = words[3]
                sx, sy, mask = (words[4] >> 8, words[4] & 255, words[5])
                assert mask and mask & mask - 1 == 0
                lane = mask.bit_length() - 1
                if major == 32:
                    index = addr + torch.arange(h)[:, None] * pitch + torch.arange(w)[None, :]
                    banks[0, :, lane, sy:sy + h, sx:sx + w] = input_flat[:, index]
                else:
                    assert addr % 2 == 0 and pitch % 2 == 0
                    index = addr // 2 + torch.arange(h)[:, None] * (pitch // 2) + torch.arange(w)[None, :]
                    if major == 34:
                        banks[0, :, lane, sy:sy + h, sx:sx + w] = work[:, index]
                    else:
                        z = banks[major & 1, :, lane, sy:sy + h, sx:sx + w]
                        if major in (50, 51):
                            if pol.get('work_half'):
                                z = half(z, pol.get('ftz', False))
                            work[:, index] = z
                        else:
                            head[:, index] = z
            elif major in (104, 105):
                h, w, _ = hw
                sx, sy = (words[1] >> 8, words[1] & 255)
                ox, oy = (words[2] >> 8, words[2] & 255)
                z = F.max_pool2d(banks[major & 1, :, :, sy:sy + h, sx:sx + w], 2)
                banks[1 - (major & 1), :, :, oy:oy + z.shape[-2], ox:ox + z.shape[-1]] = z
            elif op in (0, 16512):
                pass
            else:
                raise ValueError((hex(p), hex(op)))
        assert acc is None
        return (head[:, :16896 // 2].reshape(n, 22, 16, 24), head[:, 16896 // 2:].reshape(n, 33, 8, 16)[:, :, :, :12])
