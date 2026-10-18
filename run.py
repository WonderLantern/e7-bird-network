"""Run original-weight E7 resource744 inference; network heads are always saved."""
from pathlib import Path
import argparse,json,time,sys
import numpy as np
from PIL import Image,ImageDraw
from network import E7BirdNetwork,decode_heads

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--planar-u8',action='store_true',help='Exact u8 CHW[3,256,384] input')
    p.add_argument('--heads-only',action='store_true',help='Save network heads without the optional Windows candidate decoder')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,required=True,help='Fresh output directory outside this package')
    a=p.parse_args();root=Path(__file__).resolve().parent;out=a.output.resolve()
    if out==root or root in out.parents:p.error('Output must be outside the package')
    if out.exists():p.error('Output directory already exists; choose a fresh directory')
    if not a.heads_only and sys.platform!='win32':p.error('Use --heads-only for this platform; the bundled candidate decoder is Windows x64')
    source=a.input.read_bytes()
    if a.planar_u8:
        if len(source)!=294912:p.error('Planar input must contain exactly294912 bytes')
        raw=source;display=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(3,256,384).transpose(1,2,0))
        adapter='Exact supplied u8 planes, no resize or color conversion'
    else:
        with Image.open(a.input) as im:display=im.convert('RGB').resize((384,256),Image.Resampling.BILINEAR)
        raw=np.asarray(display).transpose(2,0,1).copy().tobytes()
        adapter='Pillow RGB bilinear stretch to384x256; standalone image adapter'
    started=time.monotonic();result=E7BirdNetwork(a.threads).infer_planar(raw);heads=result.pop('heads')
    report=dict(profile='E7-resource744-original-weights',resource=744,input_sha256=result['input_sha256'],input_layout='u8 planar[3,256,384]',network_input='u/256',image_adapter=adapter,head_format='uint16 IEEE-half bit patterns; includes signed zero',head_shapes={k:list(v.shape) for k,v in heads.items()},logical_head_halves=11616,network_seconds=time.monotonic()-started,validation_source='ZR C1.10 original744; body byte-identical to Z9 resource744')
    if not a.heads_only:
        decoded,buffer=decode_heads(heads);report.update(records=decoded['records'],record_fields=['cx','cy','w','h','native_integer_score','unnamed_class_slot'],decoder_status=decoded['status'])
    out.mkdir(parents=True,exist_ok=False);(out/'input-chw-u8.bin').write_bytes(raw);np.savez_compressed(out/'heads.npz',**heads)
    if not a.heads_only:
        (out/'head-buffer.bin').write_bytes(buffer)
        draw=ImageDraw.Draw(display)
        for x,y,w,h,score,slot in report['records']:
            draw.rectangle((x-w/2,y-h/2,x+w/2,y+h/2),outline='red',width=2)
            draw.text((max(0,x-w/2),max(0,y-h/2)),f'slot{slot}: {score}',fill='red',stroke_width=1,stroke_fill='white')
        display.save(out/'detections.png')
    (out/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(dict(output=str(out),logical_head_halves=11616,records=report.get('records'),seconds=report['network_seconds']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
