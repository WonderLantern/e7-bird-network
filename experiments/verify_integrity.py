"""Verify every delivered package file using Python standard library only."""
from pathlib import Path
import hashlib,json
def main():
    root=Path(__file__).resolve().parent
    m=json.loads((root/'PACKAGE_MANIFEST.json').read_text(encoding='utf8'))
    total=0
    for row in m['files']:
        p=root/row['path'];h=hashlib.sha256();n=0
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block);n+=len(block)
        if (n,h.hexdigest())!=(row['bytes'],row['sha256']):raise SystemExit('FAIL: '+row['path'])
        total+=n
    print(json.dumps(dict(passed=True,files=len(m['files']),bytes=total)))
if __name__=='__main__':main()
