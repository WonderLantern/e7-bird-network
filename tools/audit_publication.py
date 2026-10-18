"""Check the publication allowlist and every reachable Git blob."""
from pathlib import Path
import hashlib,json,subprocess,re
ROOT=Path(__file__).resolve().parents[1]
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def validate(name,data,allowed):
    assert name in allowed,name
    assert not re.search(r'(?i)(\.pem$|\.key$|\.html?$|\.pdf$|\.exe$|\.cab$|\.zip$|\.whl$|\.tar|\.7z$)',name),name
    assert not re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',data),name
    assert not re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',data),name
    if name.endswith(('.py','.c','.json','.txt')):
        text=data.decode('utf8');patterns=[r'[A-Z]:[/\\](?:Users|CodexWork)','user'+'_scope','e7_common_'+'profile_assumed','per'+' user','user'+' agreement'];assert not re.search('(?i)('+'|'.join(patterns)+')',text),name
    return hashlib.sha256(data).hexdigest()
def main():
    manifest=json.loads((ROOT/'publication-manifest.json').read_text());files=manifest['files'];allowed=set(files)|{'publication-manifest.json'}
    for name,pin in files.items():assert validate(name,(ROOT/name).read_bytes(),allowed)==pin,name
    blobs=set();commits=0
    if (ROOT/'.git').exists():
        tracked={n.decode() for n in git('ls-files','-z').split(b'\0') if n};assert tracked==allowed
        for commit in git('rev-list','--all').decode().splitlines():
            commits+=1
            for row in git('ls-tree','-rz',commit).split(b'\0'):
                if not row:continue
                meta,name=row.split(b'\t',1);mode,kind,oid=meta.decode().split();name=name.decode();assert kind=='blob' and mode in {'100644','100755'}
                if (oid,name) not in blobs:validate(name,git('cat-file','blob',oid),allowed);blobs.add((oid,name))
        for row in git('ls-files','--stage','-z').split(b'\0'):
            if not row:continue
            meta,name=row.split(b'\t',1);mode,oid,stage=meta.decode().split();name=name.decode();assert stage=='0'
            data=git('cat-file','blob',oid);assert validate(name,data,allowed)==hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    print(json.dumps(dict(passed=True,reviewed_files=len(allowed),history_commits=commits,history_blobs=len(blobs))))
if __name__=='__main__':main()
