import pathlib,json,tarfile,hashlib,os
b=pathlib.Path(__file__).parent;d=b/'tasks/Granite_part_0'
assert len(json.loads((d/'complete.json').read_text())['rows'])==2189
old=b/'native_export.tar.gz'
if old.exists():old.rename(b/'native_export.tar.gz.partial_timeout')
files=sorted(d.glob('*.npz'))+sorted(d.glob('*.json'))
assert len(list(d.glob('*.npz')))==2189
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
(b/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
partial=b/'native_export.tar.gz.partial'
with tarfile.open(partial,'w:gz')as z:
 for p in files:z.add(p,arcname=p.name)
 z.add(b/'EXPORT_MANIFEST.json',arcname='EXPORT_MANIFEST.json')
partial.rename(old)
r={'files':len(files),'rows':2189,'bytes':old.stat().st_size,'SHA':hashlib.sha256(old.read_bytes()).hexdigest()}
(b/'export_complete.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r),flush=True)
