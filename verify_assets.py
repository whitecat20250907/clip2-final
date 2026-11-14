from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
manifest=json.loads((r/'provenance/release_assets.json').read_text())
for row in manifest:
 p=r/'assets'/row['name'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert p.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256'],str(p)
 print('OK',p.name)
