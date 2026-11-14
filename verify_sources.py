from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
rows=json.loads((r/'provenance/source_files.json').read_text())
for row in rows:
 p=r/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['file']
print('Verified original source/interface files:',len(rows))
for p in (r/'archive/4090_dependencies').glob('*.py'):
 assert hashlib.sha256(p.read_bytes()).hexdigest().startswith(p.name.split('_')[0]),p.name
print('Verified additional 4090 scientific source files')
