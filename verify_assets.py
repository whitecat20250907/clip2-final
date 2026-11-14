from pathlib import Path
import argparse,hashlib,json,os

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 r=Path(__file__).resolve().parent
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=r/'assets');a=ap.parse_args()
 manifest=json.loads((r/'provenance/release_assets.json').read_text())
 for row in manifest:
  assert Path(row['name']).name==row['name']
  p=a.directory/row['name']
  if not p.exists() and row.get('parts'):
   tmp=p.with_name(p.name+'.assembling');whole=hashlib.sha256();size=0
   with tmp.open('wb') as out:
    for part in row['parts']:
     assert Path(part['name']).name==part['name']
     q=a.directory/part['name'];h=hashlib.sha256();n=0
     with q.open('rb') as f:
      for b in iter(lambda:f.read(8*1024*1024),b''):
       out.write(b);h.update(b);whole.update(b);n+=len(b);size+=len(b)
     assert n==part['bytes'] and h.hexdigest()==part['sha256'],str(q)
   assert size==row['bytes'] and whole.hexdigest()==row['sha256'],str(p)
   os.replace(tmp,p)
  assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],str(p)
  print('OK',p.name,flush=True)
if __name__=='__main__':main()
