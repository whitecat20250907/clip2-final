import json

def selected_responses(text,wanted):
 assert text.lstrip().startswith('{');i=text.index('{')+1;found={}
 while i<len(text):
  while text[i].isspace()or text[i]==',':i+=1
  if text[i]=='}':break
  assert text[i]=='"';a=i;i+=1;escaped=False
  while True:
   c=text[i];i+=1
   if escaped:escaped=False
   elif c=='\\':escaped=True
   elif c=='"':break
  key=json.loads(text[a:i])
  while text[i].isspace():i+=1
  assert text[i]==':';i+=1
  while text[i].isspace():i+=1
  a=i;assert text[i]in '[{';depth=0;quoted=False;escaped=False
  while i<len(text):
   c=text[i];i+=1
   if quoted:
    if escaped:escaped=False
    elif c=='\\':escaped=True
    elif c=='"':quoted=False
   elif c=='"':quoted=True
   elif c in '[{':depth+=1
   elif c in ']}':
    depth-=1
    if depth==0:break
  if key in wanted:
   assert key not in found;value=json.loads(text[a:i]);assert isinstance(value,list)and len(value)==1 and isinstance(value[0].get('response'),str);found[key]=value[0]['response'].strip()
 assert set(found)==wanted
 return found
