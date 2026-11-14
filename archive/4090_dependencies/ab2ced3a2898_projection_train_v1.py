from pathlib import Path
exec(compile((Path(__file__).resolve().parent/'projection_data_v1.py').read_text(),'projection_data_v1.py','exec'))
hs=nn.ModuleDict({str(i):LowRankJoint(W,bias,82044+i)for i in [0,1,2]}).to('cuda').requires_grad_(False)
with torch.no_grad():
 for name,head in hs.items():head.A.copy_(warm['heads'][name]['A']);head.B.copy_(warm['heads'][name]['B'])
def heads_sha():
 h=hashlib.sha256()
 for k,v in sorted(hs.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
source_heads_sha=heads_sha();first=ids[0];row=cache[first];qa=qualify_bridge(hs[str(row['severity'])],torch.tensor(frames(first),device='cuda'),row['E'].to('cuda'),row['D'].to('cuda'),labels[first]);put('PROJ_TDT_BRIDGE_GRADIENT_QUALIFICATION_V1.json',qa)
ad=ProjectionPair();assert sum(v.numel()for v in ad.parameters())==23552
test=ProjectionPair(seed=140045);testloss=bridge(hs[str(row['severity'])],test,torch.tensor(frames(first)[:3],device='cuda'),row['E'][:3].to('cuda'),row['D'][:min(2,len(labels[first]))+1].to('cuda'),labels[first][:2]);assert math.isfinite(testloss)and all(torch.isfinite(v.grad).all()for v in test.parameters());del test
opt=torch.optim.AdamW(ad.parameters(),lr=.0002,weight_decay=.001,foreach=False);rng=torch.Generator().manual_seed(60046);steps=0;losses=[]
for epoch in range(2):
 order=torch.randperm(len(ids),generator=rng).tolist()
 for pos in range(0,len(ids),8):
  batch=[ids[i]for i in order[pos:pos+8]];opt.zero_grad(set_to_none=True);value=0.
  for sid in batch:
   row=cache[sid];name=str(row['severity']);loss=bridge(hs[name],ad,torch.tensor(frames(sid),device='cuda'),row['E'].to('cuda'),row['D'].to('cuda'),labels[sid],scale=1/len(batch));value+=loss/len(batch)
  assert math.isfinite(value);gn=float(torch.nn.utils.clip_grad_norm_(ad.parameters(),1.,error_if_nonfinite=True));opt.step();steps+=1;losses.append(value)
  if steps%64==0:print(json.dumps(dict(phase='interface_projection_human_TDT_train',steps=steps,total=2520,loss=value,gradient_norm=gn,seconds=time.monotonic()-start)),flush=True)
 assert steps==(epoch+1)*1260
 for param in ad.parameters():assert int(opt.state[param]['step'])==steps
 file='PROJ_TDT_ADAPTER_STEP%d_V1.pt'%steps
 with(R/file).open('xb')as f:torch.save(dict(adapter={k:v.detach().cpu()for k,v in ad.state_dict().items()},optimizer=opt.state_dict(),steps=steps,epoch=epoch+1,source_route_steps=1260,prior_small_ASR_steps=768,ASR_ID_SHA=digest(ids),source_native_weights_sha=engine.weight_sha,source_heads_sha=source_heads_sha,source_heads_checkpoint_sha=p['asr_checkpoint_sha'],rank=8,alpha=16),f)
h=hashlib.sha256()
for k,v in sorted(engine.net.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
assert steps==2520 and h.hexdigest()==engine.weight_sha and heads_sha()==source_heads_sha and all(not v.requires_grad and v.grad is None for v in engine.net.parameters())and all(v.grad is None for v in hs.parameters())
put('PROJ_TDT_ADAPTERS_SEALED_V1.json',dict(additional_global_optimizer_steps=2520,source_route_steps=1260,prior_small_ASR_steps=768,source_output_heads_sha=source_heads_sha,source_checkpoint_sha=p['asr_checkpoint_sha'],ASR_ID_SHA=digest(ids),checkpoints={tag:dict(file='PROJ_TDT_ADAPTER_STEP%d_V1.pt'%step,sha=sha(R/('PROJ_TDT_ADAPTER_STEP%d_V1.pt'%step)),steps=step)for tag,step in [('e1',1260),('e2',2520)]},frozen_base_tensor_sha_after_training=h.hexdigest(),loss_first64=float(np.mean(losses[:64])),loss_last64=float(np.mean(losses[-64:]))))
baseE=engine.net['encoder_projector'];baseD=engine.net['decoder'].decoder_projector
engine.net['encoder_projector']=AcousticWrapper(baseE,ad);engine.net['decoder'].decoder_projector=ContextWrapper(baseD,ad)
with np.load(p['micro_roles'],allow_pickle=False)as z:cf=z['FIT'].tolist();ch=z['held'].tolist()
ss=cf+ch;assert len(cf)==4096 and len(ch)==1024
@torch.no_grad()
def independent_decode(array,head,cap):
 X=torch.tensor(array,device='cuda');E=ad.acoustic(X,F.linear(X,baseE.weight,baseE.bias));history=[8192];frame=0;tokens=[];durations=[];positions=[];values=[]
 for step in range(cap):
  if frame>=len(array):break
  emb=F.embedding(torch.tensor([history],device='cuda'),engine.net['decoder'].embedding.weight);hidden,_=engine.net['decoder'].lstm(emb);D=F.linear(hidden[:,-1],baseD.weight,baseD.bias);D=ad.context(D);joint=F.relu(E[frame:frame+1]+D);z=(F.linear(joint,head.W,head.b)+2*F.linear(F.linear(joint,head.A),head.B))[0];token=int(z[:8193].argmax());dur=int(z[8193:].argmax());dur=1 if token==8192 and dur==0 else dur;tokens.append(token);durations.append(dur);positions.append(frame);values.append(z.cpu().numpy());frame+=dur
  if token==3:break
  if token!=8192:history.append(token)
 return dict(raw_tokens=tokens,durations=durations,frames=positions),values
proof=[]
for tag,step in [('e1',1260),('e2',2520)]:
 cp=torch.load(R/('PROJ_TDT_ADAPTER_STEP%d_V1.pt'%step),map_location='cpu',weights_only=False);ad.load_state_dict(cp['adapter']);ad.requires_grad_(False);records={}
 for sev in [0,1,2]:
  name=str(sev)
  assert not hs[name].A.requires_grad and not hs[name].B.requires_grad
  engine.net['joint'].head=hs[name];group=[s for s in ss if public[s]['severity']==sev];qsignal=next(s for s in cf if public[s]['severity']==sev);array=frames(qsignal)
  with torch.no_grad():
   native,traces=engine.decode_many([array],trace_first=1);ref,raw=independent_decode(array,hs[name],engine.cfg.max_symbols_per_step*len(array)-1);assert all(native[0][k]==ref[k]for k in ['raw_tokens','durations','frames']);gap=max(float(np.max(abs(a-b)))for a,b in zip(traces[0],raw));assert gap<2e-4;proof.append(dict(checkpoint=tag,severity=sev,signal=qsignal,logit_maxgap=gap,paths_exact=True))
   for pos in range(0,len(group),32):
    rows=group[pos:pos+32];values,_=engine.decode_many([frames(s)for s in rows]);records.update(zip(rows,values))
  print(json.dumps(dict(phase='routed_micro_decode',checkpoint=tag,severity=sev,rows=len(group),seconds=time.monotonic()-start)),flush=True)
 assert set(records)==set(ss);put('PROJ_TDT_'+tag.upper()+'_MICRO_RECORDS_V1.json',records)
put('PROJ_TDT_GREEDY_QUALIFICATION_V1.json',dict(tests=proof));put('PROJ_TDT_ASR_TRAIN_TERMINAL_V1.json',dict(actual_exit_code=0,additional_global_ASR_steps=2520,seconds=time.monotonic()-start))
