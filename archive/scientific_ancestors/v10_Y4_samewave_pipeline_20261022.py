from pathlib import Path
import os,sys,json,time,shlex,tarfile,io,hashlib,traceback,concurrent.futures
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';sys.path.insert(0,str(Path(__file__).parent))
import v10_new_information_20261022 as G
from v10_human_channel_pipeline_20261022 import controller
from v10_Y_tiny_calibration_20261022 import fit
import numpy as np
T=G.T;Y=T.V/'final15h_20261022/Y4';D=Y/'samewave_repair';D.mkdir(exist_ok=True);INV=T.read(Y/'corrected_vocal_cache_inventory.json')['hosts'];REMOTE={'3090':G.P.L.R['3090']+'/runs/v10_Y4_samewave_20261022','4090':'/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_Y4_samewave_20261022'};OLDMODEL={'3090':G.P.L.R['3090']+'/runs/v10_Y4_RMVPE_20261022','4090':'/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_Y4_RMVPE_20261022'};PARTS=8

def prepare():
    if (D/'protocol_lock.json').exists():return
    source=T.read(Y/'selection.json');seen=set();qual=[]
    for r in source:
        if r['audio_SHA'] not in seen:qual.append(r);seen.add(r['audio_SHA'])
        if len(qual)==64:break
    keys={r['audio_SHA'] for r in source};available={h:set(a['vocal'])&set(a['pitch']) for h,a in INV.items()};assert keys<=available['3090']|available['4090'];owner={k:'3090' if k in available['3090'] else '4090' for k in keys};T.save(D/'qualification64.json',qual);T.save(D/'canonical_owners.json',owner)
    # Preserve the exact original64 identities; move only their missing cached
    # stems and matching independent tracker outputs, never recompute Demucs.
    missing=sorted({r['audio_SHA'] for r in qual}-available['3090']);assert all(k in available['4090'] for k in missing)
    names=[kind+'/'+key+'.npz' for key in missing for kind in ['vocals','pitch']]
    raw=G.N.run('4090','tar -C '+shlex.quote(INV['4090']['base'])+' -czf - '+ ' '.join(shlex.quote(n) for n in names),timeout=180) if names else b'';copied={}
    if raw:
        (D/'qualification_missing_caches.tgz').write_bytes(raw)
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
            for m in tar:
                assert m.isfile() and m.name in names;copied['copied/'+m.name]=tar.extractfile(m).read()
        local=D/'copied'
        for name,value in copied.items():path=D/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    T.save(D/'protocol_lock.json',dict(UTC=T.utc(),qualification64_signal_ids=[r['signal_id'] for r in qual],same_original64=True,source_bug='Previous probe mixed canonical vocal stems with raw Audio2 fallbacks because4090cache root was wrong. Correct root is private tmp/D1_full_music_evidence_20261016.',interpolation_bug='Previous f0 interpolation replaced unvoiced NaN with zero and created artificial frequencies at voicing boundaries. New comparator selects real nearest20ms pYIN sample with voicing preserved.',same_waveform='Every cached pYIN sourceSHA must equal actual vocal NPZ SHA, per unique clip; no mixture fallback',thresholds_unchanged=dict(min_RMVPE_voiced_coverage=.7,max_octave_disagreement=.2),features='RMVPE median,p90,550/650Hz fraction on independently agreeing voiced frames +voicing/independent agreement fraction, per word;18coeff shared/severity',qualification_only_technical=True,TRAIN_score_submission_gate=False,new_official_slots=0,canonical_ownership=owner,copied_qualification_keys=missing))

def files(host,phase):
    root=REMOTE[host]+('/qual' if phase=='qual' else '/full');old=INV[host]['base'];model=OLDMODEL[host];uid=1004 if host=='3090' else 1030;runtime=(G.P.L.R['3090']+'/runs/v9_D1_full_music_evidence_20261016/runtime' if host=='3090' else '/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/D1_full_music_evidence_20261016/runtime')+':'+G.P.L.R[host]+'/runtime_deps'
    p=dict(owner_uid=uid,canonical_roots=[old,root+'/copied'],runtime=runtime,model=model+'/assets/rmvpe.pt',parts=PARTS);roles=['qual_0'] if phase=='qual' else ['full_'+str(i) for i in range(PARTS)];tasks=[dict(id=r,worker='worker.py',python=G.P.L.PY[host],args=['--base',root,'--role',r],free_MiB=3072,allow_owned_share=True,env=dict(MKL_THREADING_LAYER='SEQUENTIAL',TMPDIR=root+'/private_cache/tmp',CUDA_CACHE_PATH=root+'/private_cache/cuda')) for r in roles];selection=T.read(D/'qualification64.json') if phase=='qual' else T.read(D/(host+'_selection.json'));vendor=G.N.run(host,'cat '+shlex.quote(model+'/rmvpe_vendor.py'));f={'protocol.json':json.dumps(p).encode(),'fleet.json':json.dumps(dict(root=G.P.L.R[host],owner_uid=uid,tasks=tasks)).encode(),'controller.py':controller(),'worker.py':(G.J/'v10_Y4_samewave_worker_20261022.py').read_bytes(),'rmvpe_vendor.py':vendor,'selection.json':json.dumps(selection).encode()}
    if phase=='qual':
        for path in (D/'copied').glob('*/*.npz'):f['copied/'+str(path.relative_to(D/'copied')).replace('\\','/')]=path.read_bytes()
    return root,roles,f

def deploy(host,phase):
    path=D/(host+'_'+phase+'_registered.json')
    if path.exists():return
    remote,roles,bundle=files(host,phase);parent=REMOTE[host];scope=G.P.L.R[host] if host=='3090' else '/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014'
    G.N.rpc(host,'from pathlib import Path\nimport os,json\nb=Path('+repr(parent)+');scope=Path('+repr(scope)+').resolve();resolved=b.parent.resolve();assert b.parent.stat().st_uid==os.getuid() and (resolved==scope or scope in resolved.parents);b.mkdir(mode=0o700,exist_ok=True);assert b.stat().st_uid==os.getuid();print(json.dumps(dict(own_parent_ready=True)))')
    reg=G.A.deploy_bundle(host,remote,bundle);T.save(path,dict(reg,roles=roles,UTC=T.utc()))

def collect(host,phase):
    reg=T.read(D/(host+'_'+phase+'_registered.json'));remote=reg['remote'];s=G.N.rpc(host,'from pathlib import Path\nimport json\nb=Path('+repr(remote)+');s=json.loads((b/"fleet_state.json").read_bytes()) if (b/"fleet_state.json").exists() else dict(active={},done={},pending=["startup"]);s["progress"]={r:json.loads((b/"tasks"/r/"progress.json").read_bytes()) for r in s["active"] if (b/"tasks"/r/"progress.json").exists()};s["errors"]={r:(b/r/"worker.log").read_text(errors="replace")[-2200:] for r,v in s["done"].items() if v["exit_code"]};print(json.dumps(s))');T.save(D/(host+'_'+phase+'_status.json'),dict(s,UTC=T.utc()));assert not s['errors'],s['errors']
    for role in s['done']:
        out=D/'results'/host/phase/role;out.mkdir(parents=True,exist_ok=True)
        if not (out/'complete.json').exists():
            for name in ['evidence.jsonl','complete.json']:(out/name).write_bytes(G.N.run(host,'cat '+shlex.quote(remote+'/tasks/'+role+'/'+name),timeout=120))
    return len(s['done'])==len(reg['roles'])

def qualify():
    out=D/'qualification_complete.json'
    if out.exists():return T.read(out)['passed']
    q=T.read(D/'results/3090/qual/qual_0/complete.json');c=q['comparisons'];both=sum(r['both_voiced_frames'] for r in c);voice=sum(r['RMVPE_voiced_frames'] for r in c);wrong=sum(r['octave_disagreement_frames'] for r in c);coverage=both/max(voice,1);rate=wrong/max(both,1);passed=coverage>=.7 and rate<=.2;report=dict(UTC=T.utc(),rows=q['rows'],coverage=coverage,octave_disagreement_fraction=rate,passed=passed,all_source_SHA_verified=all(r['same_waveform_SHA_verified'] for r in c),same_original64=True,thresholds_unchanged=True);T.save(out,report);print('Y4_SAMEWAVE_QUALIFICATION',report,flush=True);return passed

def full_selection():
    if (D/'3090_selection.json').exists() and (D/'4090_selection.json').exists():return
    source=T.read(Y.parent/'Y2/selection.json');native=T.V/'evidence_relations_20261022/expanded';owner=T.read(D/'canonical_owners.json');rows={h:[] for h in REMOTE}
    for i,r in enumerate(source):
        z=G.P.F.arr(native/'results'/('Granite_part_'+str(i%6))/(r['signal_id']+'.npz'));profile=(z['A2L_final']+z['A2R_final'])/2;intervals=[]
        for mass in profile:
            c=np.cumsum(mass);assert c[-1]>0;c/=c[-1];lo=int(np.searchsorted(c,.1));hi=int(np.searchsorted(c,.9));intervals.append([lo*.08,max((lo+1)*.08,(hi+1)*.08)])
        original=nextrow[r['signal_id']];rows[owner[original['audio_SHA']]].append(dict(original,intervals=intervals))
    for h,value in rows.items():T.save(D/(h+'_selection.json'),value)
    T.save(D/'alignment_fix.json',dict(UTC=T.utc(),source='Existing Granite Audio2 left/right support, corrected80ms grid,10/90%mass windows. Reference pitch now uses reference word times; no A1/A2 delay assumption.',TRAIN=3072,VALID=2117,rows={h:len(v) for h,v in rows.items()}))

def readout():
    records={}
    for path in (D/'results').glob('*/full/full_*/evidence.jsonl'):
        for line in path.read_text().splitlines():r=json.loads(line);assert r['signal_id'] not in records;records[r['signal_id']]=r
    assert len(records)==5189;features=[];er=T.V/'evidence_relations_20261022/ER01'
    for split in ['train','valid']:
        z=G.P.F.arr(er/(split+'.npz'));x=[]
        for i,sid in enumerate(z['signal_ids']):
            q=np.asarray(records[str(sid)]['x']).mean(0);sev=int(z['severity'][i]);x.append(np.r_[q,q*(sev==1),q*(sev==2)])
        features.append(np.asarray(x,np.float32))
    fit('Y4',*features,[s+'_'+k for s in ['shared','mild','moderate'] for k in ['median_F0','p90_F0','above550','above650','RMVPE_voiced_fraction','independent_agreement_fraction']],dict(corrections=T.read(D/'protocol_lock.json'),qualification=T.read(D/'qualification_complete.json'),reference_alignment=T.read(D/'alignment_fix.json'),controls='Wrong-time word summaries saved, not submitted; no extra official quota'))
    T.save(D/'evidence_complete.json',dict(UTC=T.utc(),rows=5189,original_heads_frozen=True,canonical_unique_audio=1910))

nextrow={r['signal_id']:r for r in T.read(Y/'selection.json')}
def main():
    stop=time.monotonic()+7*3600;T.save(D/'pipeline_registered.json',dict(UTC=T.utc(),pid=os.getpid()))
    while time.monotonic()<stop:
        try:
            prepare();deploy('3090','qual')
            if not collect('3090','qual'):time.sleep(20);continue
            if not qualify():G.F.progress('Y4','已修复两机缓存路径、禁止混合音回退和未发声F0插值；同64条固定门槛仍未过，保留资格材料','不扩全量；官测仍由其他6份本人候选优先');return
            full_selection()
            for h in REMOTE:deploy(h,'full')
            with concurrent.futures.ThreadPoolExecutor(2) as ex:done=list(ex.map(lambda h:collect(h,'full'),REMOTE))
            G.F.progress('Y4','同64条修复后通过固定资格；按缓存真实归属拆16分片，使用Audio2词窗与独立音高一致性，不混入原混合音','全量按首片速度修订，预算20–60分钟；18参数本人候选随后约10分钟；不增加本人官测总预算')
            if all(done):readout();return
        except Exception as exc:T.save(D/'needs_attention.json',dict(UTC=T.utc(),error=repr(exc),trace=traceback.format_exc()[-4500:]));print(traceback.format_exc(),flush=True)
        time.sleep(30)
if __name__=='__main__':main()
