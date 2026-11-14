"""Normalized same-word relationships; preserve every frozen winner head."""
from pathlib import Path
import sys,os,json,time,shlex,csv,traceback,concurrent.futures
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';sys.path.insert(0,str(Path(__file__).parent))
import v10_new_information_20261022 as G
import v10_evidence_relations_registry_20261022 as R
import numpy as np
T=G.T;D=R.D/'ER01';D.mkdir(exist_ok=True);J=G.J;SEEDS=[142,242,342,442]
REMOTE={'3090':G.P.L.R['3090']+'/runs/v10_ER01_relations_20261022','4090':'/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_ER01_relations_20261022'}

def relations(mf,mo):
    fields=[];hits=[];unc=[]
    for k in range(4):
        hm=mf[:,:,55+4*k];hs=mo[:,:,55+4*k];cm=np.exp(np.clip(mf[:,:,9*k+4],-12,0));cs=np.exp(np.clip(mo[:,:,9*k+4],-12,0));usable=(mf[:,:,9*k+6]>.5)&(mo[:,:,9*k+6]>.5)&(mf[:,:,9*k+8]<.5)&(mo[:,:,9*k+8]<.5);u=(1-cm)*(1-cs)*usable;amb=(mf[:,:,55+4*k+3]+mo[:,:,55+4*k+3]>0).astype(np.float32)
        fields.extend([hm*hs,hm*(1-hs),hs*(1-hm),(1-hm)*(1-hs),u,np.abs(cm-cs)*usable,amb,usable.astype(np.float32)]);hits.append(hm*hs);unc.append(u)
    fields.extend([hits[0]-hits[2],hits[1]-hits[3],unc[0]-unc[2],unc[1]-unc[3],np.abs(mf[:,:,55]-mf[:,:,59]),np.abs(mo[:,:,55]-mo[:,:,59])]);out=np.stack(fields,-1);assert out.shape[-1]==38 and np.isfinite(out).all();return out.astype(np.float32)

def prepare():
    R.register()
    if (D/'inputs_report.json').exists():return
    own=T.read(T.V/'current_best_own_valid.json');T.save(D/'own_reference.json',own);mf=G.P.F.arr(G.D/'G01_MF/readout/train.npz');mv=G.P.F.arr(G.D/'G01_MF/readout/valid.npz');mo=G.P.F.arr(G.H/'G01_train.npz');ov=G.P.F.arr(G.H/'G01_valid.npz');lookup={str(s):i for i,s in enumerate(mf['signal_ids'])};ix=np.asarray([lookup[str(s)] for s in mo['signal_ids']]);assert np.array_equal(mf['prompts'][ix],mo['prompts']) and np.array_equal(mf['mask'][ix],mo['mask']) and np.array_equal(mf['target'][ix],mo['target']);assert np.array_equal(mv['signal_ids'],ov['signal_ids']) and np.array_equal(mv['mask'],ov['mask']);ids,pred=G.F.csvpred(own['csv']);assert np.array_equal(ids,ov['signal_ids']) and T.sha(Path(own['csv']))==own['csv_SHA'];validmeta={r['signal_id']:r for r in T.read(T.V/'new_angles_20261021/music_witness_stage1/valid_metadata_features_only.json')};rng=np.random.default_rng(142);audit={}
    for split,m,s in [('train',mf['x'][ix],mo),('valid',mv['x'],ov)]:
        raw=s['x'];shuffled=raw.copy();fixed=0
        for i in range(len(raw)):
            n=int(s['mask'][i].sum());shift=1+int(rng.integers(n-1));indices=np.roll(np.arange(n),shift);assert not (indices==np.arange(n)).any();shuffled[i,:n]=raw[i,indices];fixed+=int((indices==np.arange(n)).sum())
        x=relations(m,raw);sh=relations(m,shuffled);mask=s['mask'];x[~mask.astype(bool)]=0;sh[~mask.astype(bool)]=0;severity=s['x'][:,0,50:53].argmax(1).astype(np.int64);assert np.all(s['x'][:,0,50:53].sum(1)==1)
        data=dict(signal_ids=s['signal_ids'],x=x,shuffled=sh,mask=mask,severity=severity,parent=(.9*s['parent']+.1*s['anchor']).astype(np.float32) if split=='train' else pred.astype(np.float32))
        if split=='train':data.update(target=s['target'],prompts=s['prompts'])
        np.savez_compressed(D/(split+'.npz'),**data);audit[split]=dict(rows=len(raw),shuffled_word_fixed_points=fixed,changed_rows=int(np.any(x!=sh,axis=(1,2)).sum()))
    protocol=dict(UTC=T.utc(),task='ER01',TRAIN=3072,VALID=2117,raw_sources='Current winning G01 normalized72dim per-word MF/MOSS; native logp preserved, never duplicated for expanded contractions',relation_dimensions=38,relation_fields='4streams*(cohit,MFonly,MOSSonly,neither,jointnativeuncertainty,nativeconfidencegap,ambiguity,bothusable) +2cohitA1A2drop+2uncertaintydrop+2model-earhitgaps',readout='Shared linear61coefficients inside sigmoid around frozen parent logit;10prespecified interactions x2severity levels;L2 .01 shared,.05 severity,.001 intercept',correction_bound='abs(logit shift)<=.2, absolute score correction<=5 points',seeds=SEEDS,epochs=80,lr=.001,controls='Rotate MOSS word rows by nonzero cyclic offset within each phrase in TRAIN and VALID; MF unchanged; raw marginal sources/labels/reference IDs unchanged',zero_replay='All correction coefficients zero must replay own587135 within2e-7 probability',TRAIN_parent='.9*previousMFown conditional OOF+.1*oldMOSS OOF anchor. New normalized frozen heads are not refit. Anchor is disclosed conditional surrogate, not full nested OOF.',VALID_parent='Exact frozen own winner CSV; all existing branch scores/weights unchanged',official_candidates=['true','shuffle'],official_only_adoption=True,no_TRAIN_performance_gate=True,EVAL_submission=False,original_heads_retrained=False)
    T.save(D/'protocol_lock.json',protocol);T.save(D/'inputs_report.json',dict(UTC=T.utc(),audit=audit,policy=protocol,own_reference=own,source_SHA={str(p):T.sha(p) for p in [G.D/'G01_MF/readout/train.npz',G.D/'G01_MF/readout/valid.npz',G.H/'G01_train.npz',G.H/'G01_valid.npz']},input_SHA={p.name:T.sha(p) for p in D.glob('*.npz')},VALID_targets=False))

def deploy(host):
    file=D/(host+'_registered.json')
    if file.exists():return
    from v10_human_channel_pipeline_20261022 import controller
    remote=REMOTE[host];uid=1004 if host=='3090' else 1030;roles=[f'{arm}_s{s}' for arm in ['true','shuffle'] for s in SEEDS];roles=roles[::2] if host=='3090' else roles[1::2];p=dict(owner_uid=uid,roles=roles,policy=T.read(D/'protocol_lock.json'));fleet=dict(root=G.P.L.R[host],owner_uid=uid,tasks=[dict(id=r,worker='worker.py',python=G.P.L.PY[host],args=['--base',remote,'--role',r],free_MiB=2048,allow_owned_share=True,env=dict(MKL_THREADING_LAYER='SEQUENTIAL',TMPDIR=remote+'/private_cache/tmp',CUDA_CACHE_PATH=remote+'/private_cache/cuda')) for r in roles]);files={'protocol.json':json.dumps(p).encode(),'fleet.json':json.dumps(fleet).encode(),'controller.py':controller(),'worker.py':(J/'v10_ER01_relation_worker_20261022.py').read_bytes(),'train.npz':(D/'train.npz').read_bytes(),'valid.npz':(D/'valid.npz').read_bytes()};reg=G.A.deploy_bundle(host,remote,files);T.save(file,dict(reg,roles=roles,UTC=T.utc()))

def collect(host):
    reg=T.read(D/(host+'_registered.json'));remote=reg['remote'];s=G.N.rpc(host,'from pathlib import Path\nimport json\nb=Path('+repr(remote)+');s=json.loads((b/"fleet_state.json").read_bytes()) if (b/"fleet_state.json").exists() else dict(active={},done={},pending=["startup"]);s["progress"]={r:json.loads((b/"tasks"/r/"progress.json").read_bytes()) for r in s["active"] if (b/"tasks"/r/"progress.json").exists()};s["errors"]={r:(b/r/"worker.log").read_text(errors="replace")[-1800:] for r,q in s.get("done",{}).items() if q["exit_code"]};print(json.dumps(s))');T.save(D/(host+'_status.json'),dict(s,UTC=T.utc()));assert not s['errors'],s['errors']
    for role in s.get('done',{}):
        out=D/'results'/role;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        for name in ['valid.npz','model.pt','complete.json']:(out/name).write_bytes(G.N.run(host,'cat '+shlex.quote(remote+'/tasks/'+role+'/'+name)))
    return len(s.get('done',{}))==len(reg['roles'])

def submit(arm):
    if (D/arm/'official_registered.json').exists():return
    v=G.P.F.arr(D/'valid.npz');p=np.mean([G.P.F.arr(D/'results'/f'{arm}_s{s}/valid.npz')['prediction'] for s in SEEDS],0);assert np.isfinite(p).all() and np.max(np.abs(p-v['parent']))<=.050001;out=D/arm/'candidate';out.mkdir(parents=True,exist_ok=True)
    with (out/'valid_predictions.csv').open('w',newline='') as f:csv.writer(f).writerows(zip(v['signal_ids'],[format(float(x)*100,'.10f') for x in p]))
    report=dict(UTC=T.utc(),arm=arm,csv_SHA=T.sha(out/'valid_predictions.csv'),protocol=T.read(D/'protocol_lock.json'),inputs=T.read(D/'inputs_report.json'),results={str(s):T.read(D/'results'/f'{arm}_s{s}/complete.json') for s in SEEDS},VALID_targets=False,EVAL_submission=False);T.save(out/'report.json',report);T.save(out/'verification.json',dict(status='verified',csv_sha256=report['csv_SHA'],report_sha256=T.sha(out/'report.json')));cid='v10_ER01_'+arm+'_MF_MOSS_sameword_linear61_20261022';c=G.P.H.F.candidate(out,cid,cid,'Frozenown587135unchanged;normalizedMF/MOSSsamewordrelation38features;61coefsharedseveritylogitcorrectionbound.2;'+arm+';fourfixedseeds,noTRAINscoregate.');G.P.H.F.record=G.P.H.record;spec=G.P.H.F.submit_candidates([c],D/arm/'official');T.save(D/arm/'official_registered.json',dict(UTC=T.utc(),spec=spec))

def finish(arm):
    if (D/arm/'complete.json').exists():return
    r=T.read(D/arm/'candidate/official_result.json');old=T.read(T.V/'current_best_own_valid.json');adopt=r['result']['RMSE']<old['RMSE']-1e-12
    if adopt:T.save(T.V/'current_best_own_valid.json',dict(old,UTC=T.utc(),submission_id=r['submission_id'],RMSE=r['result']['RMSE'],Correlation=r['result']['Correlation'],csv=r['csv'],csv_SHA=r['csv_SHA'],receipt=r['receipt'],previous_best=old,relation_stage=str(D),relation_arm=arm,full_raw_infer_all_complete=False,frozen_infer_package_not_updated=True))
    T.save(D/arm/'complete.json',dict(UTC=T.utc(),official=r,adopted=adopt,gain_RMSE=old['RMSE']-r['result']['RMSE']))

def main():
    prepare();T.save(D/'pipeline_registered.json',dict(UTC=T.utc(),pid=os.getpid(),script=str(Path(__file__).resolve())));end=time.monotonic()+48*3600
    while time.monotonic()<end:
        try:
            for host in ['3090','4090']:deploy(host)
            if not (D/'training_complete.json').exists():
                with concurrent.futures.ThreadPoolExecutor(2) as ex:done=list(ex.map(collect,['3090','4090']))
                G.F.progress('ER01','61参数修正器和打乱词对应对照已接入两机轻载/空卡，四种子分别训练；保持所有原头冻结','小头通常数分钟，随后直接登记两份官测；额度恢复另计')
                if not all(done):time.sleep(15);continue
                T.save(D/'training_complete.json',dict(UTC=T.utc(),heads=8,parameters=61,original_heads_unchanged=True,TRAIN_score_gate=False))
            for arm in ['true','shuffle']:submit(arm)
            waiting=False;G.P.H.F.record=G.P.H.record
            lm=T.V/'LMY_latest_joint_20261022/status.json';ls=T.read(lm) if lm.exists() else {};lm_done=(T.V/'LMY_latest_joint_20261022/complete.json').exists()
            if not ls.get('submission_id') and not lm_done:
                G.F.progress('ER01','两机8个小头均完成；真词关系/打乱对应两候选已登记，额度恢复时先让新LMY联合候选提交','无需额外训练；等待可用额度与新LMY优先提交，随后按官方队列出分');time.sleep(60);continue
            for arm in ['true','shuffle']:
                if not G.P.H.F.poll_official(D/arm/'official',T.read(D/arm/'official_registered.json')['spec']):waiting=True;continue
                finish(arm)
            if waiting:G.F.progress('ER01','两机8个61参数修正器已完成；真词关系/打乱对应两候选已核验，正在官方队列等待额度或回执','无需额外训练；可提交后通常5–30分钟，合法额度恢复时间另计');time.sleep(60);continue
            T.save(D/'pipeline_complete.json',dict(UTC=T.utc(),status='two_fixed_official_relation_tests_complete',arms={a:T.read(D/a/'complete.json') for a in ['true','shuffle']},no_TRAIN_gate=True));T.mark_complete('ER01',[str(D/'pipeline_complete.json'),str(D/'protocol_lock.json'),str(D/'inputs_report.json')]);T.refresh();return
        except Exception as e:T.save(D/'needs_attention.json',dict(UTC=T.utc(),error=repr(e),trace=traceback.format_exc()[-3500:]));print(traceback.format_exc(),flush=True);time.sleep(25)

if __name__=='__main__':main()
