from pathlib import Path
import os,sys,json,time,shlex,csv,traceback,concurrent.futures
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';sys.path.insert(0,str(Path(__file__).parent));import v10_new_information_20261022 as G
from v10_human_channel_pipeline_20261022 import controller
import numpy as np
T=G.T;D=T.V/'evidence_relations_20261022/expanded';SEEDS=[142,242,342,442];ERS=['ER02','ER03'];ARMS=['true','shuffle'];REMOTE={'3090':G.P.L.R['3090']+'/runs/v10_ER_expanded_readouts_20261022','4090':'/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_ER_expanded_readouts_20261022'}
def worker():
    chosen='[38,39,40,45,46,47,52,53,54,59]'
    text=(G.J/'v10_ER01_relation_worker_20261022.py').read_text();text=text.replace('selected=[0,1,2,3,8,9,10,11,32,33]','selected='+chosen);text=text.replace('penalty[39:]=.05','penalty[1+t[key].shape[-1]:]=.05');text=text.replace('selected_interactions=[0,1,2,3,8,9,10,11,32,33]','selected_interactions='+chosen);compile(text,'expanded_relation_worker','exec');return text.encode()
def roles(host):return [(er,arm,s) for er in ERS for arm in ARMS for s in (SEEDS[::2] if host=='3090' else SEEDS[1::2])]
def deploy(host):
    path=D/(host+'_readouts_registered.json')
    if path.exists():return
    remote=REMOTE[host];uid=1004 if host=='3090' else 1030;files={'controller.py':controller(),'worker.py':worker()};tasks=[]
    for er in ERS:
        p=dict(owner_uid=uid,roles=[arm+'_s'+str(s) for arm in ARMS for s in SEEDS],policy=T.read(D/'protocol_lock.json'));files[er+'/protocol.json']=json.dumps(p).encode()
        for split in ['train','valid']:files[er+'/'+split+'.npz']=(D/er/(split+'.npz')).read_bytes()
    for er,arm,s in roles(host):
        role=arm+'_s'+str(s);tasks.append(dict(id=er+'_'+role,worker='worker.py',python=G.P.L.PY[host],args=['--base',remote+'/'+er,'--role',role],free_MiB=2048,allow_owned_share=True,env=dict(MKL_THREADING_LAYER='SEQUENTIAL',TMPDIR=remote+'/private_cache/tmp',CUDA_CACHE_PATH=remote+'/private_cache/cuda')))
    files['fleet.json']=json.dumps(dict(root=G.P.L.R[host],owner_uid=uid,tasks=tasks)).encode();reg=G.A.deploy_bundle(host,remote,files);T.save(path,dict(reg,UTC=T.utc(),roles=roles(host)))
def collect(host):
    remote=REMOTE[host];s=G.N.rpc(host,'from pathlib import Path\nimport json\nb=Path('+repr(remote)+');s=json.loads((b/"fleet_state.json").read_bytes()) if (b/"fleet_state.json").exists() else dict(active={},done={},pending=["startup"]);s["failures"]={r:(b/r/"worker.log").read_text(errors="replace")[-2000:] for r,q in s["done"].items() if q["exit_code"]};print(json.dumps(s))');T.save(D/(host+'_readout_status.json'),dict(s,UTC=T.utc()));assert not s['failures'],s['failures']
    for er,arm,seed in roles(host):
        role=arm+'_s'+str(seed)
        if er+'_'+role not in s['done']:continue
        out=D/er/'results'/role;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        for name in ['model.pt','valid.npz','complete.json']:(out/name).write_bytes(G.N.run(host,'cat '+shlex.quote(remote+'/'+er+'/tasks/'+role+'/'+name),timeout=90))
    return len(s['done'])==len(roles(host))
def submit(er,arm):
    root=D/er/arm;root.mkdir(parents=True,exist_ok=True)
    if (root/'official_registered.json').exists():return
    v=G.P.F.arr(D/er/'valid.npz');p=np.mean([G.P.F.arr(D/er/'results'/(arm+'_s'+str(s))/'valid.npz')['prediction'] for s in SEEDS],0);assert np.isfinite(p).all() and np.max(np.abs(p-v['parent']))<=.050001;out=root/'candidate';out.mkdir(exist_ok=True)
    with (out/'valid_predictions.csv').open('w',newline='') as f:csv.writer(f).writerows(zip(v['signal_ids'],[format(float(x)*100,'.10f') for x in p]))
    report=dict(UTC=T.utc(),task=er,arm=arm,csv_SHA=T.sha(out/'valid_predictions.csv'),protocol=T.read(D/'protocol_lock.json'),qualification=T.read(D.parent/'native32/qualification_complete.json'),readout_protocol=T.read(D/'readout_protocol_lock.json'),time_axis_correction=T.read(D.parent/'native32/time_axis_correction.json'),feature_pack_replay=T.read(D/'feature_pack_time_axis_verified.json'),results={str(s):T.read(D/er/'results'/(arm+'_s'+str(s))/'complete.json') for s in SEEDS},TRAIN_performance_gate=False,VALID_targets=False,EVAL_submission=False)
    T.save(out/'report.json',report);T.save(out/'verification.json',dict(status='verified',csv_sha256=report['csv_SHA'],report_sha256=T.sha(out/'report.json')));cid='v10_'+er+'_'+arm+'_native_time_vocabulary_relation_20261022';c=G.P.H.F.candidate(out,cid,cid,er+';Frozenown587135unchanged;nativefullprofiles;'+arm+';boundedlinearcorrection,groupregularization,10prespecifiednewrelationseverityinteractions;4seeds;noTRAINscoregate.');G.P.H.F.record=G.P.H.record;spec=G.P.H.F.submit_candidates([c],root/'official');T.save(root/'official_registered.json',dict(UTC=T.utc(),spec=spec))
def finish(er,arm):
    root=D/er/arm
    if (root/'complete.json').exists():return
    result=T.read(root/'candidate/official_result.json');old=T.read(T.V/'current_best_own_valid.json');adopt=result['result']['RMSE']<old['RMSE']-1e-12
    if adopt:T.save(T.V/'current_best_own_valid.json',dict(old,UTC=T.utc(),submission_id=result['submission_id'],RMSE=result['result']['RMSE'],Correlation=result['result']['Correlation'],csv=result['csv'],csv_SHA=result['csv_SHA'],receipt=result['receipt'],previous_best=old,relation_stage=str(D),relation_task=er,relation_arm=arm,full_raw_infer_all_complete=False,frozen_infer_package_not_updated=True))
    T.save(root/'complete.json',dict(UTC=T.utc(),official=result,adopted=adopt,gain_RMSE=old['RMSE']-result['result']['RMSE']))
def main():
    from v10_ER_feature_pack_verify_20261022 import ensure_correct
    ensure_correct()
    stop=time.monotonic()+48*3600
    while time.monotonic()<stop:
        try:
            if not (D/'readouts_complete.json').exists():
                for h in REMOTE:deploy(h)
                with concurrent.futures.ThreadPoolExecutor(2) as ex:complete=list(ex.map(collect,REMOTE))
                statuses=[T.read(D/(h+'_readout_status.json')) for h in REMOTE];active=sum(len(s['active']) for s in statuses);done=sum(len(s['done']) for s in statuses)
                for er in ERS:G.F.progress(er,f'5189条双模型完整分布已齐；80ms坐标特征包重放通过；真实/错配四种子修正器共16小头，当前{active}运行、{done}完成','小头与候选核验约10–30分钟；新LMY优先提交后按官方额度排队')
                if not all(complete):time.sleep(20);continue
                T.save(D/'readouts_complete.json',dict(UTC=T.utc(),heads=16,ER02_parameters=97,ER03_parameters=89,models_pretrained_frozen=True,baseline_unchanged=True,selected_new_relation_interactions=[38,39,40,45,46,47,52,53,54,59]))
            for er in ERS:
                for arm in ARMS:submit(er,arm)
                G.F.progress(er,'完整原生分布与四种子有界小评分器已完成；真关系/打乱对照候选已核验，等额度先提交新LMY联合候选','无需重跑大模型；新LMY先提交后按官方可用额度顺次提交，平台耗时另计')
            # User's latest-LMY fusion is the first new score priority when quota returns.
            lm=T.V/'LMY_latest_joint_20261022/status.json';ls=T.read(lm) if lm.exists() else {};lm_done=(T.V/'LMY_latest_joint_20261022/complete.json').exists()
            if not ls.get('submission_id') and not lm_done:time.sleep(60);continue
            waiting=False;G.P.H.F.record=G.P.H.record
            for er in ERS:
                for arm in ARMS:
                    root=D/er/arm
                    if not G.P.H.F.poll_official(root/'official',T.read(root/'official_registered.json')['spec']):waiting=True;continue
                    finish(er,arm)
                G.F.progress(er,'完整分布特征和四种子受限线性修正器完成，真实/打乱对应两版已核验并登记官测','有额度后提交，官方排队时间另计；TRAIN成绩未作放行条件')
            if waiting:time.sleep(60);continue
            for er in ERS:
                out=D/er/'complete.json';T.save(out,dict(UTC=T.utc(),status='two_fixed_official_candidates_complete',arms={a:T.read(D/er/a/'complete.json') for a in ARMS}));T.mark_complete(er,[str(out),str(D/'protocol_lock.json'),str(D/'features_complete.json')])
            T.refresh();return
        except Exception as exc:T.save(D/'readout_needs_attention.json',dict(UTC=T.utc(),error=repr(exc),trace=traceback.format_exc()[-4500:]));print(traceback.format_exc(),flush=True);time.sleep(30)
if __name__=='__main__':main()
