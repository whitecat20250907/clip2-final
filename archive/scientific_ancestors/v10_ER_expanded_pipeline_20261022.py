"""After native32 technical pass, both servers extract fixed TRAIN/VALID profiles."""
from pathlib import Path
import os,sys,json,time,shlex,tarfile,io,traceback,concurrent.futures
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';sys.path.insert(0,str(Path(__file__).parent));import v10_new_information_20261022 as G
from v10_human_channel_pipeline_20261022 import controller
from v10_ER_native_bootstrap_20261022 import BASE
T=G.T;ROOT=T.V/'evidence_relations_20261022';D=ROOT/'expanded';Q=ROOT/'native32';D.mkdir(exist_ok=True);PARTS=6
REMOTE={'3090':G.P.L.R['3090']+'/runs/v10_ER_Parakeet_expanded_20261022','4090':'/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_ER_Granite_expanded_20261022'}
def prepare():
    if (D/'protocol_lock.json').exists():return
    import numpy as np
    assert (Q/'qualification_complete.json').exists();assert T.read(Q/'3090_Parakeet_identical_public_weights.json')['exact_weights_and_tokenizer'];tr=G.P.F.arr(G.H/'G01_train.npz');va=G.P.F.arr(G.H/'G01_valid.npz');meta={r['signal_id']:r for r in T.read(T.V/'new_angles_20261021/music_witness_stage1/valid_metadata_features_only.json')};rows=[]
    for split,z in [('train',tr),('valid',va)]:
        sev=z['x'][:,0,50:53].argmax(1)
        for i,s in enumerate(z['signal_ids']):rows.append(dict(signal_id=str(s),prompt=str(z['prompts'][i]) if split=='train' else meta[str(s)]['prompt'],severity=int(sev[i]),split=split))
    assert len(rows)==5189 and len({r['signal_id'] for r in rows})==5189;T.save(D/'selection.json',rows)
    peaks={m:max(T.read(f)['peak_allocated_MiB'] for f in (Q/'results').glob(m+'_part_*/complete.json')) for m in ['Granite','Parakeet']};free={m:int(max(2048,1.25*p+1024)) for m,p in peaks.items()}
    T.save(D/'protocol_lock.json',dict(UTC=T.utc(),TRAIN=3072,VALID=2117,population='Fixed shared normalized MF/MOSS cached population, same IDs/labels as ER01; no holdout score selection',streams=4,models=['Granite','Parakeet'],native32_pass=T.read(Q/'qualification_complete.json'),time='All profile relations use common physical128bin grid over max stream duration; native frame-center approximate offset disclosed; front-padding .32s diagnostic retained, exact word-boundary ground truth not claimed',heads='Original winner587135 and all MF/MOSS heads unchanged. Add linear group-regularized bounded sigmoid correction, shared severity interactions, four fixed seeds; same conditional TRAIN parent as disclosedER01',evidence='ER02: unchanged38MF/MOSS relationships +36crossmodel full-profile relations; ER03: unchanged38 +28native midpoint-final vocabulary/time changes',controls='Within-phrase Parakeet wordprofile rotation for ER02; midpoint wordprofile/support rotation for ER03; unchanged MF/MOSS field relationships',official='Four fixed VALID candidates; official alone decides adopt/revert; no TRAIN performance gate, no EVAL',native_peak_MiB=peaks,admission_free_MiB=free,parts=PARTS,public_sources_readonly=True))
def deploy(host):
    if (D/(host+'_registered.json')).exists():return
    cfg=T.read(D/'protocol_lock.json');model='Parakeet' if host=='3090' else 'Granite';remote=REMOTE[host];uid=1004 if host=='3090' else 1030;roles=[f'{model}_part_{i}' for i in range(PARTS)];p=dict(owner_uid=uid,assets=remote if host=='3090' else BASE,parts=PARTS,data_root='/home/douhan/dataset/datasets/lyricsChallenge/cadenza_data/clip2' if host=='3090' else '/home/lisiyi/ICASSP2_2026/data/clip2',model_paths=dict(Parakeet=T.read(Q/'3090_Parakeet_identical_public_weights.json')['files']['3090']['path']) if host=='3090' else {},qualification_only=False,time_shift_control=True)
    tasks=[dict(id=r,worker='worker.py',python=G.P.L.PY[host],args=['--base',remote,'--role',r],free_MiB=cfg['admission_free_MiB'][model],allow_owned_share=True,env=dict(MKL_THREADING_LAYER='SEQUENTIAL',TMPDIR=remote+'/private_cache/tmp',HF_HOME=remote+'/private_cache/hf',CUDA_CACHE_PATH=remote+'/private_cache/cuda')) for r in roles]
    files={'protocol.json':json.dumps(p).encode(),'fleet.json':json.dumps(dict(root=G.P.L.R[host],owner_uid=uid,tasks=tasks)).encode(),'controller.py':controller(),'worker.py':(G.J/'v10_ER_native32_worker_20261022.py').read_bytes(),'v10_ER_temporal_math_20261022.py':(G.J/'v10_ER_temporal_math_20261022.py').read_bytes(),'selection.json':(D/'selection.json').read_bytes()};reg=G.A.deploy_bundle(host,remote,files);T.save(D/(host+'_registered.json'),dict(reg,roles=roles,UTC=T.utc()))
def snapshot(host):
    remote=REMOTE[host];code='from pathlib import Path\nimport json\nb=Path('+repr(remote)+');s=json.loads((b/"fleet_state.json").read_bytes()) if (b/"fleet_state.json").exists() else dict(active={},done={},pending=["startup"]);s["progress"]={r:json.loads((b/"tasks"/r/"progress.json").read_bytes()) for r in s["active"] if (b/"tasks"/r/"progress.json").exists()};s["failures"]={r:(b/r/"worker.log").read_text(errors="replace")[-2000:] for r,q in s["done"].items() if q["exit_code"]};print(json.dumps(s))';s=G.N.rpc(host,code);T.save(D/(host+'_status.json'),dict(s,UTC=T.utc()));return s
def collect(host,role):
    out=D/'results'/role;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    raw=G.N.run(host,'tar -C '+shlex.quote(REMOTE[host]+'/tasks/'+role)+' -czf - .',timeout=300)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
        for m in tar:
            if not m.isfile():continue
            assert '/' not in m.name.removeprefix('./') and '\\' not in m.name
            (out/Path(m.name).name).write_bytes(tar.extractfile(m).read())
def build_features():
    if (D/'features_complete.json').exists():return
    import numpy as np
    from v10_ER_expanded_features_20261022 import row_features
    rows=T.read(D/'selection.json');lookup={r['signal_id']:(i,r) for i,r in enumerate(rows)};base=ROOT/'ER01'
    for split in ['train','valid']:
        original=G.P.F.arr(base/(split+'.npz'));n=len(original['signal_ids']);features={f'{er}_{kind}':np.zeros((n,original['mask'].shape[1],dim),np.float32) for er,dim in [('ER02',36),('ER03',28)] for kind in ['true','shuffle']}
        for k,sid in enumerate(original['signal_ids']):
            i,r=lookup[str(sid)];part=i%PARTS;gd=D/'results'/f'Granite_part_{part}';pd=D/'results'/f'Parakeet_part_{part}';g=G.P.F.arr(gd/(str(sid)+'.npz'));p=G.P.F.arr(pd/(str(sid)+'.npz'));gj=T.read(gd/(str(sid)+'.json'));pj=T.read(pd/(str(sid)+'.json'));assert gj['prompt']==pj['prompt']==r['prompt'];words=int(original['mask'][k].sum())
            for kind in ['true','shuffle']:
                tx,ix=row_features(g,p,gj,pj,kind=='shuffle');assert len(tx)==words
                features['ER02_'+kind][k,:words]=tx;features['ER03_'+kind][k,:words]=ix
        for er in ['ER02','ER03']:
            out=D/er;out.mkdir(exist_ok=True);data={key:original[key] for key in ['signal_ids','mask','severity','parent']};data.update(x=np.concatenate([original['x'],features[er+'_true']],-1),shuffled=np.concatenate([original['x'],features[er+'_shuffle']],-1))
            if split=='train':data['target']=original['target']
            np.savez_compressed(out/(split+'.npz'),**data)
    T.save(D/'features_complete.json',dict(UTC=T.utc(),rows=5189,ER02_dimensions=74,ER03_dimensions=66,full_native_profiles_retained=True,VALID_targets=False));print('ER02_ER03_FULL_FEATURES_COMPLETE',flush=True)
def main():
    T.save(D/'pipeline_registered.json',dict(pid=os.getpid(),UTC=T.utc(),script=str(Path(__file__).resolve())));stop=time.monotonic()+48*3600
    while time.monotonic()<stop:
        try:
            if (D/'features_complete.json').exists() and (D/'readouts_complete.json').exists():
                # All GPU results are already local and verified. SSH outages
                # must not block receiving or submitting official VALID jobs.
                from v10_ER_expanded_readout_20261022 import main as readout
                readout();return
            if not (Q/'qualification_complete.json').exists():time.sleep(20);continue
            prepare()
            for h in REMOTE:deploy(h)
            with concurrent.futures.ThreadPoolExecutor(2) as pool:states=dict(zip(REMOTE,pool.map(snapshot,REMOTE)))
            for h,s in states.items():
                if s['failures']:raise RuntimeError(dict(host=h,failures=s['failures']))
                for role in s['done']:collect(h,role)
            counts={h:dict(active=len(s['active']),done=len(s['done']),pending=len(s['pending'])) for h,s in states.items()};G.F.progress('ER02','32条技术检查通过；固定3072TRAIN＋2117VALID完整分布已拆12分片接两机：'+str(counts),'按32条速度首轮提取及小头约1–4小时预算，安全卡时隙/IO影响另计；官测等额度');G.F.progress('ER03','复用ER02的Granite原生中间与最终输出，随12分片扩至首轮5189条；不另跑模型','跟随ER02特征，随后约10–30分钟小头与固定官测候选')
            if not all(len(s['done'])==PARTS for s in states.values()):time.sleep(25);continue
            build_features()
            from v10_ER_expanded_readout_20261022 import main as readout
            readout();return
        except Exception as exc:T.save(D/'pipeline_needs_attention.json',dict(UTC=T.utc(),error=repr(exc),trace=traceback.format_exc()[-4500:]));print(traceback.format_exc(),flush=True);time.sleep(30)
if __name__=='__main__':main()
