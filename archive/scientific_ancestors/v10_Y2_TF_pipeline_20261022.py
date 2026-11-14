from pathlib import Path
import sys,os,time,json,shlex,concurrent.futures,traceback
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';sys.path.insert(0,str(Path(__file__).parent))
import v10_new_information_20261022 as G
from v10_human_channel_pipeline_20261022 import controller
from v10_Y_tiny_calibration_20261022 import fit
import numpy as np
T=G.T;D=T.V/'final15h_20261022/Y2-TF';D.mkdir(exist_ok=True);PARTS=16
REMOTE={'3090':G.P.L.R['3090']+'/runs/v10_Y2_blindEC_TF_20261022','4090':'/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_Y2_blindEC_TF_20261022'}
def deploy(host):
    if (D/(host+'_registered.json')).exists():return
    remote=REMOTE[host];root=G.P.L.R[host];uid=1004 if host=='3090' else 1030;roles=['tf_'+str(i) for i in range(0 if host=='3090' else 1,PARTS,2)];p=dict(owner_uid=uid,root=root,parts=PARTS,data_root='/home/douhan/dataset/datasets/lyricsChallenge/cadenza_data/clip2' if host=='3090' else '/home/lisiyi/ICASSP2_2026/data/clip2',model=root+'/models/whisper-medium' if host=='3090' else '/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/CE_VALID_NVMe_20261018/model');tasks=[dict(id=r,worker='worker.py',python=G.P.L.PY[host],args=['--base',remote,'--role',r],free_MiB=7168,allow_owned_share=True,env=dict(MKL_THREADING_LAYER='SEQUENTIAL',PYTHONPATH=root+'/runtime_deps:'+root+'/03_my_improvements',TMPDIR=remote+'/private_cache/tmp',CUDA_CACHE_PATH=remote+'/private_cache/cuda')) for r in roles];selection=[{k:r[k] for k in ['signal_id','prompt','severity','split']} for r in T.read(D.parent/'Y2/selection.json')];files={'protocol.json':json.dumps(p).encode(),'fleet.json':json.dumps(dict(root=root,owner_uid=uid,tasks=tasks)).encode(),'selection.json':json.dumps(selection).encode(),'worker.py':(G.J/'v10_Y2_TF_worker_20261022.py').read_bytes(),'controller.py':controller()};reg=G.A.deploy_bundle(host,remote,files);T.save(D/(host+'_registered.json'),dict(reg,roles=roles,UTC=T.utc()))
def collect(host):
    remote=REMOTE[host];s=G.N.rpc(host,'from pathlib import Path\nimport json\nb=Path('+repr(remote)+');s=json.loads((b/"fleet_state.json").read_bytes()) if (b/"fleet_state.json").exists() else dict(active={},done={},pending=["startup"]);s["progress"]={r:json.loads((b/"tasks"/r/"progress.json").read_bytes()) for r in s["active"] if (b/"tasks"/r/"progress.json").exists()};s["errors"]={r:(b/r/"worker.log").read_text(errors="replace")[-2200:] for r,q in s["done"].items() if q["exit_code"]};print(json.dumps(s))');T.save(D/(host+'_status.json'),dict(s,UTC=T.utc()));assert not s['errors'],s['errors']
    for role in s['done']:
        out=D/'results'/role;out.mkdir(parents=True,exist_ok=True)
        if not (out/'complete.json').exists():
            for n in ['evidence.jsonl','complete.json']:(out/n).write_bytes(G.N.run(host,'cat '+shlex.quote(remote+'/tasks/'+role+'/'+n),timeout=120))
    return len(s['done'])==8
def readout():
    records={}
    for path in (D/'results').glob('tf_*/evidence.jsonl'):
        for line in path.read_text().splitlines():r=json.loads(line);assert r['signal_id'] not in records;records[r['signal_id']]=r
    er=T.V/'evidence_relations_20261022/ER01';features=[]
    for split in ['train','valid']:
        z=G.P.F.arr(er/(split+'.npz'));x=[]
        for i,sid in enumerate(z['signal_ids']):
            q=np.asarray(records[str(sid)]['x']).mean(0);sev=int(z['severity'][i]);x.append(np.r_[q,q*(sev==1),q*(sev==2)])
        features.append(np.asarray(x,np.float32))
    # Same fitter and limits, separately named route; original Y2 candidate is
    # retained verbatim, never silently replaced after a manifest was frozen.
    fit('Y2-TF',*features,[f'{s}_{f}' for s in ['shared','mild','moderate'] for f in ['EC_L_logp_delta','EC_R_logp_delta','ear_delta_disagreement','EC_active_fraction']],dict(frozen_listener=True,new_frontend='Blind A1-only slow EC half-mixture, duplicate-ear exact fallback',not_same_as_historical_reference_driven_EC=True))
def main():
    stop=time.monotonic()+15*3600;T.save(D/'pipeline_registered.json',dict(UTC=T.utc(),pid=os.getpid()))
    while time.monotonic()<stop:
        try:
            for h in REMOTE:deploy(h)
            with concurrent.futures.ThreadPoolExecutor(2) as ex:done=list(ex.map(collect,REMOTE))
            counts={h:{k:len(T.read(D/(h+'_status.json'))[k]) for k in ['active','pending','done']} for h in REMOTE};G.F.progress('Y2-TF','盲EC波形→冻结Whisper逐词支持变化，16分片已接两机；'+str(counts),'首批实测峰值约1.53GiB、每卡约1.5–3条/秒；当前整轮约10–20分钟＋小校正核验；本人额度总额仍≤6')
            if all(done):readout();return
        except Exception as e:T.save(D/'needs_attention.json',dict(UTC=T.utc(),error=repr(e),trace=traceback.format_exc()[-4500:]));print(traceback.format_exc(),flush=True)
        time.sleep(30)
if __name__=='__main__':main()
