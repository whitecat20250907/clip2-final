from pathlib import Path
import os,sys,json,time,traceback,shlex,concurrent.futures
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';sys.path.insert(0,str(Path(__file__).parent))
import v10_new_information_20261022 as G
from v10_human_channel_pipeline_20261022 import controller
from v10_Y_tiny_calibration_20261022 import fit
import numpy as np
T=G.T;D=T.V/'final15h_20261022/Y2';D.mkdir(exist_ok=True);E=T.V/'evidence_relations_20261022/expanded';PARTS=16
REMOTE={'3090':G.P.L.R['3090']+'/runs/v10_Y2_blindEC_20261022','4090':'/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_Y2_blindEC_20261022'}

def prepare():
    if (D/'selection.json').exists():return
    source=T.read(E/'selection.json');rows=[]
    for i,r in enumerate(source):
        # Existing native Granite A1-only word support; the 80ms coordinate
        # correction was verified before this route was started.
        z=G.P.F.arr(E/'results'/('Granite_part_'+str(i%6))/(r['signal_id']+'.npz'));p=(z['A1L_final']+z['A1R_final'])/2;assert len(p)==len(r['prompt'].split())
        rows.append(dict(r,word_profile=np.round(p,5).tolist()))
    T.save(D/'selection.json',rows);T.save(D/'protocol_lock.json',dict(UTC=T.utc(),TRAIN=3072,VALID=2117,parts=PARTS,Audio2_read=False,word_time_support='Existing Granite A1 left/right mean,80ms corrected frame grid; not phoneme ground truth',new_evidence='Blind low/high coherence, phase spatial cue, symmetric EC potential with gain/delay uncertainty and bounded gain, ILD dispersion',head='18 coefficients:6 word-mean features plus mild/moderate interactions; no new large listener',scope='First blind-evidence candidate. Changed-waveform Whisper TF is separate pending work; do not call it complete front-end migration.',controls='Duplicate ear must yield zero spatial EC potential; L/R exchange must be invariant. EC energy cancellation is not target SNR.',TRAIN_performance_gate=False))

def deploy(host):
    path=D/(host+'_registered.json')
    if path.exists():return
    remote=REMOTE[host];uid=1004 if host=='3090' else 1030;roles=['blind_'+str(i) for i in range(0 if host=='3090' else 1,PARTS,2)];p=dict(owner_uid=uid,parts=PARTS,data_root='/home/douhan/dataset/datasets/lyricsChallenge/cadenza_data/clip2' if host=='3090' else '/home/lisiyi/ICASSP2_2026/data/clip2');tasks=[dict(id=r,worker='worker.py',python=G.P.L.PY[host],args=['--base',remote,'--role',r],free_MiB=2048,allow_owned_share=True,env=dict(MKL_THREADING_LAYER='SEQUENTIAL',OMP_NUM_THREADS='2',TMPDIR=remote+'/private_cache/tmp',CUDA_CACHE_PATH=remote+'/private_cache/cuda')) for r in roles]
    files={'protocol.json':json.dumps(p).encode(),'fleet.json':json.dumps(dict(root=G.P.L.R[host],owner_uid=uid,tasks=tasks)).encode(),'controller.py':controller(),'worker.py':(G.J/'v10_Y2_blind_worker_20261022.py').read_bytes(),'v10_Y2_blind_math_20261022.py':(G.J/'v10_Y2_blind_math_20261022.py').read_bytes(),'selection.json':(D/'selection.json').read_bytes()};reg=G.A.deploy_bundle(host,remote,files);T.save(path,dict(reg,roles=roles,UTC=T.utc()))

def collect(host):
    reg=T.read(D/(host+'_registered.json'));remote=reg['remote'];s=G.N.rpc(host,'from pathlib import Path\nimport json\nb=Path('+repr(remote)+');s=json.loads((b/"fleet_state.json").read_bytes()) if (b/"fleet_state.json").exists() else dict(active={},done={},pending=["startup"]);s["progress"]={r:json.loads((b/"tasks"/r/"progress.json").read_bytes()) for r in s["active"] if (b/"tasks"/r/"progress.json").exists()};s["errors"]={r:(b/"tasks"/r/"worker.log").read_text(errors="replace")[-1800:] if (b/"tasks"/r/"worker.log").exists() else str(q) for r,q in s["done"].items() if q["exit_code"]};print(json.dumps(s))');T.save(D/(host+'_status.json'),dict(s,UTC=T.utc()));assert not s['errors'],s['errors']
    for role in s['done']:
        out=D/'results'/role;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        for name in ['evidence.jsonl','qualification.json','complete.json']:(out/name).write_bytes(G.N.run(host,'cat '+shlex.quote(remote+'/tasks/'+role+'/'+name),timeout=120))
    return len(s['done'])==len(reg['roles'])

def readout():
    records={}
    for path in (D/'results').glob('blind_*/evidence.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line);assert r['signal_id'] not in records;records[r['signal_id']]=r
    er=T.V/'evidence_relations_20261022/ER01';features=[]
    for split in ['train','valid']:
        z=G.P.F.arr(er/(split+'.npz'));x=[]
        for i,sid in enumerate(z['signal_ids']):
            r=records[str(sid)];q=np.asarray(r['x']).mean(0);sev=int(z['severity'][i]);x.append(np.r_[q,q*(sev==1),q*(sev==2)])
        features.append(np.asarray(x,np.float32))
    fit('Y2',*features,[f'{s}_{f}' for s in ['shared','mild','moderate'] for f in ['coherence_low','coherence_high','phase_spatial','blind_EC_potential','ILD_spread','ILD_abs']],T.read(D/'protocol_lock.json'))

def main():
    stop=time.monotonic()+15*3600;T.save(D/'pipeline_registered.json',dict(UTC=T.utc(),pid=os.getpid()))
    while time.monotonic()<stop:
        try:
            prepare()
            for h in REMOTE:deploy(h)
            with concurrent.futures.ThreadPoolExecutor(2) as ex:done=list(ex.map(collect,REMOTE))
            counts={h:{k:len(T.read(D/(h+'_status.json'))[k]) for k in ['active','done','pending']} for h in REMOTE};G.F.progress('Y2','16个Audio1盲双耳增量分片已登记两机；'+str(counts),'提取按首批计时修订，预算1–3小时；18参数固定本人候选随后约10分钟')
            if all(done):readout();T.save(D/'evidence_complete.json',dict(UTC=T.utc(),records=5189,original_heads_frozen=True));return
        except Exception as e:T.save(D/'needs_attention.json',dict(UTC=T.utc(),error=repr(e),trace=traceback.format_exc()[-4500:]));print(traceback.format_exc(),flush=True)
        time.sleep(30)
if __name__=='__main__':main()
