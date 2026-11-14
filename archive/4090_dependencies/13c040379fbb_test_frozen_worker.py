"""Run an unchanged, SHA-pinned original feature worker on fixed public rows."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True)
    p.add_argument('--role',required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--binding-sha',required=True)
    a=p.parse_args()
    root=Path('/home/lisiyi/zhaojiahao_1').resolve()
    base=Path(a.base).resolve()
    assert base.is_relative_to(root) and base!=root
    assert base.stat().st_uid==root.stat().st_uid==os.getuid()==1030
    binding=base/'binding.json'
    assert sha(binding)==a.binding_sha
    cfg=json.loads(binding.read_text())
    assert cfg['wrapper_SHA']==sha(__file__)
    assert cfg['training_updates']==0
    assert a.role in cfg['roles']
    job=base/'wrapper_jobs'/a.role
    job.mkdir(parents=True,exist_ok=True)
    held=[]
    def lock(path):
        path.parent.mkdir(parents=True,exist_ok=True)
        f=path.open('a')
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        held.append(f)
    lock(job/'writer.lock')
    assert not (base/'tasks'/a.role/'complete.json').exists()
    assert not (job/'registration.json').exists(), 'Existing intent must be audited before a resume'
    def save(name,v):
        t=job/(name+'.next')
        t.write_text(json.dumps(v,ensure_ascii=False,indent=2))
        t.replace(job/name)
    try:
        for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR']:
            d=base/'private_cache'/a.role/key;d.mkdir(parents=True,exist_ok=True);os.environ[key]=str(d)
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        sys.dont_write_bytecode=True
        save('progress.json',dict(stage='verifying_sources_and_weights'))
        for item in cfg['files']:
            path=Path(item['path'])
            assert path.is_file() and sha(path)==item['SHA'],str(path)
        def gpu():
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
            matches=[]
            for line in lines:
                idx,uuid,free,util=[v.strip() for v in line.split(',')]
                if a.gpu in [idx,uuid]:matches.append(dict(index=idx,uuid=uuid,free=int(free),util=int(util)))
            assert len(matches)==1
            return matches[0]
        g=gpu()
        lock(root/'runs/v10_gpu_leases_20261018'/('gpu'+g['index']+'.lock'))
        lock(root/'runs/clip2_broad_trials_20261026/leases'/('GPU_'+g['uuid']+'.lock'))
        lock(root/'runs/clip2_final_TEST2189_20261028/leases'/('GPU_'+g['uuid']+'.lock'))
        for _ in range(2):
            g=gpu()
            assert g['free']>=cfg['minimum_free_MiB'],g
            assert g['util']<=cfg.get('maximum_admission_utilization',35),g
            time.sleep(1)
        save('registration.json',dict(PID=os.getpid(),UID=os.getuid(),GPU=g,
             UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             binding_SHA=a.binding_sha,source_SHA=sha(__file__),argv=sys.argv,training_updates=0))
        sys.path.insert(0,str(base))
        sys.argv=[str(base/'worker.py'),'--base',str(base),'--role',a.role]
        runpy.run_path(str(base/'worker.py'),run_name='__main__')
        complete=base/'tasks'/a.role/'complete.json'
        assert complete.is_file()
        save('complete.json',dict(original_worker_complete_SHA=sha(complete),training_updates=0))
    except BaseException as e:
        save('attention.json',dict(error=repr(e),trace=traceback.format_exc()))
        raise


if __name__=='__main__':main()
