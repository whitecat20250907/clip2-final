"""Inference-only engineering entry for the frozen CLIP2 final deployment.

Point mode extends the exact MF/Phi native scorers to the missing EVAL2189
inputs. It never calls a fit, optimizer, official API, or scheduler.
"""
import argparse, contextlib, datetime, hashlib, importlib.util, json, os
from pathlib import Path
import statistics, sys, time, traceback
from types import SimpleNamespace
sys.dont_write_bytecode = True
OWNER = Path('/home/douhan/zhaojiahao/b1_parallel/zhaojiahao_1')
OLD = OWNER / 'runs/clip2_jury_20261024'
RUN = OWNER / 'runs/clip2_final_TEST2189_20261028'
QC_SHA = '5c81f3f8d7bc822f59a5af39ced153a9f6b7bc73b7cb226b84bc5dd42a95da30'
POINT_SHA = '450e31615ac2dec81f823680d34f9444ca197075ab3b13fdf293638459942ec6'
CORE_SHA = 'd6acb30cbf6cf8b909e1bfb5fa07c9723ce36a5060c36aaf5dbf3988e4f4abe1'
HELPER_SHA = '87fa3943fb40bf598c2b78f52af5a167a87a322e397f3d26863c02999340d722'
SEATS = {
    'MF': ('jury_mf_moss_worker.py', CORE_SHA,
           'MF_MOSS_QC64_runtime_manifest.json',
           '2e68dbe67a31eca17065fe71dc008f608c394071888b52d0d2d823a7406658db'),
    'Phi4MM': ('jury_optional_mi_phi_qc64_worker.py',
              '6136626970419682f20d86cc93dea913ad42c53b91c79a0acca94fd29a7225af',
              'Phi4MM_QC64_binding_zerooutput_v2.json',
              '612407956bc1082808208fa526c338db305fdbba63d5ab4e8f71e8852e5004f0'),
}
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def checked(path, expected):
    p = Path(path).resolve()
    assert p.is_relative_to(OWNER) and sha(p) == expected, str(p)
    return p
def module(name, path, expected):
    p = checked(path, expected)
    spec = importlib.util.spec_from_file_location(name, p)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def point_prepare(a):
    cp = checked(a.config, a.config_sha); c = read(cp)
    assert c['schema'] == 'CLIP2.final.fixed_TEST2189.v1'
    assert c['worker_SHA'] == sha(__file__)
    assert c['final_selection_SHA'] == '7526380b08e8fc7a0ef2adc325621e1d5333c0860127b35c6ecb938c4982fcca'
    assert c['adapter_SHA'] == '444a577b1fd987867e1eda464c2053ca2045f1a9b670595f1de57a553686db89'
    assert os.getuid() == c['UID'] == 1004
    assert c['fits'] == c['platform_submissions'] == c['scheduled_tasks'] == 0
    assert c['seed'] == 20261024 and c['execution_authorized_by_current_user'] is True
    U = module('frozen_point_helpers', c['helper']['path'], HELPER_SHA)
    wf, wh, bf, bh = SEATS[a.judge]
    W = module('frozen_point_worker', OLD / 'runtime' / wf, wh)
    checked(OLD/'QC64_protocol.json', QC_SHA)
    binding_path = checked(OLD/'runtime'/bf, bh)
    args = SimpleNamespace(base=str(OLD), protocol=str(OLD/'QC64_protocol.json'),
        protocol_sha=QC_SHA, host='3090', judge=a.judge, reduce_only=True,
        audit_only=True, gpu=None, shards=1, shard_index=0, binding=str(binding_path),
        binding_sha=bh, runtime_manifest=str(binding_path), runtime_manifest_sha=bh)
    if a.judge == 'MF':
        _, p, _ = W.protocol_inputs(args); core = W
        binding = read(binding_path); model = p['models']['MF']
        assert Path(sys.executable).resolve() == Path(c['python']['MF']).resolve()
        for r in c['MF_weight_refs']: checked(r['path'], r['SHA'])
    else:
        _, p, binding, core, _ = W.protocol_and_tasks(args); model = binding['model']
    point = read(checked(OLD/'Qwen2Audio_unionEOS_POINT7378_protocol.json', POINT_SHA))
    assert point['prompts'] == read(OLD/'QC64_protocol.json')['prompts']
    inp = point['input_manifest']; inputs = read(checked(OLD/inp['path'], inp['SHA']))['rows']
    rows = point['rows'][5189:]; inputs = inputs[5189:]
    assert len(rows) == len(inputs) == 2189
    assert [r['signal_id'] for r in rows] == c['EVAL_signal_ids']
    assert len(set(c['EVAL_signal_ids'])) == 2189
    tasks = []
    for i, (r, src) in enumerate(zip(rows, inputs)):
        assert r['split'] == src['split'] == 'eval'
        assert r['signal_id'] == src['signal_id'] and r['audio_path'] == src['audio_path']
        path = Path(r['audio_path']).resolve()
        assert path == Path(p['data_roots']['eval']).resolve()/'signals'/(r['signal_id']+'.flac')
        assert path.stat().st_size == src['bytes']
        assert not any(k in r for k in ['target','label','correctness'])
        for pr in p['prompts']:
            for ear in (0,1):
                tasks.append(dict(row=r,row_index=i,prompt=pr,ear=ear,control='real',audio_SHA=src['audio_SHA']))
    assert len(tasks) == 26268
    identity = hashlib.sha256(U.canonical(dict(judge=a.judge,model=model,
        source_SHA=p['source_SHA'] if a.judge=='MF' else binding['source_SHA'],
        prompts_SHA=point['prompts_SHA'],input_manifest_SHA=inp['SHA'],seed=20261024,
        final_selection_SHA=c['final_selection_SHA'],scope='EVAL2189',
        numeric_expectation='exact_original_broad_positive_finite_native_mass'))).hexdigest()
    return c,U,core,p,binding,model,tasks,identity
def point_worker(a):
    c,U,core,p,binding,model,tasks,identity = point_prepare(a)
    if a.mode == 'audit':
        print(json.dumps(dict(audit_passed=True,judge=a.judge,rows=2189,measurements=len(tasks),
            identity_SHA=identity,fit=0,model_forward=0))); return
    job = next(j for j in c['jobs'] if j['job_ID'] == a.job)
    assert job['judge'] == a.judge and job['gpu'] == a.gpu
    assigned = tasks[job['row_start']*12:job['row_end']*12]
    d = RUN/'jobs'/a.job; d.mkdir(parents=True,exist_ok=True)
    def rec_path(t):
        return RUN/'measurements'/a.judge/(f"{t['row_index']:04d}_{t['prompt']['id']}_{t['ear']}.json")
    def existing(t):
        f=rec_path(t)
        if not f.exists(): return None
        q=read(f)
        assert q['identity_SHA']==identity and q['signal_id']==t['row']['signal_id']
        assert q['audio_source_SHA']==t['audio_SHA'] and q['prompt_SHA']==hashlib.sha256(t['prompt']['text'].encode()).hexdigest()
        assert q['ear']==t['ear'] and q['diagnostic']==U.expectation(q['measurement'])
        return q
    with contextlib.ExitStack() as stack:
        stack.enter_context(U.lock(RUN/'leases'/(a.job+'.lock')))
        initial=U.gpu_state(a.gpu)
        # Interoperate with the same original resource leases as prior workers.
        for lease in [OWNER/'runs/v10_gpu_leases_20261018'/('gpu'+str(a.gpu)+'.lock'),
                      OLD/'leases/GPUs'/('3090_'+str(a.gpu)+'.lock'),
                      OWNER/'runs/clip2_broad_trials_20261026/leases'/('GPU_'+initial['uuid']+'.lock'),
                      RUN/'leases'/('GPU_'+initial['uuid']+'.lock')]:
            stack.enter_context(U.lock(lease))
        first=U.gpu_state(a.gpu);time.sleep(1);second=U.gpu_state(a.gpu)
        assert first['uuid']==second['uuid']==initial['uuid']
        assert not first['computePIDs'] and not second['computePIDs']
        assert min(first['memory_free_MiB'],second['memory_free_MiB'])>=22600
        assert max(first['utilization'],second['utilization'])<=25
        os.environ.update(CUDA_VISIBLE_DEVICES=second['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        for key in ['HF_HOME','HF_MODULES_CACHE','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR']:
            q=d/'cache'/key;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
        if (d/'registration.json').exists():
            previous=read(d/'registration.json');assert not Path('/proc/'+str(previous['PID'])).exists()
            assert a.resume,'Explicit engineering resume only, after old PID is proven absent'
        U.publish(d/'registration.json',dict(UTC=utc(),PID=os.getpid(),UID=os.getuid(),argv=sys.argv,
            source_SHA=sha(__file__),config_SHA=a.config_sha,identity_SHA=identity,GPU=second,
            rows=[job['row_start'],job['row_end']],fits=0),mutable=a.resume)
        done=0;new=0;reused=0;times=[];start=time.monotonic();session=None
        try:
            for t in assigned:
                if existing(t) is not None:done+=1;reused+=1;continue
                with U.lock(RUN/'leases/measurements'/a.judge/(rec_path(t).stem+'.lock')):
                    if existing(t) is not None:done+=1;reused+=1;continue
                    if session is None:
                        if a.judge=='MF':session=core.NumericSession(OLD,p,'MF',d)
                        else:
                            A=module('frozen_Phi_native',OLD/binding['source_files']['adapter'],binding['source_SHA']['adapter'])
                            session=A.NativeNumericSession('Phi4MM',model,20261024,.8,lambda n,q:U.publish(d/n,dict(UTC=utc(),**q)))
                    if new%12==0:
                        now=U.gpu_state(a.gpu)
                        assert now['uuid']==second['uuid'] and set(now['computePIDs'])<={os.getpid()}
                    ts=time.monotonic();measurement=session.score(t);elapsed=time.monotonic()-ts
                    assert measurement['direction']==t['prompt']['direction']
                    assert measurement['numeral_token_ids']==model['expected_numeral_token_ids']
                    eos=model['termination_policy']['stop_ids'] if 'termination_policy' in model else model['native_EOS']
                    assert measurement['native_EOS']==eos and measurement['uncached_full_forward']
                    assert measurement['context_forward_count']==1
                    diagnostic=U.expectation(measurement)
                    U.publish(rec_path(t),dict(UTC=utc(),complete=True,judge=a.judge,identity_SHA=identity,
                        source_SHA=sha(__file__),signal_id=t['row']['signal_id'],row_index=t['row_index'],
                        split='eval',audio_source_SHA=t['audio_SHA'],ear=t['ear'],prompt_id=t['prompt']['id'],
                        prompt_SHA=hashlib.sha256(t['prompt']['text'].encode()).hexdigest(),
                        measurement=measurement,diagnostic=diagnostic,wall_seconds=elapsed))
                    done+=1;new+=1;times.append(elapsed)
                    if new<=2 or new%12==0:
                        steady=times[2:] or times; med=statistics.median(steady)
                        U.publish(d/'progress.json',dict(UTC=utc(),PID=os.getpid(),judge=a.judge,
                            assigned=len(assigned),completed=done,new=new,reused=reused,
                            elapsed_seconds=time.monotonic()-start,steady_median_seconds=med,
                            remaining_eta_seconds=(len(assigned)-done)*med),True)
            U.publish(d/'complete.json',dict(UTC=utc(),identity_SHA=identity,measurements=done,new=new,reused=reused,fits=0))
        except BaseException as e:
            U.publish(d/'attention.json',dict(UTC=utc(),PID=os.getpid(),error=repr(e),traceback=traceback.format_exc()),True)
            raise
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['audit','point'],default='audit')
    p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True)
    p.add_argument('--judge',choices=list(SEATS),required=True)
    p.add_argument('--job');p.add_argument('--gpu',type=int);p.add_argument('--resume',action='store_true')
    a=p.parse_args();point_worker(a)
if __name__=='__main__':main()

