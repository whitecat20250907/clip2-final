"""Read-only official EVAL integrity and frozen YJ29 input audit.

Writes only the explicitly owned audit directory. No model, fit, selection,
prediction, submission, or changes to the source confidence directories.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import tarfile


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    owner = Path('/home/lisiyi/zhaojiahao_1').resolve()
    out = Path(args.output).resolve()
    assert os.getuid() == owner.stat().st_uid == 1030
    assert out != owner and out.is_relative_to(owner)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'audit.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'complete.json').exists():
        print((out / 'complete.json').read_text())
        return

    def save(name, value):
        p = out / name
        temp = p.with_suffix('.next')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        temp.replace(p)

    start = time.monotonic()
    def progress(stage, **values):
        save('progress.json', dict(stage=stage, PID=os.getpid(),
             seconds=time.monotonic()-start, **values))

    previous_path=out/'previous_integrity_stage_evidence.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
    if previous:
        assert hashlib.sha256(previous['original_source'].encode()).hexdigest()==previous['original_source_SHA']=='bfaf38ed39e11b323298a13515a3a82187397f6b8178c61016c3b76f0cba1fec'
        assert hashlib.sha256(previous['failure_log'].encode()).hexdigest()==previous['failure_log_SHA']
        assert previous['preceding_files_verified']==4379 and previous['previous_process_exited']
    progress('archive_md5')
    root = Path('/home/yangjing/workplace/clip2/clip2_eval/cadenza_data/clip2')
    archive = root.parents[1] / 'cadenza_clip2_data.eval.v1.0.tar.gz'
    md5 = json.loads((out/'archive.json').read_text())['md5'] if previous else digest(archive, 'md5')
    official = dict(url='https://zenodo.org/records/23189482',
                    filename=archive.name, bytes=archive.stat().st_size,
                    md5=md5, expected_md5='7bda62bf1e394eb69b11f041d77fcca1')
    save('archive.json', official)
    assert md5 == official['expected_md5']
    metadata = root / 'metadata/eval_metadata.json'
    assert digest(metadata) == 'ea872c41a5967aff742b5a353c3fd9bd9d483919877ae4cb8320db0ab9418d94'
    rows = json.loads(metadata.read_text())
    ids = [r['signal'] for r in rows]
    assert len(ids) == len(set(ids)) == 2189
    assert all(not {'correctness', 'response', 'responses', 'target', 'label'} & r.keys() for r in rows)
    canonical = json.loads((out / 'eval_public_manifest.json').read_text())['rows']
    assert [r['signal_id'] for r in canonical] == ids

    progress('extracted_files_md5')
    manifest = root / 'manifest/cadenza_CLIP2.eval.v1_0.md5sum'
    if previous:assert digest(manifest)==previous['manifest_SHA']
    checked = []
    self_entry=None
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(None, 1)
        relative = relative.lstrip('*')
        candidates = [root / relative, root.parents[1] / relative]
        found = [p.resolve() for p in candidates if p.is_file()]
        assert found, relative
        path = found[0]
        assert path.is_relative_to(root), path
        if path==manifest:
            self_entry=dict(expected_md5=expected,actual_md5=digest(path,'md5'))
            continue
        actual = expected if previous else digest(path, 'md5')
        assert actual == expected, str(path)
        checked.append(dict(path=str(path), md5=actual))
        if len(checked) % 256 == 0:
            progress('extracted_files_md5', checked=len(checked))
    save('extracted_files.json', checked)
    assert len(checked)==4379 and self_entry is not None

    import numpy as np
    import pandas as pd
    import importlib.util
    source = out / 'asr_bank.py'
    assert digest(source) == 'e656814808f69b9987eb6aa92783966f9728980df50a022904a34d118958958d'
    spec = importlib.util.spec_from_file_location('frozen_yj29', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    progress('confidence_YJ29')
    frames, records, proof = [], [], []
    for folder, filename in [('confidence_eval', 'binaural_feats.pkl'), ('confidence_v2_eval', 'dual_feats.pkl')]:
        base = root.parents[2] / folder
        raw = [json.loads(line) for line in (base / 'confidence.jsonl').read_text().splitlines() if line.strip()]
        assert [r['signal'] for r in raw] == ids
        assert all(not r.get('error') for r in raw)
        frame = pd.read_pickle(base / filename)
        assert len(frame) == len(ids)
        assert np.array_equal(frame.index.to_numpy(), np.arange(len(ids)))
        frames.append(frame)
        records.append(raw)
        proof.append(dict(path=str(base), files={n:digest(base/n) for n in [filename,'confidence.jsonl','meta.json']},
                          columns=list(frame.columns), rows=len(frame), actual_split='EVAL',
                          declared_metadata=json.loads((base/'meta.json').read_text()),
                          correctness_is_placeholder=bool((frame['correctness'].to_numpy()==0).all()),
                          correctness_excluded=True, exact_canonical_ID_order=True))
    v3, dual = frames
    columns = []
    for name in module.YJ29_FIELDS:
        if name in ('sub_rate','del_rate','ins_rate'):
            columns.append(np.nanmean(v3[[name+'_L',name+'_R']].to_numpy(np.float64), axis=1))
        elif name in ('hl_mild','hl_moderate') or (name in v3.columns and name not in dual.columns):
            columns.append(v3[name].to_numpy(np.float64))
        else:
            columns.append(dual[name].to_numpy(np.float64))
    selected = np.nan_to_num(np.column_stack(columns), nan=0., posinf=0., neginf=0.)
    recomputed = np.stack([module.yj29_vector(a,b,r['prompt'],r['hearing_loss'])
                          for a,b,r in zip(records[0],records[1],rows)])
    gap = float(np.abs(selected-recomputed).max())
    assert gap <= 1e-5, gap
    with (out/'YJ29_verified_EVAL2189.npz').open('wb') as f:
        np.savez_compressed(f, signal_ids=np.asarray(ids), features=recomputed,
                            fields=np.asarray(module.YJ29_FIELDS))
    oldbase=owner/'runs/v10_EVAL_AB_20261020_0200/fresh'
    deployed={}
    for name,new in [('v3',records[0]),('v2',records[1])]:
        path=oldbase/(name+'.jsonl')
        if path.is_file():
            old=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert [r['signal'] for r in old]==ids
            deployed[name]=dict(path=str(path),SHA=digest(path),
                rows_equal=sum(a==b for a,b in zip(old,new)),rows=len(ids),
                same_bytes=digest(path)==proof[0 if name=='v3' else 1]['files']['confidence.jsonl'])
    save('confidence_qualification.json',dict(confidence=proof,YJ29_shape=list(recomputed.shape),
         YJ29_jsonl_vs_pickle_maxgap=gap,existing_AB_confidence_comparison=deployed))
    # The release checksum list includes an obsolete self-checksum. Verify its
    # actual bytes against the official, fully MD5-qualified archive instead.
    progress('checking_archive_manifest_bytes')
    class HashedReader:
        def __init__(self,f):self.f=f;self.md5=hashlib.md5()
        def read(self,n=-1):
            raw=self.f.read(n);self.md5.update(raw);return raw
    found={}
    wanted={'cadenza_data/clip2/manifest/cadenza_CLIP2.eval.v1_0.md5sum':manifest,
            'cadenza_data/clip2/metadata/eval_metadata.json':metadata}
    with archive.open('rb') as f:
        reader=HashedReader(f)
        with tarfile.open(fileobj=reader,mode='r|gz')as tar:
            for member in tar:
                name=member.name.removeprefix('./')
                if name in wanted:
                    raw=tar.extractfile(member).read()
                    assert raw==wanted[name].read_bytes(),name
                    found[name]=hashlib.sha256(raw).hexdigest()
        while reader.read(4*1024*1024):pass
        assert reader.md5.hexdigest()==official['expected_md5']
    assert set(found)==set(wanted)
    self_entry.update(matches_official_archive_bytes=True,archive_member_SHA=found)
    result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        official_archive=official,metadata_SHA=digest(metadata),rows=2189,
        extracted_manifest_SHA=digest(manifest),extracted_files_verified=len(checked),
        confidence=proof,YJ29_fields=list(module.YJ29_FIELDS),YJ29_shape=list(recomputed.shape),
        manifest_self_entry=self_entry,previous_integrity_stage_SHA=digest(previous_path)if previous else None,
        YJ29_jsonl_vs_pickle_maxgap=gap,YJ29_SHA=digest(out/'YJ29_verified_EVAL2189.npz'),
        existing_AB_confidence_comparison=deployed,training_updates=0,model_changes=0,
        original_source_files_modified=False,seconds=time.monotonic()-start)
    save('complete.json',result)
    progress('complete', rows=2189)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
