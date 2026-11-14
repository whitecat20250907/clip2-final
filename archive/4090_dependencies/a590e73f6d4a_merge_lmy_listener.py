"""Merge completed, disjoint listener ranges; preserve all original job evidence."""
from pathlib import Path
import hashlib, json, os, signal


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    assert os.getuid() == 1030
    r = Path('/home/lisiyi/zhaojiahao_1/runs/clip2_final_TEST2189_20261028')
    mainroot, partroot = r/'LMY_LISTENER_TORCH26', r/'LMY_LISTENER_TORCH26_PART1'
    mainjob, partjob = mainroot/'tasks/listener_0', partroot/'tasks/listener_part1'
    out = r/'LMY_LISTENER_MERGED'
    assert not (out/'complete.json').exists()
    pg = json.loads((mainjob/'progress.json').read_text())
    if pg['done'] < 1295 or not (partjob/'complete.json').exists():
        print(json.dumps(dict(ready=False, first_range_done=pg['done'], second_range_complete=(partjob/'complete.json').exists())))
        return
    bindings = [(mainroot, 'deaf3e5cbda9b883b22e8bc6cfd839881cde4082ccc6097cf14e3af219dd6146'),
                (partroot, '4c4f9c94c8cb61c031faf914f5cd66efc24d0c16b02008cba487392eb16bba63')]
    sources = {}
    for root, expected in bindings:
        assert sha(root/'binding.json') == expected
        cfg = json.loads((root/'binding.json').read_text())
        assert cfg['training_updates'] == 0
        for item in cfg['files']:
            # The wrapper already verified large weights before inference.
            # Recheck all small source/configuration bindings after inference.
            if Path(item['path']).stat().st_size < 4*1024**2:
                assert sha(item['path']) == item['SHA'], item['path']
        sources[str(root)] = dict(binding_SHA=expected, bound_files=cfg['files'])
    leftcfg = json.loads((mainroot/'protocol.json').read_text())
    rightcfg = json.loads((partroot/'protocol.json').read_text())
    for k in ['sources', 'model', 'adapter', 'asset_protocol', 'valid_reference', 'torch']:
        assert leftcfg[k] == rightcfg[k], k
    rows = json.loads((mainroot/'selection.json').read_text())
    partrows = json.loads((partroot/'selection.json').read_text())
    assert len(rows) == 2191 and partrows == rows[:2]+rows[1297:]
    controlproof = {}
    for name, job in [('main', mainjob), ('part1', partjob)]:
        proof = json.loads((job/'VALID_control_passed.json').read_text())
        assert len(proof['cases']) == 2 and all(x['maxgap'] == 0 for x in proof['cases'])
        controlproof[name] = proof
    pc = json.loads((partjob/'complete.json').read_text())
    assert pc['rows'] == 894 and pc['parameters_unchanged'] and pc['training_updates'] == 0
    assert sha(partjob/'EVAL_LISTENER_WORD3.npz') == pc['output_SHA']
    (out/'rows').mkdir(parents=True, exist_ok=True)
    values, lengths, receipt = [], [], []
    overlap = []
    for i, row in enumerate(rows):
        src = (mainjob if i < 1297 else partjob)/'rows'/(row['signal']+'.npz')
        with np.load(src, allow_pickle=False) as z:
            assert str(z['signal']) == row['signal'] and str(z['audio_SHA']) == row['audio_SHA']
            v = z['adapted_word3'].copy()
            assert v.shape == (row['n_words'], 3) and np.isfinite(v).all()
        if i < 2 or (1297 <= i < pg['done']+2):
            other = (partjob if i < 2 else mainjob)/'rows'/src.name
            with np.load(other, allow_pickle=False) as z:
                assert np.array_equal(z['adapted_word3'], v)
            overlap.append(row['signal'])
        dest = out/'rows'/src.name
        if not dest.exists():
            os.link(src, dest)
        assert sha(src) == sha(dest)
        receipt.append(dict(signal=row['signal'], source=str(src), SHA=sha(dest)))
        if i >= 2:
            values.append(v); lengths.append(len(v))
    output = out/'EVAL_LISTENER_WORD3.npz'
    np.savez_compressed(output, signals=np.asarray([x['signal'] for x in rows[2:]]),
                        lengths=np.asarray(lengths), adapted_word3=np.concatenate(values))
    (out/'row_manifest.json').write_text(json.dumps(receipt, indent=2))
    # All required rows are safely linked and validated. Stop only this owned
    # original job, whose remaining range duplicates the completed partition.
    registration = json.loads((mainroot/'wrapper_jobs/listener_0/registration.json').read_text())
    pid = registration['PID']; proc = Path('/proc')/str(pid)
    stopped = False
    if proc.exists() and not (mainjob/'complete.json').exists():
        cmd = (proc/'cmdline').read_bytes().split(b'\0')
        assert proc.stat().st_uid == registration['UID'] == 1030
        assert str(mainroot).encode() in cmd and b'listener_0' in cmd
        assert bindings[0][1].encode() in cmd
        os.kill(pid, signal.SIGTERM)
        stopped = True
    report = dict(rows=2189, controls_passed=True, VALID_controls=controlproof,
        parameters_unchanged=True, training_updates=0, output_SHA=sha(output),
        row_manifest_SHA=sha(out/'row_manifest.json'), source_bindings=sources,
        first_range=[0,1295], second_range=[1295,2189], overlap_identical=overlap,
        main_partial_job_used=True, main_progress_snapshot=pg, main_PID=pid,
        main_stopped_after_required_rows_complete=stopped,
        part1_completion_SHA=sha(partjob/'complete.json'), source=str(Path(__file__)), source_SHA=sha(__file__))
    (out/'complete.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_bindings','overlap_identical']}))


if __name__ == '__main__':
    main()
