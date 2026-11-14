"""Bind the fixed parent inputs and weights inside the owned CPU work directory."""
import pathlib, json, hashlib, subprocess, os

R = pathlib.Path('/home/lisiyi/zhaojiahao_1/runs/clip2_final_TEST2189_20261028')
E = pathlib.Path('/home/lisiyi/clip2_lmy/CLIP2_lmy/my_experiments')
H = E / 'd6e_qwen_internalization_autoresearch_20261015'
BR = E / 'autodl_inference_dependency_backup_20261028_v1'
T = BR / 'mirror/root/autodl-tmp/clip2_runs/qwen_rms_strict24_exclusion_20261015_v1/training_pair_direct_v4'
out = R / 'LMY_PARENT_MAIN'


def sha(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    sources = json.loads((out / 'source_pins.json').read_text())
    paths = {
        'selection': R/'LMY_QWEN17/selection.json',
        'side_raw': R/'LMY_SIDE_RAW/tasks/raw_0/EVAL_LMY_PUBLIC_SIDE_RAW.npz',
        'lm4': R/'LMY_LM4/tasks/lm4_0/EVAL_REFERENCE_LM4.npz',
        'free_asr': R/'LMY_FREE_ASR/tasks/free_0/EVAL_FREE_ASR9_AVAILABILITY.npz',
        'text': R/'LMY_TEXT/tasks/text_0/EVAL_ROBERTA_TEXT.npz',
        'side_norm': E/'d6e_pooled_candidate_export_v1/results/export_v1/bundle/normalization.json',
        'context_norm': H/'posterior45_full_context_20261017_v1/TRAIN_CONTEXT326_NORMALIZATION_V1.json',
        'reference_f96': H/'qwen_hypothesis_prefix_confidence_20261017_v1/confidence_inputs_all22380.npz',
        'reference_main': H/'under26_native_posterior_inside_strong_main_hidden16_micro_20261020_v1/v2_school_public_cache_reconstruction/RECOVERED_PUBLIC_MAIN_INPUTS_ALL22380_V1.npz',
        'module_sources': T/'inputs/module_sources.json',
        'initial': T/'inputs/initial_model.pt',
        'strict24_head': T/'strict24_head.py',
        'x9_norm': H/'baseline_16_34531218_v1/normalization.json',
    }
    for key in ['paired_source', 'side_source', 'parent_source', 'x9_source']:
        paths[key] = pathlib.Path(sources[key]['path'])
    for seed in [44, 45]:
        folder = T/('side_research_v1/human_adapted_encoder_seed%d_v1/adapted_encoder_16180' % seed)
        for key, name in [('stage', 'stage1_final.pt'), ('beta', 'beta_final.pt'), ('parent_norm', 'normalization.json')]:
            paths[key+str(seed)] = folder/name
        paths['x9_head'+str(seed)] = T/('side_research_v1/noise_followup_20261016_v1/x9_seed%d_final.pt' % seed)
        paths['main_head'+str(seed)] = H/('own_fixed_snapshot_average26_10196_20261017_v1/production_artifacts/parameter_mean__full_adam__alignment381_seed%d/final_checkpoint.pt' % seed)
    backup = json.loads((BR/'VERIFIED_SMALL_DEPENDENCIES_MANIFEST_V1.json').read_text())
    # Locate the manifest's records without depending on its container key.
    records = []
    def visit(x):
        if isinstance(x, dict):
            if 'destination' in x and 'sha256' in x:
                records.append(x)
            else:
                for v in x.values(): visit(v)
        elif isinstance(x, list):
            for v in x: visit(v)
    visit(backup)
    expected = {x['destination']: x['sha256'] for x in records}
    assets = {}
    for key, p in paths.items():
        digest = sha(p)
        if str(p) in expected:
            assert digest == expected[str(p)], key
        if key in sources:
            assert digest == sources[key]['SHA'], key
        assets[key] = dict(path=str(p), SHA=digest)
    assert assets['module_sources']['SHA'] == '801c63447ac816be3cac2bb87e9125f31f30183b0ad67fd81411caedf6c134ca'
    assert assets['initial']['SHA'] == '16f734ecc2d3b413d83d24735cde18aad2b59290f294f09f2bf721d20ebebe7e'
    assert assets['x9_norm']['SHA'] == '9c126f47e5d83a845ca39898e2fb65849a59a4fe581c22ad3b72be0e726f1bc8'
    for key, task, role in [('side_raw', 'LMY_SIDE_RAW', 'raw_0'), ('lm4', 'LMY_LM4', 'lm4_0'),
                           ('free_asr', 'LMY_FREE_ASR', 'free_0'), ('text', 'LMY_TEXT', 'text_0')]:
        receipt = json.loads((R/task/'tasks'/role/'complete.json').read_text())
        assert receipt['output_SHA'] == assets[key]['SHA'], key
    cfg = dict(assets=assets, training_updates=0,
               word_rows=str(R/'LMY_WORD261_TORCH26/tasks/word_0/rows'),
               listener_rows=str(R/'LMY_LISTENER_TORCH26/tasks/listener_0/rows'),
               native_rows=str(R/'LMY_QWEN17/tasks/native_0/rows'))
    (out/'protocol.json').write_text(json.dumps(cfg, indent=2))
    (out/'stage_complete.json').write_text(json.dumps(dict(assets=len(assets), protocol_SHA=sha(out/'protocol.json'),
                                                        worker_SHA=sha(out/'worker.py'), training_updates=0), indent=2))
    py = '/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_private_runtime_20261019_0945/runtime/bin/python'
    p = subprocess.run([py, '-B', str(out/'worker.py'), '--root', str(out), '--phase', 'static'],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
    assert p.returncode == 0


if __name__ == '__main__':
    main()
