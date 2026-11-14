"""Read-only source-model frontend witness in the preserved original runtime."""
import argparse, hashlib, json, pathlib, os, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import torch, librosa, transformers, peft, soundfile, soxr
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    processor = AutoProcessor.from_pretrained(a.model, fix_mistral_regex=True,
                                              local_files_only=True)
    prefix = processor.apply_chat_template([[{'role': 'system', 'content': ''},
        {'role': 'user', 'content': [{'type': 'audio', 'audio': None}]}]],
        add_generation_prompt=True, tokenize=False)[0] + 'language English<asr_text>'
    records = []
    for row in json.loads((root / 'selection.json').read_text()):
        part = 'valid' if row['role'] == 'VALID_CONTROL' else 'eval'
        path = pathlib.Path(a.data) / part / 'signals' / (row['signal'] + '.flac')
        assert sha(path) == row['audio_SHA']
        wave, sr = librosa.load(path, sr=16000, mono=True)
        x = processor(text=[prefix + row['prompt']], audio=[wave],
                      return_tensors='pt', padding=True, truncation=False).to('cpu')
        payload = {'wave': wave}
        for key, val in x.items():
            payload[key] = val.detach().numpy()
        x.to(torch.bfloat16)
        payload['BF16_input_features_bits'] = x['input_features'].view(torch.uint16).numpy()
        out = root / (row['signal'] + '.npz')
        np.savez_compressed(out, **payload)
        records.append(dict(signal=row['signal'], audio_SHA=sha(path),
                            frontend_SHA=sha(out), fields={k: list(v.shape) for k, v in payload.items()}))
    result = dict(records=records, prefix=prefix,
        tokenizer_backend_SHA=hashlib.sha256(processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        versions={k: v.__version__ for k, v in [('torch', torch), ('librosa', librosa),
             ('numpy', np), ('transformers', transformers), ('peft', peft),
             ('soundfile', soundfile), ('soxr', soxr)]}, training_updates=0,
        GPU_used=False, worker_SHA=sha(__file__))
    (root / 'complete.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
