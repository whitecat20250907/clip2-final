"""Audio-only compatibility structure and attention-math check; no trained weights."""
import os,sys,json,hashlib,time,importlib.util,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent;OUT=H/'structural_preflight_v1';OUT.mkdir(exist_ok=False)
sys.dont_write_bytecode=True
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
try:
 import numpy as np,torch
 torch.set_num_threads(2)
 spec=importlib.util.spec_from_file_location('_qwen_audio_only',H/'audio_only.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 cfg=json.loads((ROOT/'native_decoder_feasibility_v1/public_metadata_relay_v1/config.json').read_text())['thinker_config']['audio_config']
 header=json.loads((ROOT/'audio_tower_weight_range_preflight_v1/model-00001-of-00002.safetensors.header.json').read_text())
 model=m.make_meta_audio(cfg);expected={k.split('thinker.audio_tower.',1)[1]:v for k,v in header.items() if k.startswith('thinker.audio_tower.')}
 actual=model.state_dict();assert set(actual)==set(expected) and len(actual)==397
 for k,v in actual.items():assert list(v.shape)==expected[k]['shape']
 torch.manual_seed(20261015);q=torch.randn(2,4,9,16);k=torch.randn(2,4,13,16);v=torch.randn(2,4,13,16);mask=torch.zeros(2,1,9,13);mask[:,:,:,-2:]=torch.finfo(torch.float32).min
 dummy=type('M',(),{'num_key_value_groups':1,'training':False})()
 eager,_=m.eager_attention_forward(dummy,q,k,v,mask,16**-.5)
 sdpa,_=m.sdpa_audio_forward(dummy,q,k,v,mask,16**-.5)
 gap=float(abs(eager-sdpa).max());assert gap<1e-6
 params=sum(v.numel() for v in actual.values());assert params*2==634955008
 rep={'status':'audio_only_structure_and_attention_math_passed','weights_loaded':False,'training':False,'n_parameters':params,'weight_tensor_keys':len(actual),'header_shapes_exact':True,'float32_SDPA_vs_native_eager_maxgap':gap,'audio_config':{k:cfg[k] for k in ['d_model','encoder_layers','encoder_attention_heads','encoder_ffn_dim','num_mel_bins','output_dim','n_window','n_window_infer']},'implementation_sha256':hashlib.sha256((H/'audio_only.py').read_bytes()).hexdigest(),'source_reference_sha256':hashlib.sha256((H/'reference/modeling_qwen3_asr.py').read_bytes()).hexdigest(),'limits':'Structure and isolated attention math only. Does not yet establish whole pretrained encoder equivalence; cached real TRAIN frame/scorer replay is required beforeVALID.'}
 with (OUT/'report.json').open('x') as f:json.dump(rep,f,indent=2)
 (OUT/'exitcode').write_text('0');print(json.dumps(rep),flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
