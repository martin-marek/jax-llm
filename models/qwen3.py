import json
import jax
import jax.numpy as jnp
from pathlib import Path
from safetensors import safe_open
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Dict
from functools import partial, reduce
from huggingface_hub import snapshot_download
from jax.sharding import PartitionSpec as P, AxisType, reshard
from transformers import PreTrainedTokenizerFast, AddedToken


@dataclass
class Model:
    weights: Dict
    forward: Callable
    init_kv: Callable
    tokenizer: Callable


def apply_rope(x, theta, pos=0):
    B, T, N, H = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))
    inp = jnp.einsum('bt,h->bth', positions, freq, precision=jax.lax.Precision.HIGHEST)
    sin, cos = jnp.sin(inp).astype(x.dtype), jnp.cos(inp).astype(x.dtype)
    x1, x2 = x[:, :, :, :H//2], x[:, :, :, H//2:]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :] # [B, T, 1, H/2]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def rms_norm(x, gamma, eps):
    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32)**2, axis=-1, keepdims=True) + eps)
    return (gamma * x / rms).astype(x.dtype)


def forward_layer(cfg, x, w, kv=None, pos=0):
    B, T, D = x.shape
    
    # input norm
    x_norm = rms_norm(x, w['input_layernorm'], cfg['rms_norm_eps'])
    
    # QKV projection
    q = jnp.einsum('btd,nhd->btnh', x_norm, w['q_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))
    k = jnp.einsum('bsd,khd->bskh', x_norm, w['k_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))
    v = jnp.einsum('bsd,khd->bskh', x_norm, w['v_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model', None))

    # Q/K norm
    q = rms_norm(q, w['q_norm'], cfg['rms_norm_eps'])
    k = rms_norm(k, w['k_norm'], cfg['rms_norm_eps'])
    
    # RoPE
    q = apply_rope(q, cfg['rope_theta'], pos)
    k = apply_rope(k, cfg['rope_theta'], pos)

    # load kv cahce
    if kv is not None:
        kv = jax.lax.dynamic_update_slice(kv, jnp.stack([k, v]), (0, 0, pos, 0, 0))
        k, v = kv

    # grouped-query attention
    attn_mask = jnp.tri(T, dtype=bool)[None] if kv is None else (jnp.arange(k.shape[1]) <= pos)[None, None] # [B, T, S]
    attn_out = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)

    # O projection
    o = jnp.einsum('btnh,dnh->btd', attn_out, w['o_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None, None))
    x += o

    # post-attention norm
    x_norm = rms_norm(x, w['post_attention_layernorm'], cfg['rms_norm_eps'])
    
    # MLP
    gate = jax.nn.silu(jnp.einsum('btd,fd->btf', x_norm, w['gate_proj'], preferred_element_type=jnp.float32, out_sharding=P('data', None, 'model')))
    up = jnp.einsum('btd,fd->btf', x_norm, w['up_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, 'model'))
    x += jnp.einsum('btf,df->btd', gate * up, w['down_proj'], preferred_element_type=x.dtype, out_sharding=P('data', None, None))

    return x, kv


def forward(cfg, x, weights, kv=None, pos=0):

    # embedding
    x = reshard(x, P('data', None))
    x = weights['embed_tokens'].at[x, :].get(out_sharding=P('data', None, None)).astype(jnp.bfloat16)
    
    # iterate over hidden layers
    if kv is None: kv = defaultdict(lambda: None)
    for i in range(cfg['num_hidden_layers']):
        layer_weights = {k.replace(prefix, ''):v for k,v in weights.items() if (prefix:=f'layers.{i}.') in k}
        x, kv[i] = jax.remat(partial(forward_layer, cfg))(x, layer_weights, kv[i], pos)

    # logits
    out_embed = weights['embed_tokens'] if cfg['tie_word_embeddings'] else weights['lm_head']
    x = rms_norm(x, weights['norm'], cfg['rms_norm_eps'])
    logits = jnp.einsum('btd,vd->btv', x, out_embed, preferred_element_type=x.dtype, out_sharding=P('data', None, 'model'))

    return logits, kv


def init_kv(L, K, H, B, T):
    sharding = P(None, 'data', None, 'model', None)
    kv = [jnp.zeros((2, B, T, K, H), dtype=jnp.bfloat16, out_sharding=sharding) for _ in range(L)]
    return kv


def load(model_id='Qwen/Qwen3-0.6B-Base', tp_devices=1, hf_ckpt_dir='~/weights/huggingface'):
    """loads huggingface checkpoint"""
    model_ckpt_dir = Path(hf_ckpt_dir).expanduser() / model_id

    # download checkpoint
    if not model_ckpt_dir.exists():
        snapshot_download(repo_id=model_id, local_dir=model_ckpt_dir)

    # load tokenizer
    tokenizer_config = json.loads((model_ckpt_dir/'tokenizer_config.json').read_text())
    tokenizer_config['added_tokens_decoder'] = {int(k): AddedToken(**v) for k, v in tokenizer_config['added_tokens_decoder'].items()}
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_ckpt_dir/'tokenizer.json'), **tokenizer_config)

    # load model config
    cfg = json.loads((model_ckpt_dir/'config.json').read_text())
    L, N, K, H, D = cfg['num_hidden_layers'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim'], cfg['hidden_size']

    # define sharding (FSDP + TP)
    fsdp_devices = jax.device_count() // tp_devices
    mesh = jax.make_mesh((fsdp_devices, tp_devices), ('data', 'model'), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    def get_sharding(key):
        if any(k in key for k in ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj')):
            return P('model', 'data')
        if any(k in key for k in ('o_proj', 'down_proj')):
            return P('data', 'model')
        if any(k in key for k in ('embed_tokens', 'lm_head')):
            return P('model', 'data')
        return P()

    # load weights
    weights = {}
    for file in model_ckpt_dir.glob('*.safetensors'):
        with safe_open(file, framework='numpy') as f:
            for key in f.keys():
                weights[key] = jax.device_put(f.get_tensor(key), get_sharding(key))

    # shorten layer keys
    substrings = ['model.', 'self_attn.', 'mlp.', '.weight']
    weights = {reduce(lambda k, s: k.replace(s, ''), substrings, k):v for k,v in weights.items()}
    
    # split head dimension
    for key in weights.keys():
        if 'q_proj' in key: weights[key] = weights[key].reshape([N, H, D])
        if 'k_proj' in key: weights[key] = weights[key].reshape([K, H, D])
        if 'v_proj' in key: weights[key] = weights[key].reshape([K, H, D])
        if 'o_proj' in key: weights[key] = weights[key].reshape([D, N, H])
    
    return Model(weights, partial(forward, cfg), partial(init_kv, L, K, H), tokenizer)
