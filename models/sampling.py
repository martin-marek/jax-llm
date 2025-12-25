import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import PartitionSpec as P, reshard
from typing import NamedTuple


class SamplingState(NamedTuple):
    step: int
    key: jax.Array
    tokens: jax.Array # [B, T]
    kv: jax.Array # [B, T, ...]
    done: jax.Array # [B]


def _sample_step(state, forward, weights, tokenizer, temperature=1):
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # sample next token
    key, key_sampling = jax.random.split(state.key)
    input_token = state.tokens[:, state.step, None] # [B, 1]
    logits, kv = forward(input_token, weights, state.kv, state.step) # [B, 1, V]
    sampled_token = jax.random.categorical(key_sampling, logits[:, 0, :]/temperature)

    # update buffer
    next_token = state.tokens[:, state.step+1]
    update_token = jnp.where((~state.done) & (next_token==pad_id), sampled_token, next_token)
    tokens = state.tokens.at[:, state.step+1].set(update_token)

    # check if sampling is done
    done = state.done | ((next_token==pad_id) & (sampled_token==eos_id))

    return SamplingState(state.step+1, key, tokens, kv, done)


@partial(jax.jit, static_argnames=('forward', 'init_kv', 'tokenizer'))
def sample(key, forward, init_kv, weights, tokenizer, tokens, temperature=1):
    B, T = tokens.shape
    tokens = reshard(tokens, P('data', None))

    # initialize state
    state = SamplingState(
        step=0,
        key=key,
        tokens=tokens,
        kv=init_kv(B, T),
        done=jnp.zeros([B], dtype=bool, out_sharding=P('data')),
    )

    # sample next token inside a while loop
    step_fn = lambda state: _sample_step(state, forward, weights, tokenizer, temperature)
    cond_fn = lambda state: (state.step < T) & jnp.any(~state.done)
    state = jax.lax.while_loop(cond_fn, step_fn, state)

    return state.tokens
