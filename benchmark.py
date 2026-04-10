import importlib
import time
from functools import partial
from typing import NamedTuple

import fire
import jax
import jax.numpy as jnp
import jax.experimental.pallas.tpu as pltpu
from jax.sharding import PartitionSpec as P, reshard


class SamplingState(NamedTuple):
    token: jax.Array
    kv: object
    key: jax.Array


def _sample_step(forward, weights, token, kv, pos, key):
    logits, kv = forward(token, weights, kv, pos)
    key, subkey = jax.random.split(key)
    token = jax.random.categorical(subkey, logits[:, 0, :]).astype(jnp.int32)[:, None]
    return reshard(token, P("data", None)), kv, key


@partial(jax.jit, static_argnames=("forward",), donate_argnames=("kv",))
def sample_step(forward, weights, token, kv, pos, key):
    return _sample_step(forward, weights, token, kv, pos, key)


@partial(jax.jit, static_argnames=("forward", "init_kv", "steps"))
def sample_loop(forward, init_kv, weights, token, key, steps):
    state = SamplingState(reshard(token, P("data", None)), init_kv(token.shape[0], steps), key)

    def body(pos, state):
        token, kv, key = _sample_step(forward, weights, state.token, state.kv, pos, state.key)
        return SamplingState(token, kv, key)

    return jax.lax.fori_loop(0, steps, body, state)


@partial(jax.jit, static_argnames=("forward",), donate_argnames=("weights",))
def train_step(forward, weights, x, learning_rate):
    y = jnp.roll(x, -1, axis=1)

    def loss_fn(weights):
        logits = forward(x, weights).astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits)
        return -(jax.nn.one_hot(y, logits.shape[-1]) * log_probs).sum(axis=-1).mean()

    loss, grads = jax.value_and_grad(loss_fn)(weights)
    weights = jax.tree.map(lambda w, g: w - learning_rate * g, weights, grads)
    return weights, loss


def load_model(model, model_id=None, tp_devices=1, hf_ckpt_dir="~/weights/huggingface"):
    return importlib.import_module(f"models.{model}").load(
        tp_devices=tp_devices,
        hf_ckpt_dir=hf_ckpt_dir,
        **({} if model_id is None else {"model_id": model_id}),
    )


def peak_specs(peak_flops_per_device=None, peak_mem_bandwidth_per_device=None):
    info = pltpu.get_tpu_info()
    return (
        peak_flops_per_device or info.bf16_ops_per_second * info.num_cores,
        peak_mem_bandwidth_per_device or info.mem_bw_bytes_per_second * info.num_cores,
    )


def sampling(
    model="qwen3",
    model_id=None,
    batch_size=1,
    steps=512,
    tp_devices=None,
    hf_ckpt_dir="~/weights/huggingface",
    seed=0,
    peak_flops_per_device=None,
    peak_mem_bandwidth_per_device=None,
):
    devices = jax.devices()
    tp_devices = tp_devices or len(devices)
    peak_flops_per_device, peak_mem_bandwidth_per_device = peak_specs(
        peak_flops_per_device,
        peak_mem_bandwidth_per_device,
    )
    loaded = load_model(model, model_id, tp_devices, hf_ckpt_dir)
    key, token_key = jax.random.split(jax.random.key(seed))
    token = jax.random.randint(token_key, (batch_size, 1), 0, 1 << 15, dtype=jnp.int32)
    kv = loaded.init_kv(batch_size, steps)
    cost = sample_step.lower(loaded.forward, loaded.weights, token, kv, 0, key).compile().cost_analysis()
    flops = sum(v for k, v in cost.items() if k.startswith("flops"))
    del kv

    jax.block_until_ready(sample_loop(loaded.forward, loaded.init_kv, loaded.weights, token, key, steps))
    start = time.perf_counter()
    state = sample_loop(loaded.forward, loaded.init_kv, loaded.weights, token, key, steps)
    state = jax.block_until_ready(state)
    dt = (time.perf_counter() - start) / steps

    bytes_in_use = sum(device.memory_stats()["bytes_in_use"] for device in devices)
    weights_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(loaded.weights))
    kv_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(state.kv))
    hbm_bytes = (weights_bytes + kv_bytes) / len(devices)

    return {
        "model": model_id or model,
        "batch_size": batch_size,
        "steps": steps,
        "tp_devices": tp_devices,
        "device_kind": devices[0].device_kind,
        "device_count": len(devices),
        "time_per_step_sec": dt,
        "interactivity_tok_s": 1 / dt,
        "throughput_tok_s": batch_size / dt,
        "sampling_mfu": flops / dt / peak_flops_per_device,
        "memory_bandwidth_utilization": hbm_bytes / dt / peak_mem_bandwidth_per_device,
        "sampling_memory_bytes": bytes_in_use,
        "weights_bytes": weights_bytes,
        "kv_cache_bytes_theoretical": kv_bytes,
        "memory_footprint": bytes_in_use / (weights_bytes + kv_bytes),
        "sample_step_flops": flops,
        "sample_step_hbm_bytes_theoretical": hbm_bytes,
    }


def training(
    model="qwen3",
    model_id=None,
    batch_size=8,
    seq_len=512,
    tp_devices=1,
    hf_ckpt_dir="~/weights/huggingface",
    measure_steps=5,
    seed=0,
    learning_rate=1e-3,
    train_weights_dtype="float32",
    peak_flops_per_device=None,
):
    devices = jax.devices()
    peak_flops_per_device, _ = peak_specs(peak_flops_per_device, 1.0)
    loaded = load_model(model, model_id, tp_devices, hf_ckpt_dir)
    weights = jax.tree.map(lambda x: x.astype(train_weights_dtype), loaded.weights)
    x = jax.random.randint(jax.random.key(seed), (batch_size, seq_len), 0, 1 << 15, dtype=jnp.int32)

    cost = train_step.lower(loaded.forward, weights, x, learning_rate).compile().cost_analysis()
    flops = sum(v for k, v in cost.items() if k.startswith("flops"))

    weights, loss = train_step(loaded.forward, weights, x, learning_rate)
    weights, loss = jax.block_until_ready((weights, loss))

    start = time.perf_counter()
    for _ in range(measure_steps):
        weights, loss = train_step(loaded.forward, weights, x, learning_rate)
    weights, loss = jax.block_until_ready((weights, loss))
    dt = (time.perf_counter() - start) / measure_steps

    return {
        "model": model_id or model,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tp_devices": tp_devices,
        "device_kind": devices[0].device_kind,
        "device_count": len(devices),
        "time_per_step_sec": dt,
        "steps_per_sec": 1 / dt,
        "tokens_per_sec": batch_size * seq_len / dt,
        "training_mfu": flops / dt / peak_flops_per_device,
        "train_step_flops": flops,
        "num_parameters": sum(x.size for x in jax.tree.leaves(loaded.weights)),
        "loss": float(loss),
        "train_weights_dtype": train_weights_dtype,
    }


if __name__ == "__main__":
    fire.Fire({"sampling": sampling, "training": training})
