
# Getting started

This library provides minimal implementations of language models in pure JAX with support for training, inference, FSDP, TP, and HuggingFace checkpoints. Each model is implemented in a standalone <160 LOC file. Because the implementations are so short, the hope is that you can use them as a reference for implementing new models -- e.g. `qwen3.py` and `llama3.py` differ in just 3 LOC!

A model is defined as tuple of weights and pure functions: `weights`, `forward`, `init_kv`, and `tokenizer`. Here's an example of a forward pass:

```python
from models import qwen3
model = qwen3.load('Qwen/Qwen3-4B')
x = jnp.ones([8, 128], dtype=jnp.int32)
logits, kv = model.forward(x, model.weights)
```

The weights and tokenizer are loaded directly from a HuggingFace checkpoint -- there's no need to run any separate scripts to convert between checkpoint formats.

The following architectures are currently supported:
- [x] Qwen3 0.6B...32B (dense only)
- [x] Llama3
- [x] Olmo2

# Benchmarks

### Training
| Device | Model | Batch size | Seq. length | TP devices | MFU |
| --- | --- | --- | --- | --- | --- |
| TPU v6e-1 | Qwen3-4B | 8 | 512 | 1 | 28% |
| TPU v6e-8 | Qwen3-8B | 128 | 512 | 1 | 30% |

### Sampling
| Device | Model | Batch size | Seq. length | TP devices | Interactivity | Throughput | HBM bw. util. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TPU v6e-8 | Qwen3-32B | 1 | 512 | 8 | 108 tok/s | 108 tok/s | 28% |
| TPU v6e-8 | Qwen3-32B | 512 | 512 | 8 | 42 tok/s | 21,600 tok/s | 33% |
| TPU v6e-8 | Qwen3-32B | 128 | 4096 | 8 | 42 tok/s | 5,400 tok/s | 56% |

Benchmarks were measured using [this notebook](examples/profiling.ipynb).

# FAQ

- **Q:** What sharding should I use? **A:** Each model uses a 2D device mesh: one dimension for TP (Tensor Parallelism) and the other dimension for FSDP (Fully Sharded Data Parallel). In most cases, it is preferable to use pure FSDP for training (`tp_devices=1`) and pure TP for inference (e.g. `tp_devices=8`).
- **Q:** What's the main performance bottleneck? **A:** Currently it is the attention implementation ([jax.nn.dot_product_attention](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)), which uses a simple dot product. Once there's a universal one-line flash attention implementation in [tokamax](https://github.com/openxla/tokamax/tree/main), I will switch to that. For more perfomant (but also more complex) JAX model implementations, please see [jax-ml/jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main).

# Notation

All model implementations use the following notation:
- B = batch size
- S = length of key/value (source)
- T = length of query (target)
- D = embedding dimension
- F = hidden dimension
- H = head dimension
- N = number of query heads
- K = number of key/value heads
- G = number of groups, equal to N // K
- V = vocab size
