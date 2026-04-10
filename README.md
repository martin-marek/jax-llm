
# Getting started

This library provides minimal implementations of language models in pure JAX with support for training, inference, FSDP, TP, and HuggingFace checkpoints. Each model is implemented in a standalone <160 LOC file. Because the implementations are so short, the hope is that you can use them as a reference for implementing new models -- e.g. [qwen3.py](models/qwen3.py) and [llama3.py](models/llama3.py) differ in just 3 LOC!

A model is defined as tuple of weights and pure functions: `weights`, `forward`, `init_kv`, and `tokenizer`. Here's an example of a forward pass:

```python
import jax.numpy as jnp
from models import qwen3
model = qwen3.load('Qwen/Qwen3-0.6B-Base')
x = jnp.ones([8, 128], dtype=jnp.int32)
logits, _ = model.forward(x, model.weights)
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
| TPU v6 lite-1 | Qwen3-4B | 8 | 512 | 1 | 35% |
| TPU v6 lite-8 | Qwen3-8B | 128 | 512 | 1 | 25% |

### Sampling
| Device | Model | Batch size | Seq. length | TP devices | Interactivity | Throughput | HBM bw. util. | Memory footprint | MFU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TPU v6 lite-8 | Qwen3-32B | 1 | 512 | 8 | 108 tok/s | 108 tok/s | 54% | 1.01x | 0.18% |
| TPU v6 lite-8 | Qwen3-32B | 512 | 512 | 8 | 41 tok/s | 20,992 tok/s | 42% | 1.00x | 18.6% |
| TPU v6 lite-8 | Qwen3-32B | 128 | 4096 | 8 | 42 tok/s | 5,363 tok/s | 65% | 1.00x | 5.3% |

# Example

Here's how to sample a short completion:

```python
import jax
import numpy as np
from models.qwen3 import load
from models.sampling import sample

model = load("Qwen/Qwen3-0.6B-Base")
prompt = "Alchemy is"
prompt_tokens = model.tokenizer(prompt)["input_ids"]
tokens = np.full([1, 32], model.tokenizer.pad_token_id, dtype=np.int32)
tokens[:, : len(prompt_tokens)] = prompt_tokens
out_tokens = sample(
    jax.random.key(0),
    model.forward,
    model.init_kv,
    model.weights,
    model.tokenizer,
    tokens,
    temperature=0.01,
)
print(model.tokenizer.batch_decode(out_tokens)[0])
```

# Requirements

For running on TPU, this should be sufficient:
```bash
pip install jax[tpu] transformers
```

# FAQ

- **Q:** What sharding should I use? **A:** Each model uses a 2D device mesh: one dimension for TP (Tensor Parallelism) and the other dimension for FSDP (Fully Sharded Data Parallel). In most cases, it is preferable to use pure FSDP for training (`tp_devices=1`) and pure TP for inference (e.g. `tp_devices=8`).
- **Q:** What's the main performance bottleneck? **A:** Currently it is the attention implementation ([jax.nn.dot_product_attention](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)), which uses a simple dot product. Once there's a universal one-line flash attention implementation in [tokamax](https://github.com/openxla/tokamax/tree/main), I will switch to that. For more perfomant JAX model implementations, please see [jax-ml/jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main).

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
