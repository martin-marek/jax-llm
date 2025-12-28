import jax
import jax.numpy as jnp
import numpy as np
from qwen3 import load
from sampling import sample


# load model
model = load()

# forward pass (random input)
x = jax.random.randint(jax.random.key(0), [8, 32], 0, 1<<15)
logits = model.forward(x, model.weights)
print(logits.shape)

# sampling (random input)
x = jax.random.randint(jax.random.key(0), [8, 1], 0, 1<<15)
kv = model.init_kv(8, 16)
logits, kv = model.forward(x, model.weights, kv, 0)
print(logits.shape)

# sampling (text)
tokens = model.tokenizer('Alchemy is')['input_ids']
pad_id = model.tokenizer.pad_token_id
tokens_padded = np.full([1, 32], pad_id, dtype=np.int32)
tokens_padded[:, :len(tokens)] = tokens
out_tokens = sample(jax.random.key(0), model.forward, model.init_kv, model.weights, model.tokenizer, tokens_padded, 0.01)
out_text = model.tokenizer.batch_decode(out_tokens)
for response in out_text:
    print(response)
