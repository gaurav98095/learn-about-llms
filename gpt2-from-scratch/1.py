"""Run the complete GPT-2 model on a tiny piece of dummy text."""

import torch

from gpt2 import GPT2, GPT2Config


text = "The quick brown fox jumps over the lazy dog. "
prompt = "The quick"

# This is only a small runnable demo tokenizer. GPT-2 itself uses byte-level
# BPE; here each UTF-8 byte is one ID so the model can run without a tokenizer
# dependency.
def encode(value: str) -> list[int]:
    return list(value.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


torch.manual_seed(7)
config = GPT2Config(
    vocab_size=256,
    context_length=64,
    n_layer=2,
    n_head=4,
    n_embd=128,
)
model = GPT2(config)

text_ids = torch.tensor([encode(text)], dtype=torch.long)
inputs = text_ids[:, :-1]
targets = text_ids[:, 1:]
logits, loss = model(inputs, targets)

prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long)
generated_ids = model.generate(prompt_ids, max_new_tokens=32, top_k=20)

print("input text:", text)
print("input shape:", tuple(inputs.shape))
print("logits shape:", tuple(logits.shape))
print("next-token loss:", round(loss.item(), 4))
print("generated text:", decode(generated_ids[0].tolist()))
