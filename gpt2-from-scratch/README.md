# GPT-2 from scratch: dimensions and computations

This document follows one forward pass through [gpt2.py](gpt2.py). It explains
what each operation means, the shape of every tensor, and how one attention head
becomes many heads.

The runnable example is [1.py](1.py). It uses smaller dimensions, but every
operation is the same as GPT-2 small.

## 1. Configuration and notation

GPT-2 small uses:

| Symbol | Meaning | Value |
|---|---|---:|
| \(B\) | batch size | example: \(2\) |
| \(T\) | sequence length | example: \(8\), maximum \(1024\) |
| \(V\) | vocabulary size | \(50{,}257\) |
| \(C\) | residual-stream width | \(768\) |
| \(L\) | Transformer block count | \(12\) |
| \(H\) | attention-head count | \(12\) |
| \(d_h\) | width of one head | \(C/H = 64\) |
| \(F\) | MLP intermediate width | \(4C = 3{,}072\) |

The central invariant is:

\[
\text{residual stream shape}=(B,T,C).
\]

Attention temporarily exposes the head dimensions. The vocabulary projection
changes the final dimension from \(C\) to \(V\).

## 2. Complete data flow

~~~~text
text
  │ tokenizer
  ▼
token IDs                         (B, T)
  │ token and position lookup
  ▼
initial residual stream           (B, T, C)
  │
  ├── Transformer block 1 ──────── (B, T, C)
  ├── Transformer block 2 ──────── (B, T, C)
  │                 ...
  └── Transformer block 12 ─────── (B, T, C)
  │ final LayerNorm
  ▼
contextual representations        (B, T, C)
  │ vocabulary projection
  ▼
logits                            (B, T, V)
~~~~

For a concrete example, use \(B=2\) and \(T=8\). Every position starts as one
integer token ID and eventually becomes one vector of \(V\) vocabulary scores.

## 3. Token IDs and embeddings

A tokenizer maps text to integer IDs:

\[
\text{text}\longrightarrow
\text{token\_ids}\in\{0,\ldots,V-1\}^{B\times T}.
\]

The token table is:

\[
W_{\text{token}}\in\mathbb{R}^{V\times C}
=\mathbb{R}^{50{,}257\times768}.
\]

Indexing it gives:

\[
E_{\text{token}}\in\mathbb{R}^{B\times T\times C}
=\mathbb{R}^{2\times8\times768}.
\]

The position table is:

\[
W_{\text{pos}}\in\mathbb{R}^{1024\times768}.
\]

For \(T=8\), its lookup has shape \(\mathbb{R}^{8\times768}\) and broadcasts
across the batch. The initial residual stream is:

\[
X^{(0)}
=E_{\text{token}}+E_{\text{pos}}
\in\mathbb{R}^{B\times T\times C}.
\]

Elementwise:

\[
X^{(0)}_{b,t,:}
=W_{\text{token}}[\text{token\_ids}_{b,t},:]
+W_{\text{pos}}[t,:].
\]

The embedding parameter counts are:

\[
N_{\text{token}}=VC
=50{,}257\cdot768
=38{,}597{,}376,
\]

\[
N_{\text{pos}}=1024C
=1024\cdot768
=786{,}432.
\]

## 4. One pre-norm Transformer block

GPT-2 repeats this block \(L=12\) times:

~~~~text
                 ┌──────────────┐
                 │ LayerNorm    │
                 └──────┬───────┘
                        ▼
residual X ──────────── (+) ◄──── causal self-attention
                        │
                 ┌──────▼───────┐
                 │ LayerNorm    │
                 └──────┬───────┘
                        ▼
residual X ──────────── (+) ◄──── MLP
                        │
                        ▼
                 block output
~~~~

\[
X'=X+\operatorname{Attention}(\operatorname{LayerNorm}_1(X)),
\]

\[
X_{\text{block}}
=X'+\operatorname{MLP}(\operatorname{LayerNorm}_2(X')).
\]

Every residual addition has shape \(\mathbb{R}^{B\times T\times C}\).

LayerNorm normalizes the \(C\) channels of each token independently:

\[
\mu_{b,t}=\frac{1}{C}\sum_{c=1}^{C}x_{b,t,c},
\qquad
\sigma^2_{b,t}
=\frac{1}{C}\sum_{c=1}^{C}(x_{b,t,c}-\mu_{b,t})^2,
\]

\[
y_{b,t,c}
=\gamma_c\frac{x_{b,t,c}-\mu_{b,t}}
{\sqrt{\sigma^2_{b,t}+\varepsilon}}
+\beta_c.
\]

Here \(\gamma,\beta\in\mathbb{R}^{C}\), so each LayerNorm has \(2C=1{,}536\)
parameters.

## 5. Fused QKV projection

Let \(X_{\text{norm}}\in\mathbb{R}^{B\times T\times C}\). Conceptually:

\[
Q=X_{\text{norm}}W_Q+b_Q,\qquad
K=X_{\text{norm}}W_K+b_K,\qquad
V=X_{\text{norm}}W_V+b_V.
\]

For the fused implementation:

\[
W_{QKV}
=\begin{bmatrix}W_Q&W_K&W_V\end{bmatrix}
\in\mathbb{R}^{C\times3C},
\]

\[
QKV=X_{\text{norm}}W_{QKV}+b_{QKV}
\in\mathbb{R}^{B\times T\times3C}.
\]

Numerically:

\[
\mathbb{R}^{2\times8\times768}
\longrightarrow
\mathbb{R}^{2\times8\times2304}.
\]

Splitting the last axis gives:

\[
Q,K,V\in\mathbb{R}^{B\times T\times C}
=\mathbb{R}^{2\times8\times768}.
\]

PyTorch stores the linear weight as \((3C,C)\), but the row-vector
multiplication is written with \(W_{QKV}\) as \((C,3C)\).

The fused projection has:

\[
N_{QKV}=C(3C)+3C
=3C^2+3C
=768\cdot2304+2304
=1{,}771{,}776.
\]

## 6. Split into heads

The channel width is divided evenly:

\[
d_h=\frac{C}{H}
=\frac{768}{12}
=64.
\]

Each projected tensor is reshaped:

\[
\mathbb{R}^{B\times T\times C}
\longrightarrow
\mathbb{R}^{B\times T\times H\times d_h}
\longrightarrow
\mathbb{R}^{B\times H\times T\times d_h}.
\]

Numerically:

\[
\mathbb{R}^{2\times8\times768}
\longrightarrow
\mathbb{R}^{2\times8\times12\times64}
\longrightarrow
\mathbb{R}^{2\times12\times8\times64}.
\]

Thus:

\[
Q,K,V\in\mathbb{R}^{B\times H\times T\times d_h}.
\]

For a single head \(h\), the learned matrices would be:

\[
W_Q^{(h)},W_K^{(h)},W_V^{(h)}
\in\mathbb{R}^{C\times d_h}
=\mathbb{R}^{768\times64}.
\]

For one position:

\[
\underbrace{x_{b,t,:}}_{1\times768}
\underbrace{W_Q^{(h)}}_{768\times64}
=
\underbrace{q^{(h)}_{b,t,:}}_{1\times64}.
\]

The same \(768\to64\) multiplication creates \(k^{(h)}\) and \(v^{(h)}\).
For the complete batch, each has shape
\(\mathbb{R}^{B\times T\times d_h}=\mathbb{R}^{2\times8\times64}\).

## 7. One head: query-key scores

For one head and one batch item:

\[
Q^{(h)},K^{(h)},V^{(h)}
\in\mathbb{R}^{T\times d_h}
=\mathbb{R}^{8\times64}.
\]

Transpose the keys:

\[
(K^{(h)})^\top\in\mathbb{R}^{d_h\times T}
=\mathbb{R}^{64\times8}.
\]

The scores are:

\[
S^{(h)}
=Q^{(h)}(K^{(h)})^\top
\in\mathbb{R}^{T\times T},
\]

\[
\underbrace{(8\times64)}_{Q^{(h)}}
\underbrace{(64\times8)}_{(K^{(h)})^\top}
\longrightarrow
\underbrace{(8\times8)}_{S^{(h)}}.
\]

Each score is a 64-term dot product:

\[
S^{(h)}_{b,t,s}
=\sum_{i=1}^{64}Q^{(h)}_{b,t,i}K^{(h)}_{b,s,i}.
\]

The index \(t\) is the target position asking a question; \(s\) is the source
position offering a key; \(i\) is a channel inside the head.

## 8. Scale and causal mask

Scale the scores by the square root of the head width:

\[
\widetilde{S}^{(h)}
=\frac{S^{(h)}}{\sqrt{d_h}}
=\frac{S^{(h)}}{\sqrt{64}}
=\frac{S^{(h)}}{8}.
\]

The scale keeps the score variance from growing with \(d_h\).

For a four-token sequence, the causal mask is:

~~~~text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
~~~~

With \(M_{t,s}=0\) for allowed positions and \(M_{t,s}=-\infty\) for future
positions:

\[
A
=\operatorname{softmax}_{s}
\left(
\frac{QK^\top}{\sqrt{d_h}}+M
\right).
\]

The full shapes are:

\[
QK^\top\in\mathbb{R}^{B\times H\times T\times T},
\qquad
M\in\mathbb{R}^{1\times1\times T\times T},
\]

\[
A\in\mathbb{R}^{B\times H\times T\times T}
=\mathbb{R}^{2\times12\times8\times8}.
\]

The mask broadcasts over batch and head axes. Softmax is taken across source
positions \(s\), so each target row is a probability distribution over the
allowed history.

## 9. One head: weighted values

For one head:

\[
A^{(h)}\in\mathbb{R}^{B\times T\times T},
\qquad
V^{(h)}\in\mathbb{R}^{B\times T\times d_h}.
\]

The context output is:

\[
O^{(h)}=A^{(h)}V^{(h)}
\in\mathbb{R}^{B\times T\times d_h}.
\]

The contracted dimensions are both \(T\):

\[
\underbrace{(B\times T\times T)}_{A^{(h)}}
\underbrace{(B\times T\times d_h)}_{V^{(h)}}
\longrightarrow
\underbrace{(B\times T\times d_h)}_{O^{(h)}}.
\]

For \(B=2,T=8,d_h=64\):

\[
(2,8,8)\cdot(2,8,64)
\longrightarrow
(2,8,64).
\]

Elementwise:

\[
O^{(h)}_{b,t,:}
=\sum_{s=1}^{T}A^{(h)}_{b,t,s}V^{(h)}_{b,s,:}.
\]

The sum is over source positions; the 64 value channels remain.

## 10. All heads in parallel

The input to this parallel attention calculation is:

\[
X_{\text{norm}}\in\mathbb{R}^{B\times T\times C}
=\mathbb{R}^{2\times8\times768}.
\]

For one head \(h\), the three learned projection matrices are:

\[
W_Q^{(h)},W_K^{(h)},W_V^{(h)}
\in\mathbb{R}^{C\times d_h}
=\mathbb{R}^{768\times64},
\]

with biases:

\[
b_Q^{(h)},b_K^{(h)},b_V^{(h)}
\in\mathbb{R}^{d_h}
=\mathbb{R}^{64}.
\]

For one sequence in the batch, the three multiplications are:

\[
\underbrace{X_{\text{norm}}[b]}_{8\times768}
\underbrace{W_Q^{(h)}}_{768\times64}
\longrightarrow
\underbrace{Q^{(h)}[b]}_{8\times64},
\]

\[
\underbrace{X_{\text{norm}}[b]}_{8\times768}
\underbrace{W_K^{(h)}}_{768\times64}
\longrightarrow
\underbrace{K^{(h)}[b]}_{8\times64},
\]

\[
\underbrace{X_{\text{norm}}[b]}_{8\times768}
\underbrace{W_V^{(h)}}_{768\times64}
\longrightarrow
\underbrace{V^{(h)}[b]}_{8\times64}.
\]

Including the batch axis:

\[
Q^{(h)},K^{(h)},V^{(h)}
\in\mathbb{R}^{B\times T\times d_h}
=\mathbb{R}^{2\times8\times64}.
\]

The implementation fuses all three projections and all heads. Its mathematical
weight has shape:

\[
W_{QKV}\in\mathbb{R}^{C\times3C}
=\mathbb{R}^{768\times2304}.
\]

The fused multiplication is:

\[
\underbrace{X_{\text{norm}}}_{2\times8\times768}
\underbrace{W_{QKV}}_{768\times2304}
\longrightarrow
\underbrace{QKV}_{2\times8\times2304}.
\]

Splitting the final \(2304\) dimension into three \(768\)-wide sections gives
Q, K, and V. Reshaping each \(768\) into \(12\times64\) gives the batched head
tensors:

\[
Q\in\mathbb{R}^{B\times H\times T\times d_h},
\qquad
K^\top\in\mathbb{R}^{B\times H\times d_h\times T}.
\]

Numerically:

\[
Q,K,V\in\mathbb{R}^{2\times12\times8\times64},
\qquad
K^\top\in\mathbb{R}^{2\times12\times64\times8}.
\]

Therefore:

\[
QK^\top
\in\mathbb{R}^{B\times H\times T\times T}
=\mathbb{R}^{2\times12\times8\times8}.
\]

The multiplication contracts the \(64\)-wide head dimension:

\[
\underbrace{Q}_{2\times12\times8\times64}
\underbrace{K^\top}_{2\times12\times64\times8}
\longrightarrow
\underbrace{QK^\top}_{2\times12\times8\times8}.
\]

For every batch item \(b\), head \(h\), target position \(t\), and source
position \(s\):

\[
(QK^\top)_{b,h,t,s}
=\sum_{i=1}^{64}Q_{b,h,t,i}K_{b,h,s,i}.
\]

There are \(H=12\) independent \(8\times8\) score matrices per batch item.
The head axis makes them coexist in one tensor; heads do not multiply each
other's scores.

After value mixing:

\[
\underbrace{A}_{2\times12\times8\times8}
\underbrace{V}_{2\times12\times8\times64}
\longrightarrow
\underbrace{AV}_{2\times12\times8\times64}
\]

Equivalently, for one head:

\[
\underbrace{A^{(h)}}_{2\times8\times8}
\underbrace{V^{(h)}}_{2\times8\times64}
\longrightarrow
\underbrace{O^{(h)}}_{2\times8\times64}.
\]

For all heads:

\[
AV
\in\mathbb{R}^{B\times H\times T\times d_h}
=\mathbb{R}^{2\times12\times8\times64}.
\]

## 11. Concatenate heads and project

Move the head axis after time:

\[
\mathbb{R}^{B\times H\times T\times d_h}
\longrightarrow
\mathbb{R}^{B\times T\times H\times d_h}.
\]

Merge \(H\) and \(d_h\):

\[
\mathbb{R}^{B\times T\times H\times d_h}
\longrightarrow
\mathbb{R}^{B\times T\times(Hd_h)}.
\]

Since:

\[
Hd_h=12\cdot64=768=C,
\]

the concatenated output is:

\[
O_{\text{concat}}\in\mathbb{R}^{B\times T\times C}.
\]

The output projection mixes channels from different heads:

\[
Y_{\text{attn}}
=O_{\text{concat}}W_O+b_O,
\qquad
W_O\in\mathbb{R}^{C\times C},
\]

\[
Y_{\text{attn}}
\in\mathbb{R}^{B\times T\times C}.
\]

The residual update is:

\[
X'=X+Y_{\text{attn}}
\in\mathbb{R}^{B\times T\times C}.
\]

The attention parameter count is:

\[
N_{\text{attn}}
=\underbrace{C(3C)+3C}_{\text{fused QKV}}
+\underbrace{C^2+C}_{\text{output projection}}
=2{,}362{,}368.
\]

## 12. MLP

The MLP acts independently on each sequence position:

\[
\mathbb{R}^{B\times T\times C}
\longrightarrow
\mathbb{R}^{B\times T\times F}
\longrightarrow
\mathbb{R}^{B\times T\times F}
\longrightarrow
\mathbb{R}^{B\times T\times C}.
\]

The intermediate width is:

\[
F=4C=4\cdot768=3{,}072.
\]

The two projections are:

\[
H_{\text{mlp}}=X'W_{\text{fc}}+b_{\text{fc}}
\in\mathbb{R}^{B\times T\times F},
\]

\[
Y_{\text{mlp}}
=\operatorname{GELU}(H_{\text{mlp}})W_{\text{proj}}+b_{\text{proj}}
\in\mathbb{R}^{B\times T\times C}.
\]

The second residual update is:

\[
X_{\text{block}}=X'+Y_{\text{mlp}}
\in\mathbb{R}^{B\times T\times C}.
\]

The MLP parameter count is:

\[
N_{\text{MLP}}
=\underbrace{C(4C)+4C}_{\text{expansion}}
+\underbrace{(4C)C+C}_{\text{contraction}}
=4{,}722{,}432.
\]

## 13. Stack, final norm, and logits

For every block index \(\ell\):

\[
X^{(\ell)}\in\mathbb{R}^{B\times T\times C}.
\]

After \(L=12\) blocks:

\[
X^{(12)}\in\mathbb{R}^{B\times T\times C}.
\]

The final LayerNorm preserves this shape. The vocabulary head maps \(C\) to
\(V\):

\[
W_{\text{lm}}\in\mathbb{R}^{C\times V},
\]

\[
\text{logits}
=X^{(12)}W_{\text{lm}}+b_{\text{lm}}
\in\mathbb{R}^{B\times T\times V}.
\]

For GPT-2 small:

\[
\mathbb{R}^{2\times8\times768}
\longrightarrow
\mathbb{R}^{2\times8\times50{,}257}.
\]

The output weight is tied to the token embedding:

\[
W_{\text{lm}}=W_{\text{token}}^\top.
\]

The total unique parameter count is:

\[
N_{\text{total}}
=VC+1024C+L N_{\text{block}}+2C
=124{,}439{,}808.
\]

## 14. Loss and generation

For next-token training:

\[
[t_0,t_1,t_2,t_3]
\longrightarrow
\begin{cases}
\text{input}=[t_0,t_1,t_2],\\
\text{target}=[t_1,t_2,t_3].
\end{cases}
\]

The loss inputs are:

\[
\text{logits}\in\mathbb{R}^{B\times T\times V},
\qquad
\text{targets}\in\{0,\ldots,V-1\}^{B\times T}.
\]

Cross-entropy flattens them to:

\[
\text{logits}\in\mathbb{R}^{(BT)\times V},
\qquad
\text{targets}\in\{0,\ldots,V-1\}^{BT}.
\]

For generation, the final position produces:

\[
\text{logits}_{:,-1,:}\in\mathbb{R}^{B\times V}.
\]

Sampling one token gives:

\[
\text{next token}\in\{0,\ldots,V-1\}^{B\times1},
\]

and concatenation extends the sequence:

\[
(B,T)+(B,1)\longrightarrow(B,T+1).
\]

A KV cache would retain:

\[
K_{\text{cache}},V_{\text{cache}}
\in\mathbb{R}^{B\times H\times T_{\text{seen}}\times d_h}.
\]

## 15. The CPU demo

The demo uses:

\[
V=256,\qquad T_{\max}=64,\qquad L=2,\qquad H=4,\qquad C=128,
\]

so:

\[
d_h=\frac{128}{4}=32,
\qquad
F=4\cdot128=512.
\]

Its shape path is:

\[
(B,T)
\to(B,T,128)
\to(B,T,384)
\to(B,4,T,32)
\to(B,4,T,T)
\to(B,T,128)
\to(B,T,512)
\to(B,T,128)
\to(B,T,256).
\]

Run it with:

~~~~bash
python 1.py
~~~~

The model is randomly initialized, so generated text demonstrates the complete
data path but is not expected to be meaningful until training or pretrained
weights are added.
+

## 16. Fully worked calculation for one block

This section substitutes the GPT-2-small numbers into every attention operation.
Use:

\[
B=2,\qquad T=8,\qquad C=768,\qquad H=12,\qquad d_h=64.
\]

The input to block \(\ell\) is:

\[
X^{(\ell)}\in\mathbb{R}^{2\times8\times768}.
\]

It is not \(8+768\). The tensor has three axes:

\[
\underbrace{2}_{\text{batch examples}}
\times
\underbrace{8}_{\text{positions per example}}
\times
\underbrace{768}_{\text{channels per position}}.
\]

For one batch item alone, the matrix is \(8\times768\). For the complete
batch, it is \(2\times8\times768\).

### Dimension ledger before attention

The following table separates the object being multiplied, the weight, and the
result. The \(T\times C\) notation is a matrix shape; it is not an addition.

| object | mathematical shape | GPT-2-small shape |
|---|---|---|
| normalized input \(X_{\text{norm}}\) | \(B\times T\times C\) | \(2\times8\times768\) |
| one sequence \(X_{\text{norm}}[b]\) | \(T\times C\) | \(8\times768\) |
| one token row \(X_{\text{norm}}[b,t]\) | \(C\) | \(768\) |
| one-head \(W_Q^{(h)},W_K^{(h)},W_V^{(h)}\) | \(C\times d_h\) | \(768\times64\) |
| fused \(W_{QKV}\) | \(C\times3C\) | \(768\times2304\) |
| one-head \(Q^{(h)},K^{(h)},V^{(h)}\) | \(B\times T\times d_h\) | \(2\times8\times64\) |
| all-head \(Q,K,V\) after reshape | \(B\times H\times T\times d_h\) | \(2\times12\times8\times64\) |
| one-head scores | \(B\times T\times T\) | \(2\times8\times8\) |
| all-head scores | \(B\times H\times T\times T\) | \(2\times12\times8\times8\) |

For example, the one-head query operation is exactly:

$$
\underbrace{X_{\text{norm}}[b]}_{8\times768}
\underbrace{W_Q^{(h)}}_{768\times64}
\longrightarrow
\underbrace{Q^{(h)}[b]}_{8\times64}.
$$

The inner \(768\) dimensions are contracted. The remaining dimensions are the
eight sequence positions and the 64 channels belonging to that head.

### 16.1 First LayerNorm

LayerNorm is applied to each of the \(2\cdot8=16\) token vectors.

For one token:

\[
x_{b,t,:}\in\mathbb{R}^{768}
\longrightarrow
\operatorname{LayerNorm}(x_{b,t,:})
\in\mathbb{R}^{768}.
\]

For the batch:

\[
X^{(\ell)}
\in\mathbb{R}^{2\times8\times768}
\longrightarrow
X_{\text{norm}}
\in\mathbb{R}^{2\times8\times768}.
\]

Nothing is multiplied across positions here. The normalization reduces and
rescales the 768 channels belonging to each individual position.

### 16.2 Fused query-key-value multiplication

The mathematical fused weight has shape:

\[
W_{QKV}\in\mathbb{R}^{C\times3C}
=\mathbb{R}^{768\times2304}.
\]

The input and output multiplication is:

\[
\underbrace{X_{\text{norm}}}_{2\times8\times768}
\underbrace{W_{QKV}}_{768\times2304}
\longrightarrow
\underbrace{QKV}_{2\times8\times2304}.
\]

The contracted dimension is 768. For each of the \(2\cdot8=16\) positions,
a \(1\times768\) row is multiplied by a \(768\times2304\) matrix:

\[
\underbrace{(1\times768)}_{\text{one token}}
\underbrace{(768\times2304)}_{W_{QKV}}
\longrightarrow
\underbrace{(1\times2304)}_{\text{qkv for one token}}.
\]

The bias satisfies:

\[
b_{QKV}\in\mathbb{R}^{2304}
\]

and broadcasts over the first two axes.

The number of scalar multiplication terms in this batch matrix multiplication
is:

\[
B\cdot T\cdot C\cdot3C
=2\cdot8\cdot768\cdot2304
=28{,}311{,}552.
\]

The output is split into contiguous sections:

\[
QKV_{b,t,:}
=
\left[
Q_{b,t,:}\;
K_{b,t,:}\;
V_{b,t,:}
\right].
\]

Therefore:

\[
Q,K,V\in\mathbb{R}^{2\times8\times768}.
\]

The split does not calculate anything new. It only selects ranges of the final
axis:

\[
Q=QKV[:,:,0:768],
\]

\[
K=QKV[:,:,768:1536],
\]

\[
V=QKV[:,:,1536:2304].
\]

### 16.3 Separate the 12 heads

The channel axis of each projection has length 768:

\[
768=12\cdot64.
\]

Reshape Q:

\[
Q\in\mathbb{R}^{2\times8\times768}
\longrightarrow
\mathbb{R}^{2\times8\times12\times64}.
\]

The axes currently mean:

\[
(\text{batch},\text{time},\text{head},\text{channel within head}).
\]

Transpose time and head:

\[
\mathbb{R}^{2\times8\times12\times64}
\longrightarrow
\mathbb{R}^{2\times12\times8\times64}.
\]

Now the axes mean:

\[
(\text{batch},\text{head},\text{time},\text{channel within head}).
\]

Apply exactly the same reshape to K and V:

\[
Q,K,V\in\mathbb{R}^{2\times12\times8\times64}.
\]

For head \(h\), the slice is:

\[
Q^{(h)}=Q[:,h,:,:]\in\mathbb{R}^{2\times8\times64},
\]

\[
K^{(h)}=K[:,h,:,:]\in\mathbb{R}^{2\times8\times64},
\]

\[
V^{(h)}=V[:,h,:,:]\in\mathbb{R}^{2\times8\times64}.
\]

For one batch item, remove the batch axis:

\[
Q^{(h)},K^{(h)},V^{(h)}
\in\mathbb{R}^{8\times64}.
\]

### 16.4 One head: calculate every query-key match

For one head and one batch item:

\[
Q^{(h)}\in\mathbb{R}^{8\times64},
\qquad
K^{(h)}\in\mathbb{R}^{8\times64}.
\]

Transpose K:

\[
(K^{(h)})^\top\in\mathbb{R}^{64\times8}.
\]

Multiply:

\[
\underbrace{Q^{(h)}}_{8\times64}
\underbrace{(K^{(h)})^\top}_{64\times8}
=
\underbrace{S^{(h)}}_{8\times8}.
\]

The inner dimensions match because both are 64. The output has one score for
every ordered pair of target and source positions:

\[
8\cdot8=64\text{ scores}.
\]

One output element is:

\[
S^{(h)}_{t,s}
=\sum_{i=1}^{64}Q^{(h)}_{t,i}K^{(h)}_{s,i}.
\]

So \(S^{(h)}_{3,5}\) is one dot product between:

\[
Q^{(h)}_{3,:}\in\mathbb{R}^{64}
\quad\text{and}\quad
K^{(h)}_{5,:}\in\mathbb{R}^{64}.
\]

For both batch items, one head gives:

\[
Q^{(h)}\in\mathbb{R}^{2\times8\times64},
\]

\[
(K^{(h)})^\top\in\mathbb{R}^{2\times64\times8},
\]

\[
S^{(h)}\in\mathbb{R}^{2\times8\times8}.
\]

For all 12 heads at once:

\[
Q\in\mathbb{R}^{2\times12\times8\times64},
\]

\[
K^\top\in\mathbb{R}^{2\times12\times64\times8},
\]

\[
S=QK^\top
\in\mathbb{R}^{2\times12\times8\times8}.
\]

The head axis is independent: this is 12 separate \(8\times8\) matrices for each
batch item, not one \(96\times96\) matrix.

### 16.5 Scale the score matrix

Each score is a sum of 64 products. As the number of summed terms grows, the
score variance grows. Scale by:

\[
\sqrt{d_h}=\sqrt{64}=8.
\]

Thus:

\[
\widetilde{S}
=\frac{S}{\sqrt{d_h}}
=\frac{S}{8}
\in\mathbb{R}^{2\times12\times8\times8}.
\]

The division is elementwise. It does not change any dimension.

If we did not divide by 8, the logits entering softmax could become too large.
Softmax would become almost one-hot, and most positions would receive almost no
gradient. The square root compensates for the 64-term dot product.

### 16.6 Apply the causal mask

For \(T=8\), the conceptual mask is:

\[
M_{t,s}
=
\begin{cases}
0,&s\le t,\\
-\infty,&s>t.
\end{cases}
\]

Its shape for one sequence is:

\[
M\in\mathbb{R}^{8\times8}.
\]

The implementation stores it with singleton axes so it broadcasts:

\[
M\in\mathbb{R}^{1\times1\times8\times8}.
\]

The score tensor is:

\[
\widetilde{S}\in\mathbb{R}^{2\times12\times8\times8}.
\]

Adding the mask produces:

\[
\widehat{S}=\widetilde{S}+M
\in\mathbb{R}^{2\times12\times8\times8}.
\]

The broadcast means the same lower-triangular pattern is applied to every
batch item and every head.

For one head, the \(8\times8\) pattern is:

\[
\begin{bmatrix}
s_{0,0}&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty\\
s_{1,0}&s_{1,1}&-\infty&-\infty&-\infty&-\infty&-\infty&-\infty\\
s_{2,0}&s_{2,1}&s_{2,2}&-\infty&-\infty&-\infty&-\infty&-\infty\\
\vdots&\vdots&\vdots&\ddots&\ddots&\vdots&\vdots&\vdots\\
s_{7,0}&s_{7,1}&s_{7,2}&\cdots&s_{7,6}&s_{7,7}
\end{bmatrix}.
\]

The actual code uses the smallest representable floating-point value instead
of literal \(-\infty\), which has the same softmax effect.

### 16.7 Softmax over source positions

Softmax is applied along the last axis, the source position \(s\):

\[
A_{b,h,t,s}
=
\frac{\exp(\widehat{S}_{b,h,t,s})}
{\displaystyle\sum_{j=0}^{T-1}
\exp(\widehat{S}_{b,h,t,j})}.
\]

Therefore:

\[
A\in\mathbb{R}^{2\times12\times8\times8}.
\]

For every fixed \((b,h,t)\):

\[
\sum_{s=0}^{7}A_{b,h,t,s}=1.
\]

Because future scores were masked:

\[
A_{b,h,t,s}=0\qquad\text{when }s>t.
\]

For target position \(t=3\), only source positions \(0,1,2,3\) can receive
probability.

### 16.8 Multiply attention weights by values

For one head:

\[
A^{(h)}\in\mathbb{R}^{2\times8\times8},
\qquad
V^{(h)}\in\mathbb{R}^{2\times8\times64}.
\]

The source-position dimension contracts:

\[
\underbrace{(2\times8\times8)}_{A^{(h)}}
\underbrace{(2\times8\times64)}_{V^{(h)}}
\longrightarrow
\underbrace{(2\times8\times64)}_{O^{(h)}}.
\]

For one batch item and one target position:

\[
\underbrace{(1\times8)}_{\text{weights for the 8 sources}}
\underbrace{(8\times64)}_{\text{8 source value vectors}}
\longrightarrow
\underbrace{(1\times64)}_{\text{one context vector}}.
\]

Elementwise:

\[
O^{(h)}_{b,t,:}
=\sum_{s=0}^{7}A^{(h)}_{b,t,s}V^{(h)}_{b,s,:}.
\]

Thus one head returns:

\[
O^{(h)}\in\mathbb{R}^{2\times8\times64}.
\]

All heads together return:

\[
O=AV
\in\mathbb{R}^{2\times12\times8\times64}.
\]

### 16.9 Combine the 12 head outputs

The current layout is:

\[
O\in\mathbb{R}^{2\times12\times8\times64}.
\]

Transpose the head and time axes:

\[
\mathbb{R}^{2\times12\times8\times64}
\longrightarrow
\mathbb{R}^{2\times8\times12\times64}.
\]

Merge the last two axes:

\[
\mathbb{R}^{2\times8\times12\times64}
\longrightarrow
\mathbb{R}^{2\times8\times(12\cdot64)}
=
\mathbb{R}^{2\times8\times768}.
\]

No information is averaged away. The 12 head vectors are placed side by side:

\[
O_{\text{concat},b,t,:}
=
\left[
O^{(0)}_{b,t,:}\;
O^{(1)}_{b,t,:}\;
\cdots\;
O^{(11)}_{b,t,:}
\right].
\]

### 16.10 Output projection and residual addition

The output projection has mathematical weight:

\[
W_O\in\mathbb{R}^{768\times768}.
\]

The multiplication is:

\[
\underbrace{O_{\text{concat}}}_{2\times8\times768}
\underbrace{W_O}_{768\times768}
\longrightarrow
\underbrace{Y_{\text{attn}}}_{2\times8\times768}.
\]

The number of scalar multiplication terms is:

\[
B\cdot T\cdot C^2
=2\cdot8\cdot768^2
=9{,}437{,}184.
\]

The residual update is elementwise:

\[
X'=X+Y_{\text{attn}}
\in\mathbb{R}^{2\times8\times768}.
\]

At this point the attention sub-block is complete.

### 16.11 MLP calculation in the same block

The second LayerNorm preserves:

\[
X'\in\mathbb{R}^{2\times8\times768}
\longrightarrow
X'_{\text{norm}}\in\mathbb{R}^{2\times8\times768}.
\]

The expansion matrix is:

\[
W_{\text{fc}}\in\mathbb{R}^{768\times3072}.
\]

Therefore:

\[
\underbrace{X'_{\text{norm}}}_{2\times8\times768}
\underbrace{W_{\text{fc}}}_{768\times3072}
\longrightarrow
\underbrace{H_{\text{mlp}}}_{2\times8\times3072}.
\]

The multiplication terms are:

\[
2\cdot8\cdot768\cdot3072
=37{,}748{,}736.
\]

GELU is applied to all \(2\cdot8\cdot3072\) elements independently:

\[
\operatorname{GELU}(H_{\text{mlp}})
\in\mathbb{R}^{2\times8\times3072}.
\]

The contraction matrix is:

\[
W_{\text{proj}}\in\mathbb{R}^{3072\times768}.
\]

Thus:

\[
\underbrace{\operatorname{GELU}(H_{\text{mlp}})}_{2\times8\times3072}
\underbrace{W_{\text{proj}}}_{3072\times768}
\longrightarrow
\underbrace{Y_{\text{mlp}}}_{2\times8\times768}.
\]

The contraction multiplication terms are:

\[
2\cdot8\cdot3072\cdot768
=37{,}748{,}736.
\]

Finally:

\[
X_{\text{block}}
=X'+Y_{\text{mlp}}
\in\mathbb{R}^{2\times8\times768}.
\]

This is the complete output of one Transformer block.

## 17. What happens in blocks 2 through 12?

Block 2 receives exactly the shape that block 1 returned:

\[
X^{(1)}\in\mathbb{R}^{2\times8\times768}.
\]

It performs the same sequence:

\[
\mathbb{R}^{2\times8\times768}
\to
\mathbb{R}^{2\times8\times768}
\]

but uses different learned parameters. The same is true for every block:

\[
X^{(0)}
\xrightarrow{\text{block 1}}
X^{(1)}
\xrightarrow{\text{block 2}}
\cdots
\xrightarrow{\text{block 12}}
X^{(12)}.
\]

The dimensions never change between blocks. Only the learned transformation
changes.

One block has:

\[
N_{\text{block}}
=
\underbrace{2{,}362{,}368}_{\text{attention}}
+
\underbrace{4{,}722{,}432}_{\text{MLP}}
+
\underbrace{2(1{,}536)}_{\text{two LayerNorms}}
=
7{,}087{,}872.
\]

The 12 blocks therefore contain:

\[
12\cdot7{,}087{,}872
=
85{,}054{,}464
\]

parameters.

## 18. Final vocabulary calculation

After block 12 and the final LayerNorm:

\[
X^{(12)}_{\text{norm}}
\in\mathbb{R}^{2\times8\times768}.
\]

The vocabulary projection has mathematical weight:

\[
W_{\text{lm}}\in\mathbb{R}^{768\times50{,}257}.
\]

Therefore:

\[
\underbrace{X^{(12)}_{\text{norm}}}_{2\times8\times768}
\underbrace{W_{\text{lm}}}_{768\times50{,}257}
\longrightarrow
\underbrace{\text{logits}}_{2\times8\times50{,}257}.
\]

At each of the \(2\cdot8=16\) positions, the model produces 50,257 scores.

GPT-2 ties this matrix to the token embedding:

\[
W_{\text{lm}}=W_{\text{token}}^\top.
\]

So this projection does not allocate another 38,597,376 unique parameters.

## 19. Final shape trace for one block

For \(B=2,T=8,C=768,H=12,d_h=64,F=3072\):

\[
\begin{aligned}
X
&: (2,8,768)\\
\operatorname{LayerNorm}(X)
&: (2,8,768)\\
QKV
&: (2,8,2304)\\
Q,K,V
&: 3\times(2,8,768)\\
\text{split heads}
&: 3\times(2,12,8,64)\\
QK^\top
&: (2,12,8,8)\\
\operatorname{softmax}
&: (2,12,8,8)\\
AV
&: (2,12,8,64)\\
\text{concatenate heads}
&: (2,8,768)\\
\text{attention residual}
&: (2,8,768)\\
\text{MLP expansion}
&: (2,8,3072)\\
\operatorname{GELU}
&: (2,8,3072)\\
\text{MLP contraction}
&: (2,8,768)\\
\text{block output}
&: (2,8,768).
\end{aligned}
\]

The next block starts again at the final line.
