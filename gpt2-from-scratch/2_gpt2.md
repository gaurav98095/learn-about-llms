# From Encoder–Decoder to Decoder-Only: The Architecture of GPT-2

> *A continuation of "Attention Is All You Need" (Vaswani et al., 2017), written for a reader who already understands the original Transformer down to its gradients.*

---

## 0. Where we left off

In 2017 the Transformer was presented as a **sequence transduction** machine: a stack of $N=6$ encoder blocks that ingest a source sequence into a set of contextual representations, and a parallel stack of $N=6$ decoder blocks that generate the target sequence one token at a time, attending back into the encoder's output through **cross-attention**. The whole apparatus was justified by one idea — that attention alone, with no recurrence and no convolution, suffices to model dependencies of arbitrary length.

The 2017 design carries three distinct attention patterns:

1. **Encoder self-attention** — bidirectional, every source token sees every other source token.
2. **Decoder masked self-attention** — causal, each target token sees only itself and its predecessors.
3. **Encoder–decoder cross-attention** — each target position queries the entire encoded source.

GPT-2 (Radford et al., 2019, *"Language Models are Unsupervised Multitask Learners"*) makes a radical simplification: **it keeps only the second pattern**. The encoder is deleted. Cross-attention is deleted. What remains is a single tower of masked-self-attention blocks operating on one undivided stream of tokens. Everything in this document follows from that single decision and the engineering needed to make a *deep* stack of such blocks train stably.

This is not a small edit to the 2017 paper. It is a change of philosophy: from **transduction** (map sequence A to sequence B) to **autoregressive density estimation** (model the joint distribution of one sequence and let conditioning fall out of it).

---

## 1. The conceptual pivot: why drop half the Transformer?

### 1.1 The encoder–decoder split encodes an assumption

The 2017 architecture assumes the world comes pre-divided into a *source* and a *target* with different roles. The encoder gets the luxury of bidirectional context because the source is fully known up front; the decoder is constrained to be causal because the target is being produced. Cross-attention is the bridge between these two regimes.

This is the right inductive bias for machine translation, where the input/output boundary is real and fixed. It is the *wrong* bias if your goal is a general model of text, because most text is not a clean (input, output) pair. A paragraph, a dialogue, a piece of code — these are just sequences. The source/target distinction is an artifact you would have to impose.

### 1.2 Collapse the two streams into one

GPT-2's claim is that a single causal stream is enough to express *any* conditional task. If you want $P(\text{answer} \mid \text{question})$, you do not need a separate encoder for the question. You concatenate:

$$
\underbrace{w_1, w_2, \dots, w_k}_{\text{question (context)}}, \underbrace{w_{k+1}, \dots, w_T}_{\text{answer (continuation)}}
$$

and train one model to predict each token from its left context. Conditioning on the question is just *attending to the earlier part of the same sequence*. Cross-attention was only ever a special case of self-attention with an artificial wall between the two halves; remove the wall and self-attention subsumes it.

This is why the GPT-2 paper is titled around *unsupervised multitask learning*: by modeling $P(\text{sequence})$ over a sufficiently diverse corpus, the model is implicitly trained on translation, QA, summarization, and so on — every time those tasks happen to appear, naturally formatted, in the data. The architecture does not know about tasks. It knows about next tokens.

### 1.3 The objective that replaces seq2seq

The 2017 model maximized $P(\text{target} \mid \text{source})$. GPT-2 maximizes the plain **autoregressive likelihood** of the corpus, factorized by the chain rule:

$$
\mathcal{L}(\theta) = \sum_{t=1}^{T} \log P_\theta\!\left(w_t \mid w_1, \dots, w_{t-1}\right)
$$

There is exactly one task — *predict the next token* — applied at every position simultaneously. The causal mask is what makes "every position simultaneously" honest: it guarantees position $t$ never sees $w_{\ge t}$, so a single forward pass yields $T$ independent supervised predictions without leakage. This is the same trick the 2017 decoder used for training-time parallelism, now elevated to the entire model's reason for being.

### 1.4 Why not the other two doors? (encoder-only and encoder–decoder)

It is worth being explicit that decoder-only was a *choice against two live alternatives*, not a default. By 2018–2019 the 2017 Transformer had already branched into three families, and each was a real, well-supported option. Understanding why GPT-2 picked the third is most of the conceptual content of this chapter.

**Door 1 — Encoder-only (BERT, 2018).** Keep only the encoder: fully **bidirectional** self-attention, where every token sees every other in both directions. This is excellent for *understanding* tasks — classification, named-entity tagging, retrieval embeddings — precisely because a token's representation may draw on both its left and right context at once. But it is structurally unable to *generate*: with no causal direction there is no well-defined "next token" to sample, so you cannot roll a sequence out left-to-right. Its pretraining objective, **masked language modeling** (predict a few blanked-out tokens), also pays a steep efficiency price — only the ~15% of positions that are masked contribute to the loss on each pass, so the supervision extracted per token of compute is a fraction of what next-token prediction gets, where *every* position is a label (Section 1.3). BERT is the right tool if the end goal is to *read*; it is the wrong tool if the goal is to *write*.

**Door 2 — Encoder–decoder (the 2017 original, T5, BART).** Keep both stacks. This is genuinely the best fit when the task is an **explicit input→output map with a hard boundary**: translate *this* sentence, summarize *this* document, transcribe *this* audio. But as the foundation for a *general* model of text it carries two costs. First, it bakes in the source/target split that Section 1.1 argued is artificial for most text — raw web pages, code, and dialogue are not (input, output) pairs, so you would have to manufacture the boundary that the architecture insists on. Second, it is *two* stacks plus a cross-attention bridge: more moving parts to balance, parameters divided between encoder and decoder, and a more delicate object to scale than a single homogeneous tower (the uniformity argument of Section 15). You are paying for machinery whose central assumption your data does not actually satisfy.

**Door 3 — Decoder-only (GPT).** Keep only the causal decoder. You *give up* bidirectional encoding of a fixed input — a real cost on pure-understanding benchmarks — and in exchange you get a model that is **generation-native, supervised at every token, free of any task-specific boundary, and built from one repeated primitive**. For the specific bet GPT-2 is making — that a single model trained on enough raw text will absorb every task that happens to appear in that text (Section 1.2) — those four properties are exactly the ones that matter, and the thing you gave up (bidirectionality) is exactly the thing that bet does not need.

> **The trade in one line.** Encoder-only sees everything but cannot speak; encoder–decoder speaks fluently but only across a boundary you must impose; decoder-only speaks, sees only the past, and treats *everything* as one stream — which turns out to be all you need when "the task" is just "model the text."

### 1.5 The deeper reason: text is its own supervision

Underneath the architectural comparison sits an economic one, and it is arguably the real driver. The scarce resource in 2019 was not compute — it was **labeled** data. Encoder–decoder translation models need parallel corpora; supervised classifiers need human annotations; even BERT's fine-tuning stage needs labeled task data. Next-token prediction needs **none of it**: every naturally occurring document is, for free, a stream of billions of (left-context → next-token) training pairs. Decoder-only language modeling is the architecture that most directly converts the one genuinely abundant resource on the internet — raw, unlabeled text — into supervision, with no annotation step in between. The objective *is* self-supervised by construction.

That single fact has two compounding consequences the GPT line would ride for years:

- **Density of signal.** A length-$T$ sequence yields $T$ supervised predictions in one forward pass (Section 1.3) — versus a single target sequence for seq2seq, or only ~15% of positions for masked language modeling. More gradient extracted per token of data and per FLOP of compute.
- **Conditioning is free, so *tasks* are free.** Because context and continuation share one causal stream (Section 1.2), any task you can phrase as "given this text, produce that text" is *already in-distribution* — no new head, no architectural change, in principle no fine-tuning. This is the seed of the **in-context / few-shot learning** that GPT-3 would later make famous, and it is available *only* because the model was built as an unconditional density estimator $P(\text{sequence})$ rather than a conditional input→output map $P(\text{target}\mid\text{source})$.

So the choice was already well-motivated on grounds of **supervision economics** and **generation-nativeness** *before* any large-scale evidence existed. The fuller retrospective — why decoder-only kept winning specifically as models got *bigger* (uniformity as a clean scaling primitive, the well-behaved residual stream) — is collected in Section 15; this section is the argument as it could have been made on the whiteboard in 2018, looking only at what kind of data the world actually has and what you want the model to *do* with it.

---

## 2. The architecture at a glance

A GPT-2 forward pass, top to bottom — the **entire model in one diagram**. The embedding stage and the head are drawn once; the decoder block is drawn once *with its internals expanded* and stacked `× N`. Read the two `(+)` nodes inside the block as the two places the residual stream is written to, and the `│` running down the **left edge of the block** as the residual stream itself — the skip path that is *never* normalized (Sections 4 and 5).

```
              token ids  [the, cat, sat] = [1, 2, 3]            (T,)
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                         ▼
        token embedding                          positional embedding
          W_e[ids]   (T, d)                        W_p[0:T]   (T, d)         ← learned, not sinusoidal
              └───────────────────┬───────────────────┘
                                  ▼
                       x₀ = W_e[ids] + W_p[0:T]      (T, d)   ◄── residual stream begins
                                  │
   ╔══════════════════════════════╪══════════════════════════════════════════╗
   ║  DECODER BLOCK                │                              repeat × N   ║
   ║                               │                                          ║
   ║   residual stream ───────────►├──────────────┐                           ║
   ║                               ▼              │ skip (identity, unnormed)  ║
   ║                          LayerNorm₁          │                           ║
   ║                               ▼              │                           ║
   ║              Masked Multi-Head Self-Attention│   ← causal mask:           ║
   ║                 (token t attends to ≤ t only)│     no peeking ahead       ║
   ║                               ▼              │                           ║
   ║                              (+)◄────────────┘   a = x + MHSA(LN₁(x))     ║
   ║                               │                                          ║
   ║                               ├──────────────┐                           ║
   ║                               ▼              │ skip (identity, unnormed)  ║
   ║                          LayerNorm₂          │                           ║
   ║                               ▼              │                           ║
   ║              Feed-Forward (GELU),  d → 4d → d │                           ║
   ║                               ▼              │                           ║
   ║                              (+)◄────────────┘   x' = a + FFN(LN₂(a))     ║
   ║                               │                                          ║
   ╚═══════════════════════════════╪══════════════════════════════════════════╝
                                  ▼   x_N   (T, d)
                          final LayerNorm  (ln_f)          ◄── new in GPT-2 (Section 5.3)
                                  ▼
                  unembed:  logits = x_N · W_eᵀ   (T, |V|)  ◄── weight-tied to W_e (Section 9)
                                  ▼
                              softmax (row-wise)
                                  ▼
                   P(next token | left context)   at every position t
```

The single most important thing to read off this picture: each **Decoder Block** is a **Pre-LN residual block** with exactly two sub-layers, and there are **two independent residual connections** — one wrapping attention, one wrapping the FFN. Each LayerNorm sits *inside* its branch (on the path feeding the sub-layer), **never on the skip path**; the skip carries an unmodified copy of the stream straight into the `(+)`. That clean, never-normalized vertical line down the block is the residual stream of Section 4, and keeping it clean is the whole point of Pre-LN (Section 5) and the $1/\sqrt N$ init (Section 8).

Two things visually distinguish this from the **2017 decoder block**: **LayerNorm sits *before* each sub-layer rather than after** (Pre-LN, Section 5), and there is **no third sub-layer for cross-attention** (Section 6). The 2017 decoder block had *three* sub-layers stacked — masked self-attention, then encoder–decoder cross-attention, then FFN, each followed by a Post-LN — whereas GPT-2's block has only the first and last of those, each preceded by a Pre-LN.

One conceptual split worth burning in before the zoom-ins: of the block's two sub-layers, **only attention moves information *between* positions** — it is the sole place where token $t$'s vector is influenced by token $s\neq t$. The **FFN is applied independently to each position** (same weights, no cross-token interaction). So a decoder block does exactly two things in sequence: *mix across time* (causally), then *transform each position in place*. Everything else — the LayerNorms, the residual adds — is plumbing that keeps those two operations trainable at depth.

### 2.1 Zoom-in A: inside "Masked Multi-Head Self-Attention"

The overview draws attention as one box. It is actually a six-step pipeline. Input is the normalized stream $\tilde x = \mathrm{LN}_1(x)$, shape $(T,d)$; output is what gets added back at `(+)` #1. (Full treatment in Section 6; this is the shape-annotated map.)

```
        x̃ = LayerNorm₁(x)        (T, d)
              │
   c_attn:  x̃ · W_qkv            W_qkv is (d, 3d)  ── ONE fused matmul, then split
              │
       split columns into 3
   ┌──────────┼──────────┐
   ▼          ▼          ▼
  Q (T,d)   K (T,d)   V (T,d)
   │          │          │
   └ reshape each to heads:  (T, d) ──► (h, T, d_k),   d_k = d/h   (=64 for every GPT-2 size)
              │
              │     ── for each head i = 1 … h, independently: ──
              ▼
   scores_i = Q_i · K_iᵀ / √d_k                         (T, T)     similarity of every query to every key
              │
   + causal mask M     (M_ij = 0 if j ≤ i,  −∞ if j > i)  (T, T)   ◄── the load-bearing piece
              │
   softmax over each row                                 (T, T)     row t = how token t weights tokens ≤ t
              │
   head_i = (softmax scores) · V_i                       (T, d_k)   weighted sum of values
              │
   concat(head_1, …, head_h)                             (T, d)     glue heads back to width d
              │
   c_proj:  · W_o                W_o is (d, d)            (T, d)   ◄── 1/√N-scaled residual-write (Section 8)
              ▼
        MHSA(x̃)                  (T, d)   ──────────────► added to the stream at (+) #1
```

Read the two `(T,T)` matrices as the heart of it: `scores` asks "how much should each token care about each other token," the mask zeroes out everything to the right (no token sees its future), and softmax turns each surviving row into a probability distribution over the past. The `(h, T, d_k)` split is why it is *multi-head*: $h$ of these little attention problems run in parallel on disjoint $d_k$-slices, each free to specialize (one head tracks the previous token, another long-range agreement, etc.), and `c_proj` merges their verdicts.

### 2.2 Zoom-in B: inside "Feed-Forward (GELU)"

The FFN box is simpler — two linear layers with a nonlinearity between, applied to each position on its own. Input is $\hat a = \mathrm{LN}_2(a)$, shape $(T,d)$. (Full treatment in Section 7.)

```
        â = LayerNorm₂(a)         (T, d)
              │
   c_fc:   â · W₁ + b₁            W₁ is (d, 4d)            (T, 4d)   ◄── expand to 4× width
              │
   GELU (tanh approximation)                              (T, 4d)   smooth gate, ≠ ReLU
              │
   c_proj: · W₂ + b₂             W₂ is (4d, d)            (T, d)    ◄── contract; 1/√N-scaled write (Section 8)
              ▼
        FFN(â)                    (T, d)   ──────────────► added to the stream at (+) #2
```

Note both sub-layers end in a module literally named `c_proj` — these are the **two residual-write matrices** ($W_o$ and $W_2$) that receive the $1/\sqrt N$ initialization of Section 8, because they are the only two paths that *add into* the stream.

### 2.3 The boxes mapped to real GPT-2 weights

Every box above is a named module in the released checkpoints (HuggingFace / nanoGPT names shown). Shapes are for **GPT-2 small** ($d=768$, $d_{ff}=3072$, $|V|=50257$, $n_{ctx}=1024$):

| Box in the diagram | Module name | Weight(s) | Shape (small) | $1/\sqrt N$ init? |
|---|---|---|---|---|
| token embedding | `wte` | $W_e$ | $50257\times768$ | no |
| positional embedding | `wpe` | $W_p$ | $1024\times768$ | no |
| LayerNorm₁ | `ln_1` | $\gamma,\beta$ | $768$ each | — |
| fused QKV | `attn.c_attn` | $W_{qkv},b$ | $768\times2304$ | no |
| attention output proj | `attn.c_proj` | $W_o,b$ | $768\times768$ | **yes** |
| LayerNorm₂ | `ln_2` | $\gamma,\beta$ | $768$ each | — |
| FFN up | `mlp.c_fc` | $W_1,b$ | $768\times3072$ | no |
| FFN down | `mlp.c_proj` | $W_2,b$ | $3072\times768$ | **yes** |
| final LayerNorm | `ln_f` | $\gamma,\beta$ | $768$ each | — |
| unembed (LM head) | *(tied to `wte`)* | $W_e^\top$ | — (shared) | — |

*(Dropout, omitted from the data-path diagrams as a training-only regularizer, sits in exactly three places in GPT-2: once on the embedding sum $x_0$, once on the softmax attention weights inside Section 2.1, and once on each sub-layer's `c_proj` output just **before** it is added back at the `(+)`. All three are identity at inference.)*

---

## 3. Input representation

### 3.1 Token embeddings

A learned matrix $W_e \in \mathbb{R}^{|V| \times d}$ maps each token id to a $d$-dimensional vector. Nothing new versus 2017 here except the vocabulary, which is worth a paragraph because it shapes $W_e$.

### 3.2 Byte-level BPE (and why it matters architecturally)

GPT-2 tokenizes with **byte-level Byte-Pair Encoding (BPE)**. To understand the architecture you do not strictly need the tokenizer's internals, but the *shape* of $W_e$ — and the model's freedom from `<UNK>` — falls directly out of how BPE works, so it is worth doing properly.

#### What BPE is

BPE began life (Gage, 1994) as a **data-compression** scheme and was repurposed for NLP tokenization by Sennrich et al. (2016). The idea is to find a vocabulary that sits *between* two bad extremes:

- **Word-level tokens** give short sequences but an unbounded vocabulary, and any word not seen in training becomes `<UNK>` — information is destroyed.
- **Character- (or byte-) level tokens** never have an out-of-vocabulary problem, but sequences become very long and each token carries little meaning, so the model must spend capacity reassembling words from characters.

BPE interpolates: start from the base units (characters/bytes) and **greedily glue together the most frequent adjacent pair, over and over**, until you have as many merged symbols as your vocabulary budget allows. Frequent words end up as a single token; rare words gracefully fall back to a few sub-word pieces; truly novel strings fall back to base units. Nothing is ever unrepresentable.

#### The training algorithm (learning the merges)

BPE "training" is just learning an **ordered list of merge rules**. The procedure:

1. **Initialize** the vocabulary with every base symbol (for GPT-2: the 256 bytes). Represent each word in the corpus as a sequence of base symbols, and keep a count of how often each word occurs.
2. **Count** every adjacent symbol pair across the corpus, weighted by word frequency.
3. **Merge** the single most frequent pair: add the merged symbol to the vocabulary, record the rule `(A, B) → AB`, and rewrite every occurrence of that adjacent pair as the new symbol.
4. **Repeat** steps 2–3 until the vocabulary reaches the target size (for GPT-2: until 50 000 merges have been learned).

The output is (a) the final vocabulary and (b) the **ordered** list of merge rules. The order matters — it is replayed at encoding time.

#### Worked example: learning merges on a tiny corpus

Take a corpus of five distinct words with these counts (the units here are *characters*, to keep it readable; the byte-level twist comes after):

| word | frequency | as symbols |
|---|---|---|
| `hug`  | 10 | `h u g` |
| `pug`  |  5 | `p u g` |
| `pun`  | 12 | `p u n` |
| `bun`  |  4 | `b u n` |
| `hugs` |  5 | `h u g s` |

Base vocabulary: `{b, g, h, n, p, s, u}` (7 symbols).

**Round 1 — count adjacent pairs** (each pair's count is the sum of the frequencies of the words it appears in):

| pair | where it occurs | count |
|---|---|---|
| `(u, g)` | hug(10), pug(5), hugs(5) | **20** |
| `(p, u)` | pug(5), pun(12) | 17 |
| `(u, n)` | pun(12), bun(4) | 16 |
| `(h, u)` | hug(10), hugs(5) | 15 |
| `(g, s)` | hugs(5) | 5 |
| `(b, u)` | bun(4) | 4 |

Winner: `(u, g)` at 20. **Merge 1: `(u, g) → ug`.** Rewrite the corpus:

`h ug` (10), `p ug` (5), `p u n` (12), `b u n` (4), `h ug s` (5)

**Round 2 — recount:**

| pair | count |
|---|---|
| `(u, n)` | pun(12)+bun(4) = **16** |
| `(h, ug)` | hug(10)+hugs(5) = 15 |
| `(p, u)` | pun(12) = 12 |
| `(p, ug)` | pug(5) = 5 |
| `(ug, s)` | hugs(5) = 5 |

Winner: `(u, n)` at 16. **Merge 2: `(u, n) → un`.** Corpus:

`h ug` (10), `p ug` (5), `p un` (12), `b un` (4), `h ug s` (5)

**Round 3 — recount:**

| pair | count |
|---|---|
| `(h, ug)` | hug(10)+hugs(5) = **15** |
| `(p, un)` | pun(12) = 12 |
| `(p, ug)` | 5 |
| `(b, un)` | 4 |
| `(ug, s)` | 5 |

Winner: `(h, ug)` at 15. **Merge 3: `(h, ug) → hug`.** Corpus:

`hug` (10), `p ug` (5), `p un` (12), `b un` (4), `hug s` (5)

If we stop after three merges, the learned artifacts are:

```
merge rules (ordered):     vocabulary:
  1.  (u, g)  → ug           b g h n p s u   ug  un  hug
  2.  (u, n)  → un
  3.  (h, ug) → hug
```

Notice the rules are *frequency-driven*, not linguistic: `ug` and `un` became tokens because they were common substrings, and `hug` was promoted to a whole-word token because it appeared 15 times. This is exactly how real GPT-2 ends up with single tokens for common words like ` the` and ` and`, while a rare word is left in pieces.

#### Encoding new text (applying the merges)

To tokenize a *new* string, split it into base symbols and then **apply the learned merge rules greedily, in the order they were learned**, until none apply:

- `bug` → `b u g` → apply rule 1 `(u,g)→ug` → `b ug` → rules 2, 3 don't match → **`["b", "ug"]`** (2 tokens).
- `hug` → `h u g` → rule 1 → `h ug` → rule 3 `(h,ug)→hug` → **`["hug"]`** (1 token — it learned the whole word).
- `hugs` → `h u g s` → rule 1 → `h ug s` → rule 3 → `hug s` → **`["hug", "s"]`**.

A word the corpus never saw still encodes fine — it just lands as more pieces. There is no `<UNK>`: the worst case is a fallback all the way to individual base symbols.

#### The "byte-level" part, and why $\lvert V\rvert = 50257$

GPT-2's one twist on classic BPE is the choice of **base unit**. Instead of Unicode *characters* (of which there are ~150 000, so the base alphabet alone would be huge and still miss rare scripts), the base alphabet is the **256 possible bytes**. Any text in any language is first encoded to UTF-8 bytes, and BPE merges are learned on top of that byte stream. Two practical details make this work:

- A fixed **byte ↔ printable-Unicode remap** so that bytes like space (`0x20`) or control characters become safe, visible symbols (this is why GPT-2 tokens famously render spaces as `Ġ`).
- A **regex pre-tokenization** step that first chops the text on whitespace/punctuation boundaries, so merges never glue across, say, the end of one word and the start of the next.

The vocabulary is then fixed at

$$
|V| = 50257 = \underbrace{256}_{\text{base bytes}} + \underbrace{50000}_{\text{learned merges}} + \underbrace{1}_{\langle\texttt{endoftext}\rangle}
$$

Every merge adds exactly one entry, so 50 000 merges add 50 000 tokens on top of the 256 base bytes, plus the single special token. This $|V|$ is precisely the number of rows in the embedding table $W_e$ (Section 3.1) and the width of the logits (Section 9) — the tokenizer literally sizes both ends of the model.

The architectural consequence: because the base units are bytes, the tokenizer is **lossless and universal** — any byte string, in any language or encoding, maps to *some* token sequence, so there is never an out-of-vocabulary token and never a need for a `<UNK>` symbol. This is what lets the model treat the corpus as one undifferentiated stream of bytes-turned-tokens, consistent with the "everything is one sequence" philosophy. The single $\langle\texttt{endoftext}\rangle$ token is the only structural delimiter; it marks document boundaries and doubles as the start-of-sequence symbol.

### 3.3 Positional information: learned, not sinusoidal

The 2017 paper injected position with **fixed sinusoids** of geometrically spaced frequencies. Its headline selling point was **extrapolation**: because $\sin$ and $\cos$ are defined for *any* real argument, the formula produces a perfectly valid encoding vector at position 5,000 even if training never went past 512. The paper explicitly hoped this would let a model "extrapolate to sequence lengths longer than the ones encountered during training."

GPT-2 **discards this** and uses a **learned positional embedding** matrix $W_p \in \mathbb{R}^{n_{ctx} \times d}$ with $n_{ctx} = 1024$: position $t$ is just a lookup of row $t$ — a trained parameter vector like any other. This deliberately *throws away* the "any length" property (there is simply no row 1024) and accepts a hard context wall. On its face that is a strange trade: why surrender unbounded length for a fixed 1024-token cap? The answer, which the rest of this subsection unpacks, is that **the length capability being surrendered was never real to begin with.**

#### Why "handles any length" was an illusion

The sinusoidal scheme's extrapolation is a property of the **encoding function**, not of the **trained model**. The formula *defines* a vector at position 5,000 — but what the network *does* with that vector depends on the learned weights $W_Q, W_K$ (and every layer above), and those were **only ever exposed to the positional patterns inside the training range**. Two concrete failure modes:

- **Attention scores go out of distribution.** Recall (from the 2017 companion document's positional deep-dive) that the $QK^\top$ dot product mixes content and position. $W_Q, W_K$ get tuned during training so those dot products land in a sensible range *for the absolute positions and relative offsets actually seen*. A position-5,000 barcode is mathematically valid, but it drives the dot products into magnitude regimes the softmax was never calibrated against — so the attention distribution degenerates (collapses to near-uniform, or spikes), and output quality falls off a cliff.
- **The whole stack is calibrated to the training length.** Position is added *once*, at the input, and then every layer builds its representations assuming the positional statistics of the training range. Feeding genuinely novel positions is out-of-distribution input to the entire model, not just to the embedding lookup.

This is not hypothetical — it is the well-known empirical result that **sinusoidal length extrapolation does not work in practice.** A model trained at 512 tokens performs catastrophically at 2,048; perplexity explodes rather than degrading gracefully. So the "unbounded length" edge that sinusoidal *appears* to hold over a learned table is a theoretical artifact of the encoding formula, not a usable model capability. **In practice both schemes are capped at the training length.** Once you internalize that, sinusoidal's one advantage over a learned table evaporates, and the only question left is *which is better within the length you actually train on.*

#### Why learned wins, given the lengths are effectively equal

Within the trained range the two are empirically a **wash** — the 2017 paper itself ablated learned vs. sinusoidal positional embeddings and reported *nearly identical* quality (their Table 3, row E). Given that tie, GPT-2 (following GPT-1, which had already gone learned) picks the table for two reasons:

- **No hand-designed prior.** Sinusoids hand the model a fixed multi-frequency basis and say "express position in *this* geometry." A learned table imposes nothing — the model discovers whatever positional structure the data rewards. This is the same "weak inductive bias, let the data decide" stance that motivates the decoder-only design itself (Section 1).
- **Mechanical simplicity and symmetry.** A positional embedding becomes just another row-lookup into a parameter matrix — identical machinery to the token embedding $W_e$, with no transcendental functions and no frequency schedule to hand-tune. Token identity and position are now perfectly symmetric operations, both summed into the same residual stream.

So the trade GPT-2 actually made was: **give up an extrapolation ability that did not function anyway, in exchange for a simpler, prior-free scheme that is at least as good over the range the model can really use.** The 1024 wall is not a regression *caused* by learned embeddings — it is an honest declaration of the length the model was trained on. Sinusoidal would have failed past roughly that length too; it simply would not have *told* you so by refusing the input.

#### The honest cost — and how length was eventually fixed properly

The cost is nonetheless real and worth stating bluntly: GPT-2 **cannot represent any position $\ge 1024$** — the row does not exist — so the context window is a hard 1024 tokens. You cannot even *feed* a longer sequence, never mind extrapolate. As demand for longer contexts grew, this became a binding constraint.

The genuine fix arrived later and is **neither** sinusoids-at-the-input **nor** a bigger learned table. **RoPE** and **ALiBi** (a later chapter) move the position signal *out of the input embedding and into the attention score computation itself*, encoding **relative** offset directly where attention consumes it — which is what finally delivered real length generalization. GPT-2 didn't *solve* the length problem; it correctly judged that sinusoidal didn't either, and declined to pay for the pretense. The narrow lesson: *don't carry a theoretical capability that doesn't survive contact with a trained model — keep the simple mechanism, and fix length properly with a different one when it actually matters.*

To summarize the trade-off crisply:

- **Pro (learned):** no hand-designed positional prior; the model learns whatever geometry helps; mechanically identical to token embedding.
- **Con (learned):** exactly $n_{ctx}=1024$ rows, so a hard context wall with no representation of later positions — but note the alternative's "soft" unbounded length was non-functional anyway.

The input to the first block is the elementwise sum:

$$
x_0 = W_e[\,\text{ids}\,] + W_p[\,0{:}T\,] \in \mathbb{R}^{T \times d}
$$

Token identity and position live in the *same* vector space and are added, not concatenated — the model must disentangle them internally, which it learns to do.

---

## 4. The residual stream — the right mental model

Before the block internals, adopt the framing that makes everything below click. Think of $x \in \mathbb{R}^{T \times d}$ not as "the activations" but as a **communication channel** — a residual stream that runs unbroken from $x_0$ at the embedding all the way to $x_N$ at the final LayerNorm.

Every sub-layer does the same three things: it **reads** a (normalized) copy of the stream, **computes** something, and **writes** the result back by addition. No sub-layer ever overwrites the stream; they only add to it. Formally, the entire network is

$$
x_N = x_0 + \sum_{\ell=1}^{N}\Big[\, \text{Attn}_\ell(\cdot) + \text{FFN}_\ell(\cdot) \,\Big]
$$

— a sum of an identity path and many small additive contributions. This is the deep reason residual connections work, and it is the lens through which GPT-2's two big departures from 2017 (Pre-LN and the $1/\sqrt N$ init) should be read: both are about **keeping that additive channel clean and well-scaled as the stack gets deep.**

---

## 5. The Pre-LayerNorm reformulation

This is the most important architectural change inside the block, and it is easy to miss because it is "just" a reordering.

### 5.1 Post-LN (2017) vs Pre-LN (GPT-2)

The 2017 Transformer applies LayerNorm **after** the residual addition:

$$
\textbf{Post-LN:}\qquad x_{\ell} = \mathrm{LN}\big(x_{\ell-1} + \mathrm{Sublayer}(x_{\ell-1})\big)
$$

GPT-2 applies LayerNorm **to the input of the sub-layer**, *inside* the residual branch, leaving the skip connection untouched:

$$
\textbf{Pre-LN:}\qquad x_{\ell} = x_{\ell-1} + \mathrm{Sublayer}\big(\mathrm{LN}(x_{\ell-1})\big)
$$

### 5.2 Why it matters

In Post-LN, the normalization sits **on** the residual highway — every skip connection passes through a LayerNorm, so the clean identity path of Section 4 does not actually exist. Gradients flowing backward are repeatedly rescaled by the LN Jacobian at every layer, and for deep stacks this makes the early layers' gradients unstable; the original Transformer needed a learning-rate **warmup** schedule largely to survive this.

In Pre-LN, the residual addition is **never normalized**. The stream $x_0 \to x_N$ is a pure sum (Section 4), and the gradient of the loss w.r.t. any earlier $x_\ell$ contains an undiminished identity term $\partial \mathcal{L}/\partial x_N$ that flows straight back with no per-layer rescaling. The result: deep Pre-LN stacks train stably, are far less sensitive to warmup, and tolerate higher learning rates. This is a large part of what made GPT-2's deeper stacks (up to 48 layers) practical to train. Post-LN is not *impossible* to scale — BERT-large is 24 Post-LN layers, and later tricks like DeepNorm push it much further — but it grows increasingly sensitive to warmup and initialization as depth rises, which Pre-LN largely removes.

### 5.3 The final LayerNorm

Pre-LN has one loose end. Because the *last* sub-layer's output is added to the stream but never normalized afterward, the activations reaching the output projection are un-normalized and can have grown large over $N$ additive writes. GPT-2 therefore inserts a **single LayerNorm after the last block** (`ln_f`), absent from the 2017 design, to put the stream into a clean, unit-scale state before unembedding. So the pattern is: *normalize on the way into every sub-layer, and once more at the very end.*

### 5.4 LayerNorm itself (unchanged form)

For completeness, LayerNorm normalizes across the feature dimension $d$ per token:

$$
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\qquad
\mu = \frac{1}{d}\sum_{i} x_i,\;\;
\sigma^2 = \frac{1}{d}\sum_{i}(x_i - \mu)^2
$$

with learned $\gamma, \beta \in \mathbb{R}^d$. (Contrast BatchNorm, which normalizes across the batch and is unusable here since sequence positions are not exchangeable and inference is autoregressive.)

---

## 6. Masked multi-head self-attention

The attention mechanism is mechanically the 2017 one — the difference is that **only the causal-masked variant survives, and it now does the work of both decoder self-attention and the deleted cross-attention.**

### 6.1 Projections

Let $\tilde{x} = \mathrm{LN}(x) \in \mathbb{R}^{T \times d}$ be the normalized input. With $h$ heads and per-head dimension $d_k = d/h$ (GPT-2 holds $d_k = 64$ across *all* model sizes), compute

$$
Q = \tilde{x}W_Q,\quad K = \tilde{x}W_K,\quad V = \tilde{x}W_V,
\qquad W_Q, W_K, W_V \in \mathbb{R}^{d \times d}
$$

In the actual implementation these three are a single fused matrix $W_{qkv}\in\mathbb{R}^{d\times 3d}$ for efficiency (the `c_attn` layer), then split. Each is reshaped to $(h, T, d_k)$ so the heads attend independently.

### 6.2 Scaled dot-product with the causal mask

Per head:

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

The $1/\sqrt{d_k}$ scaling is the 2017 fix for keeping the logits' variance $\approx 1$ so softmax does not saturate. The new-in-spirit piece is the **causal mask** $M \in \mathbb{R}^{T\times T}$:

$$
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

i.e. a strictly upper-triangular $-\infty$. Adding $M$ before the softmax drives the attention weight of any "future" key to exactly zero. This single matrix is the entire load-bearing structure of the model:

- It is what makes the $T$ next-token predictions in one forward pass non-cheating (Section 1.3).
- It is what replaces the source/target wall of the 2017 decoder — "context" is just "tokens to my left," reachable by self-attention; "future" is masked. There is no separate encoded sequence to cross-attend to because the left context *is* the sequence.

### 6.3 Heads, concatenation, output projection

The $h$ head outputs $\mathrm{head}_i \in \mathbb{R}^{T\times d_k}$ are concatenated back to $\mathbb{R}^{T\times d}$ and passed through an output projection $W_O \in \mathbb{R}^{d\times d}$ (the `c_proj` layer):

$$
\mathrm{MHSA}(\tilde{x}) = \big[\mathrm{head}_1 \,\|\, \cdots \,\|\, \mathrm{head}_h\big]\,W_O
$$

and this is what gets **added** back to the residual stream. $W_O$ is one of the two "residual-write" matrices that receive the special initialization of Section 8.

---

## 7. The position-wise feed-forward network

Unchanged in shape from 2017, changed in nonlinearity. Applied independently to every position on the normalized stream:

$$
\mathrm{FFN}(\tilde{x}) = \mathrm{GELU}\big(\tilde{x}W_1 + b_1\big)\,W_2 + b_2,
\qquad W_1 \in \mathbb{R}^{d\times 4d},\; W_2 \in \mathbb{R}^{4d\times d}
$$

The inner dimension is $4d$ (the same expansion ratio as 2017's $d_{ff}=2048$ for $d=512$). This block holds the bulk of the parameters — roughly two-thirds of each layer's weights — and is the standard place to locate the model's "stored knowledge" / key-value memory.

### 7.1 GELU instead of ReLU

The FFN's nonlinearity is not decoration — it is *the* source of non-linear expressivity in the block. (Attention, in its values, is only a weighted average — a linear operation; stack two linear maps and you still have a linear map. The activation between $W_1$ and $W_2$ is what makes the whole network more than one big matrix multiply.) So the choice of activation directly shapes what each layer can compute, and it is worth understanding *why* GPT-2 swapped 2017's ReLU for the **Gaussian Error Linear Unit (GELU)** rather than just noting that it did.

**ReLU is a hard gate.** $\mathrm{ReLU}(x)=\max(0,x)$ is cleanest read not as "clip negatives" but as a *gate on the input*:

$$
\mathrm{ReLU}(x) = x\cdot\mathbb{1}[x>0]
$$

— "multiply the input by 1 if it is positive, 0 otherwise." Cheap, sparsifying, and good enough to train early deep nets. But the hard gate has three liabilities:

- a **non-differentiable kink** at $x=0$;
- **dead neurons** — once a unit's pre-activation is pushed negative, it outputs $0$ *and* receives exactly $0$ gradient (the derivative is $0$ for all $x<0$), so it can get stuck off forever (the "dying ReLU" problem);
- an **information-destroying discontinuity** at $0$: inputs of $-0.001$ and $+0.001$ are treated as completely different, and *every* negative value is flattened to the same $0$, throwing away the magnitude of negative evidence.

**GELU keeps the gate but makes it soft and self-referential.** The idea (Hendrycks & Gimpel, 2016) comes from a thought experiment that unifies ReLU with **dropout**. Dropout multiplies $x$ by a random $\{0,1\}$ coin *independent* of $x$; ReLU multiplies $x$ by a deterministic $\{0,1\}$ decided by its *sign*. GELU asks: what if the keep-probability of the mask *depended on how large $x$ is*?

$$
m \sim \mathrm{Bernoulli}\big(\Phi(x)\big), \qquad \text{output} = m\cdot x,
$$

where $\Phi$ is the standard-normal CDF, $\Phi(x)=P(X\le x)$ for $X\sim\mathcal{N}(0,1)$. A large input (so $\Phi(x)$ near $1$) is almost always kept; a very negative one ($\Phi$ near $0$) is almost always zeroed; near $x=0$ it is roughly a coin flip. **GELU is simply the deterministic expectation of that stochastic gate:**

$$
\mathrm{GELU}(x) = \mathbb{E}[m\cdot x] = \Phi(x)\cdot x.
$$

That is the philosophy in one line: *gate each input by the probability — under a normal model of activations — that it is "large," and use the expected value so the function is deterministic.*

> **Reading the notation $m \sim \mathrm{Bernoulli}(\Phi(x))$ — it is not a function of a function.** Three different objects are stacked here, so unpack them one at a time:
> 1. **$\Phi(x)$ is a function that returns a number.** Plug in $x$ and you get a probability in $[0,1]$ — e.g. $\Phi(1.0)\approx0.841$, $\Phi(0)=0.5$, $\Phi(-1.0)\approx0.159$. Call it $p=\Phi(x)$.
> 2. **$m \sim \mathrm{Bernoulli}(p)$ means $m$ is a random coin flip, *not* a function.** The symbol "$\sim$" reads "is drawn from." A $\mathrm{Bernoulli}(p)$ coin lands $1$ with probability $p$ and $0$ with probability $1-p$. So $m$ is a *random variable* whose value is either $0$ or $1$; it isn't a function of $x$, it's a dice roll whose *odds* are set by $p=\Phi(x)$.
> 3. **$\text{output}=m\cdot x$** just multiplies the input by that $0/1$ result — keep it ($m{=}1$) or zero it ($m{=}0$).
>
> *Concretely:* if $x=1.0$ then $p=\Phi(1.0)\approx0.841$, so $84.1\%$ of the time $m{=}1$ and the output is $1.0$ (kept), $15.9\%$ of the time $m{=}0$ and the output is $0$ (dropped). If $x=-1.0$ then $p\approx0.159$, so the output is $-1.0$ only $15.9\%$ of the time and $0$ otherwise. Large inputs are usually kept, very negative ones usually dropped — a soft version of ReLU's hard keep/drop.
>
> *Why this becomes $\Phi(x)\cdot x$:* you don't want randomness at inference, so GELU uses the coin's **average** output. For a Bernoulli coin the average of $m$ is just its probability, $\mathbb{E}[m]=p=\Phi(x)$, hence $\mathbb{E}[m\cdot x]=\Phi(x)\cdot x$. The randomness is only a *motivating story*; the deterministic $\Phi(x)\cdot x$ is what actually runs. **So $m$ is a random $0/1$ value, and $\Phi$ is the only real function — it supplies the probability that controls the coin.**

**Why the *Gaussian* CDF specifically?** Because $\Phi(x)$ answers exactly the right question: "is this neuron's pre-activation large *relative to the typical activation*?" Pre-activations in trained nets are roughly zero-mean and Gaussian-ish — and in GPT-2 the FFN input has *just* been LayerNormed (Section 5) to ≈zero mean, unit variance, which makes the standard normal an unusually faithful reference distribution. Under that model $\Phi(x)$ is literally "the fraction of activations below this one," so GELU scales each value by its own **percentile**. *Position in the distribution*, not a fixed threshold at zero, decides how much signal passes.

**Reading the exact formula and its shape.**

$$
\mathrm{GELU}(x) = x\,\Phi(x) = x\cdot\tfrac{1}{2}\Big[1 + \mathrm{erf}\!\big(x/\sqrt 2\big)\Big]
$$

The limits recover ReLU's *envelope* without its sharp corner:

- $x\to+\infty$: $\Phi\to1$, so $\mathrm{GELU}(x)\to x$ — large positives pass through, like ReLU;
- $x\to-\infty$: $\Phi\to0$ (and fast), so $\mathrm{GELU}(x)\to0$ — large negatives are suppressed, like ReLU;
- $x=0$: $\Phi(0)=\tfrac12$, so $\mathrm{GELU}(0)=0$ — but as a smooth pass-through-half, not a kink.

Two properties of the smooth version genuinely matter for training:

- **It is non-monotonic.** For moderately negative inputs $\Phi(x)$ is still appreciable, so the product $x\Phi(x)$ dips *below zero* — reaching a minimum of about $-0.17$ near $x\approx-0.75$ — before rising back toward $0$ as $x\to-\infty$. ReLU can never emit a negative number; GELU's small negative lobe lets a unit express "mild negative evidence" instead of collapsing it to nothing.
- **Its gradient never fully dies.** $\mathrm{GELU}'(x) = \Phi(x) + x\,\varphi(x)$ (with $\varphi$ the normal PDF) is smooth everywhere and stays nonzero for negative $x$ near the origin — so a unit sitting at a slightly negative pre-activation still receives gradient and can climb back, curing ReLU's dead-neuron pathology.

A few concrete values make the contrast vivid:

| $x$ | $-3$ | $-1$ | $-0.5$ | $0$ | $0.5$ | $1$ | $3$ |
|---|---|---|---|---|---|---|---|
| $\mathrm{ReLU}(x)$ | $0$ | $0$ | $0$ | $0$ | $0.500$ | $1.000$ | $3.000$ |
| $\mathrm{GELU}(x)$ | $-0.004$ | $-0.159$ | $-0.154$ | $0$ | $0.346$ | $0.841$ | $2.996$ |

Notice GELU barely alters strong positives ($2.996\approx3$) or strong negatives ($-0.004\approx0$) — it agrees with ReLU at the extremes — and differs precisely in the **soft transition zone** around zero, exactly where ReLU's hard decision is most arbitrary.

**The tanh approximation (what GPT-2 actually runs).** The exact form needs $\mathrm{erf}$, which was comparatively slow on 2019 hardware, so GPT-2 evaluates a closed-form approximation:

$$
\mathrm{GELU}(x) \approx 0.5\,x\Big(1 + \tanh\!\big[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\big]\Big)
$$

The constants are fitted, not magical: $\sqrt{2/\pi}\approx0.798$ reproduces the **leading (linear) term** of $\mathrm{erf}$ near the origin, and $0.044715$ is a small **cubic correction** that aligns the tails. (A second common approximation, $\mathrm{GELU}(x)\approx x\,\sigma(1.702\,x)$ with the logistic sigmoid $\sigma$, exposes GELU's kinship with **Swish / SiLU**, $x\,\sigma(\beta x)$ — both belong to the same *self-gated* family: an input multiplied by a smooth function of itself.) The approximation matches the true GELU to a few thousandths everywhere, and it is the form that produced $\mathrm{GELU}(-0.295)=-0.113$ in the worked example of Section 12.

**Why it stuck.** Empirically the smooth, self-gating shape yields a better-conditioned loss surface and slightly better final quality than ReLU in Transformers, at negligible extra cost — which is why GELU became the default across BERT and the entire GPT line. The lineage then continues: most modern models (LLaMA, PaLM) replace this GELU MLP with **SwiGLU**, a *gated* variant that splits the up-projection in two and multiplies the halves, pushing the same self-gating idea one step further (Section 16).

---

## 8. Initialization — the $1/\sqrt{N}$ detail

This is a small clause in the GPT-2 paper that has outsized importance and is the third change (after Pre-LN and `ln_f`) aimed at the residual stream.

Base scheme: linear and attention weights drawn from $\mathcal{N}(0, 0.02^2)$, biases at 0, LayerNorm $\gamma=1,\beta=0$. The token embedding $W_e$ uses std $0.02$; in OpenAI's original implementation the positional table $W_p$ uses a smaller std of $0.01$. (These init constants are an implementation detail of the released code, not stated in the paper text — and many reimplementations such as nanoGPT simply use $0.02$ for both.)

The special clause, quoting the idea: *the weights of residual layers are scaled at initialization by $1/\sqrt{N}$, where $N$ is the number of residual layers.*

**Why.** Recall the stream is a sum of $N$ block-writes (Section 4). If each write has variance $\sim v$ and they are roughly independent, the variance of the stream grows like $N\cdot v$ — it *accumulates with depth*. Left unchecked, deep models would have wildly large activations at the top of the stack before any training. Scaling each residual-write matrix's init by $1/\sqrt N$ makes each contribution's std shrink like $1/\sqrt N$, so the accumulated variance stays $\mathcal{O}(1)$ regardless of depth. Concretely the factor is applied to the two matrices that write into the stream — the attention output projection $W_O$ (`c_proj`) and the FFN down-projection $W_2$ (`c_proj`). Here $N$ counts the residual *additions* (= $2 \times n_{layer}$).

Together, Pre-LN (clean gradient path), `ln_f` (clean final scale), and $1/\sqrt N$ (controlled forward variance) are the three-part answer to "how do you train a 48-layer Transformer *stably and without delicate tuning*" — something the bare 2017 recipe handles far less gracefully as depth grows.

---

## 9. Output: weight tying and the unembedding

After `ln_f`, the stream $x_N \in \mathbb{R}^{T\times d}$ is projected to vocabulary logits. GPT-2 **ties** the unembedding to the input embedding — it reuses $W_e^\top$ rather than learning a separate $|V|\times d$ output matrix:

$$
\text{logits} = x_N\,W_e^\top \in \mathbb{R}^{T\times |V|},
\qquad
P(w_t \mid w_{<t}) = \mathrm{softmax}(\text{logits}_t)
$$

Rationale: the input embedding already learns a map *token $\to$ vector*; the output needs the inverse map *vector $\to$ token-similarity*. Sharing the matrix enforces a consistent token geometry on both ends, saves a large parameter block ($50257 \times 768 \approx 38.6$M for the small model — about a third of all parameters), and acts as a regularizer. There is no bias on the logits. (This is not a GPT-2 invention — the 2017 Transformer already tied its embedding and pre-softmax weights, citing Press & Wolf 2017; GPT-2 simply keeps the practice.)

---

## 10. The complete forward pass, in one place

Putting Sections 3–9 together, with $\ell = 1\dots N$:

$$
\begin{aligned}
x_0 &= W_e[\text{ids}] + W_p[0{:}T] \\[4pt]
a_\ell &= x_{\ell-1} + \mathrm{MHSA}\big(\mathrm{LN}_1^{(\ell)}(x_{\ell-1})\big) \quad\text{(causal mask inside MHSA)}\\[4pt]
x_\ell &= a_\ell + \mathrm{FFN}\big(\mathrm{LN}_2^{(\ell)}(a_\ell)\big) \\[4pt]
\text{logits} &= \mathrm{LN}_f(x_N)\,W_e^\top \\[4pt]
P(w_t \mid w_{<t}) &= \mathrm{softmax}(\text{logits}_t)
\end{aligned}
$$

That is the entire model. Contrast the 2017 forward pass, which additionally maintained an encoder stack producing memory $\mathbf{m}$, and whose decoder block had a *third* sub-layer $\mathrm{CrossAttn}(x, \mathbf{m})$ between self-attention and FFN. GPT-2 is the 2017 decoder with the cross-attention sub-layer excised, the LayerNorms moved to the front, and a language-model head bolted on.

---

## 11. Training vs. generation

### 11.1 Training: fully parallel

During training the whole target sequence is known, so all $T$ positions run in one forward pass. The causal mask guarantees position $t$'s prediction depends only on $w_{<t}$, so the $T$ cross-entropy losses are computed jointly and averaged. This is the same parallelism the 2017 *decoder* enjoyed at training time via teacher forcing; GPT-2 just applies it to the entire model since the entire model *is* a decoder.

### 11.2 Generation: inherently sequential

At inference there is no known continuation, so generation is autoregressive: sample $w_{t}\sim P(\cdot\mid w_{<t})$, append it, and feed the longer sequence back in to get $w_{t+1}$. Architecturally this means:

- The model is called once per generated token.
- Each call, the new token's $Q$ attends to **all previous** $K,V$. Naively re-encoding the whole prefix every step is $\mathcal{O}(T^2)$ wasted work, which is exactly why **KV caching** exists — store each position's $K,V$ once and reuse them. (GPT-2 predates the modern inference-optimization stack, but its causal structure is precisely what makes KV caching valid: a past token's representation never depends on future tokens, so its $K,V$ are immutable once computed.)
- Generation halts on $\langle\texttt{endoftext}\rangle$ or a length cap, and is bounded by the 1024-position learned table (Section 3.3).

---

## 12. A complete worked example: one small sequence, end to end

Everything above is the architecture in symbols. This section runs **one tiny sequence all the way through GPT-2 with real numbers**, first as a single **training** forward pass (all positions at once) and then as **inference** (autoregressive generation). It is the GPT-2 analogue of the encoder/decoder walkthrough in the 2017 companion document, and it is deliberately built to expose the four things that make GPT-2 *GPT-2* and not the 2017 decoder: **learned positional embeddings added at the input, Pre-LN inside each branch, the causal mask, and a weight-tied LM head.**

To keep every matrix readable we shrink the model to toy dimensions. The *shapes* are smaller; the *operations* are bit-for-bit the ones in Sections 3–11.

| Symbol | Real GPT-2 small | This toy |
|---|---|---|
| $d$ ($n_{embd}$) | 768 | **4** |
| $h$ (heads) | 12 | **2** ($d_k=2$) |
| $n_{layer}$ | 12 | **1** (one block, traced fully) |
| $d_{ff}$ | $4d=3072$ | **8** ($=2d$, shrunk for width) |
| $\lvert V\rvert$ | 50257 | **10** |
| $n_{ctx}$ | 1024 | **12** |

> All values are rounded to 3 decimals for display. The **explicit step-by-step computations** (the worked dot products, the softmax in 12.4, the losses in 12.5) are internally consistent at that precision — you can reproduce them with a calculator from the numbers shown. The **large intermediate matrices** were computed in full precision, so if you recompute an entire matrix by hand from rounded inputs you may see a last-digit drift. The weights are arbitrary small constants standing in for trained parameters — so the model's *predictions* here are meaningless (an untrained net), but every *operation* is exactly GPT-2's.

### 12.0 The toy vocabulary and sequence

A 10-token byte-level-BPE-style vocabulary (recall Section 3.2: there is never an `<UNK>`, and `<eot>` is the only structural token). It is deliberately larger than the handful of words this one sequence will use — exactly like real GPT-2, where any short text touches only a tiny fraction of the 50,257 tokens:

```
0:<eot>   1:the   2:cat   3:sat   4:on   5:mat   6:dog   7:ran   8:red   9:box
```

We feed the text **"the cat sat"**, which tokenizes to ids $[1,2,3]$, so $T=3$. Because GPT-2's one and only task is next-token prediction (Section 1.3), the **training targets are the same stream shifted left by one** — each position is supervised to predict the token that actually follows it in the corpus text *"the cat sat on"*. The remaining vocabulary words (`mat, dog, ran, red, box`) simply never occur in this short snippet, so they are neither inputs nor targets here — they only ever appear as *competing candidates* in the output distribution:

| position $t$ | input id | input token | target id | target token |
|---|---|---|---|---|
| 0 | 1 | the | 2 | cat |
| 1 | 2 | cat | 3 | sat |
| 2 | 3 | sat | 4 | on |

This "labels = inputs shifted by one" is the whole supervision signal. No separate label set exists; a length-$T$ sequence yields $T$ supervised predictions for free (Section 15, point 3).

### 12.1 Input: token embedding + *learned* positional embedding (Section 3)

Token embedding table $W_e \in \mathbb{R}^{10\times 4}$ (one row per vocab id) and learned positional table $W_p \in \mathbb{R}^{12\times 4}$ (one row per absolute position — **not** a sinusoid):

$$
\begin{aligned}
W_e &= \begin{bmatrix}
0.10&0.20&-0.10&0.30\\
0.50&-0.20&0.30&0.10\\
-0.30&0.40&0.20&-0.10\\
0.20&0.10&-0.40&0.50\\
0.30&-0.30&0.10&0.20\\
-0.10&0.30&0.40&-0.20\\
0.40&0.10&-0.20&-0.30\\
-0.20&-0.40&0.30&0.20\\
0.10&0.50&-0.30&0.00\\
-0.40&0.20&0.10&0.30
\end{bmatrix} \quad (\text{10 rows: } \langle\text{eot}\rangle,\text{the},\text{cat},\text{sat},\text{on},\text{mat},\text{dog},\text{ran},\text{red},\text{box})\\[6pt]
W_p &= \begin{bmatrix}
0.00&0.10&0.20&-0.10\\
0.10&-0.10&0.00&0.20\\
-0.20&0.20&0.10&0.00\\
\vdots&\vdots&\vdots&\vdots
\end{bmatrix} \quad (\text{only rows } 0,1,2 \text{ used here})
\end{aligned}
$$

Look up rows $[1,2,3]$ of $W_e$ and rows $[0,1,2]$ of $W_p$ and **add them** (Section 3.3 — same space, summed, not concatenated):

$$
\begin{aligned}
W_e[\text{ids}] &= \begin{bmatrix}0.50&-0.20&0.30&0.10\\-0.30&0.40&0.20&-0.10\\0.20&0.10&-0.40&0.50\end{bmatrix} \quad (\text{rows for the, cat, sat})\\[6pt]
W_p[0{:}3] &= \begin{bmatrix}0.00&0.10&0.20&-0.10\\0.10&-0.10&0.00&0.20\\-0.20&0.20&0.10&0.00\end{bmatrix} \quad (\text{rows for positions } 0,1,2)\\[6pt]
x_0 = W_e[\text{ids}] + W_p[0{:}3] &= \begin{bmatrix}0.500&-0.100&0.500&0.000\\-0.200&0.300&0.200&0.100\\0.000&0.300&-0.300&0.500\end{bmatrix}
\end{aligned}
$$

This $x_0\in\mathbb{R}^{3\times4}$ is where the **residual stream** begins (Section 4). Every operation from here only *reads a normalized copy of it and adds back*.

### 12.2 The decoder block, residual #1: Pre-LN → masked attention → add

**(a) Pre-LayerNorm (Section 5).** Unlike 2017, LN comes *first*, inside the branch, with $\gamma=1,\beta=0$ here. Normalizing each row across its 4 features:

$$
\tilde{x}=\mathrm{LN}_1(x_0)=
\begin{bmatrix}
0.992&-1.172&0.992&-0.811\\
-1.603&1.069&0.534&0.000\\
-0.412&0.577&-1.402&1.237
\end{bmatrix}
$$

(Row 0 check: mean of $[0.5,-0.1,0.5,0]=0.225$, variance $=0.0769$, so $(0.5-0.225)/\sqrt{0.0769+\epsilon}\approx0.992$. ✓)

**(b) Fused QKV projection (Section 6.1).** GPT-2 computes $Q,K,V$ with one fused matrix `c_attn` $\in\mathbb{R}^{4\times12}$, then splits it into three $4\times4$ blocks $W_Q,W_K,W_V$. Using

$$
\begin{aligned}
W_Q &= \begin{bmatrix}0.2&-0.1&0.3&0.0\\0.1&0.2&-0.1&0.1\\0.0&0.3&0.1&-0.2\\0.2&0.0&0.1&0.3\end{bmatrix}\\[6pt]
W_K &= \begin{bmatrix}0.1&0.0&-0.2&0.1\\0.3&0.1&0.0&-0.1\\0.1&-0.2&0.2&0.0\\0.0&0.1&0.1&0.2\end{bmatrix}\\[6pt]
W_V &= \begin{bmatrix}0.0&0.2&0.1&-0.1\\0.1&-0.1&0.2&0.0\\-0.2&0.1&0.0&0.3\\0.2&0.0&-0.1&0.1\end{bmatrix}
\end{aligned}
$$

gives ($\tilde{x}W_Q$, etc.):

$$
\begin{aligned}
Q &= \begin{bmatrix}-0.081&-0.036&0.433&-0.559\\-0.214&0.534&-0.534&0.000\\0.223&-0.264&-0.198&0.709\end{bmatrix}\\[6pt]
K &= \begin{bmatrix}-0.153&-0.397&-0.081&0.054\\0.214&0.000&0.428&-0.267\\-0.008&0.462&-0.074&0.148\end{bmatrix}\\[6pt]
V &= \begin{bmatrix}-0.478&0.415&-0.054&0.117\\0.000&-0.374&0.053&0.321\\0.586&-0.280&-0.049&-0.256\end{bmatrix}
\end{aligned}
$$

Each entry is one row·column dot product. Worked, the top-left of $Q$ ($=\tilde x$ row 0 dotted with column 0 of $W_Q$):

$$
Q_{0,0}=(0.992)(0.2)+(-1.172)(0.1)+(0.992)(0.0)+(-0.811)(0.2)=-0.081\ \checkmark
$$

Every other entry of $Q,K,V$ is the same operation with a different row/column.

**(c) Split into $h=2$ heads.** Columns $0{:}2$ are head 0, columns $2{:}4$ are head 1, each with $d_k=2$.

**(d) Scaled dot-product with the causal mask (Section 6.2).** Per head, $\frac{Q_hK_h^\top}{\sqrt2}+M$ where $M$ is the strictly-upper-triangular $-\infty$ mask. For **head 0** the raw masked score matrix and its softmax (row-wise) are:

$$
\text{scores}_0=\begin{bmatrix}0.019&-\infty&-\infty\\-0.127&-0.032&-\infty\\0.050&0.034&-0.087\end{bmatrix}
\;\xrightarrow{\text{softmax}}\;
A_0=\begin{bmatrix}1.000&0&0\\0.476&0.524&0\\0.350&0.345&0.305\end{bmatrix}
$$

Worked, for **position 1 in head 0** (it may attend to positions 0 and 1, but not 2). Using the head-0 slices $Q_1^{(0)}=[-0.214,0.534]$, $K_0^{(0)}=[-0.153,-0.397]$, $K_1^{(0)}=[0.214,0.000]$, the two live scaled scores are

$$
\frac{Q_1^{(0)}\!\cdot K_0^{(0)}}{\sqrt2}=\frac{(-0.214)(-0.153)+(0.534)(-0.397)}{1.414}=-0.127,\qquad
\frac{Q_1^{(0)}\!\cdot K_1^{(0)}}{\sqrt2}=-0.032,
$$

and position 2 is forced to $-\infty$ by the mask $M$. Softmax over the row turns these into weights — and the masked entry contributes $e^{-\infty}=0$, which is *exactly* how the future is zeroed out:

$$
A_{0,\text{row }1}=\frac{[\,e^{-0.127},\,e^{-0.032},\,e^{-\infty}\,]}{e^{-0.127}+e^{-0.032}+0}=\frac{[\,0.881,\,0.969,\,0\,]}{1.850}=[\,0.476,\,0.524,\,0\,]\ \checkmark
$$

The mask is doing its one load-bearing job: position 0 can attend **only to itself** (row sums to 1 on the diagonal alone), position 1 to $\{0,1\}$, position 2 to all three. No position sees its future. Head 1 similarly:

$$
A_1=\begin{bmatrix}1.000&0&0\\0.548&0.452&0\\0.352&0.279&0.369\end{bmatrix}
$$

**(e) Weighted sum of values, per head, then concatenate.** $\text{head}_i=A_iV_i$, concatenated back to width 4:

$$
\text{concat}=\big[\text{head}_0 \,\|\, \text{head}_1\big]=
\begin{bmatrix}
-0.478&0.415&-0.054&0.117\\
-0.228&0.002&-0.005&0.209\\
0.011&-0.069&-0.022&0.036
\end{bmatrix}
$$

Worked, row 1 of head 0 — the attention weights of position 1 ($A_{0,\text{row }1}=[0.476,0.524,0]$) applied to head 0's three value rows $V^{(0)}_0=[-0.478,0.415]$, $V^{(0)}_1=[0.000,-0.374]$, $V^{(0)}_2=[0.586,-0.280]$:

$$
\text{head}_0[\text{row }1]=0.476\,[-0.478,0.415]+0.524\,[0.000,-0.374]+0\,[0.586,-0.280]=[-0.228,\,0.002]
$$

These two numbers are the first two columns of concat-row 1; head 1's two columns ($[-0.005,0.209]$) are computed the same way with $A_1$ and $V^{(1)}$, then laid side by side to rebuild width 4.

**(f) Output projection `c_proj` (Section 6.3)** $W_O\in\mathbb{R}^{4\times4}$ (one of the two $1/\sqrt N$-scaled residual-write matrices, Section 8):

$$
\mathrm{MHSA}(\tilde x)=\text{concat}\cdot W_O=
\begin{bmatrix}
-0.096&0.066&-0.047&0.124\\
-0.068&-0.021&-0.003&0.086\\
-0.001&-0.020&0.000&0.003
\end{bmatrix}
$$

**(g) Residual add #1.** Write it back onto the *un-normalized* stream (the skip carries $x_0$, not $\tilde x$ — Section 5.1):

$$
a = x_0 + \mathrm{MHSA}(\mathrm{LN}_1(x_0))=
\begin{bmatrix}
0.404&-0.034&0.453&0.124\\
-0.268&0.279&0.197&0.186\\
-0.001&0.280&-0.300&0.503
\end{bmatrix}
$$

### 12.3 The decoder block, residual #2: Pre-LN → FFN(GELU) → add

**(a) Second Pre-LN** on $a$:

$$
\tilde a=\mathrm{LN}_2(a)=
\begin{bmatrix}
0.832&-1.352&1.080&-0.560\\
-1.707&0.841&0.460&0.407\\
-0.404&0.530&-1.395&1.269
\end{bmatrix}
$$

**(b) FFN up-projection** $W_1\in\mathbb{R}^{4\times8}$ (expand to $d_{ff}=8$), giving the pre-activations $\tilde a W_1$:

$$
\begin{bmatrix}
-0.295&-0.470&0.295&0.414&-0.295&0.266&-0.274&0.438\\
-0.048&0.548&0.048&-0.507&-0.048&-0.041&0.296&-0.290\\
0.205&0.515&-0.205&-0.388&0.205&0.008&0.220&-0.487
\end{bmatrix}
$$

**(c) GELU (tanh approximation, Section 7.1)** applied elementwise. Note the hallmark of GELU vs ReLU: small negative inputs stay *slightly negative* rather than being hard-zeroed (e.g. $-0.295\to-0.113$, not $0$):

$$
\mathrm{GELU}(\cdot)=
\begin{bmatrix}
-0.113&-0.150&0.182&0.273&-0.113&0.161&-0.108&0.293\\
-0.023&0.388&0.025&-0.155&-0.023&-0.020&0.182&-0.112\\
0.119&0.358&-0.086&-0.135&0.119&0.004&0.129&-0.152
\end{bmatrix}
$$

Worked, the first entry (using $\sqrt{2/\pi}=0.798$):

$$
\mathrm{GELU}(-0.295)=0.5(-0.295)\Big(1+\tanh\big[\,0.798\,(-0.295+0.044715\,(-0.295)^3)\,\big]\Big)=-0.113
$$

A ReLU would have returned exactly $0$ here; GELU lets a small negative value through, which is the whole point of the smoother gate (Section 7.1).

**(d) FFN down-projection** $W_2\in\mathbb{R}^{8\times4}$ (the second $1/\sqrt N$ residual-write matrix, Section 8) back to width 4:

$$
\mathrm{FFN}(\tilde a)=
\begin{bmatrix}
-0.064&0.065&0.053&0.039\\
0.064&0.017&-0.001&-0.020\\
0.061&0.024&-0.007&-0.031
\end{bmatrix}
$$

Worked, the top-left: $\mathrm{FFN}_{0,0}=\mathrm{GELU}_{\text{row }0}\cdot W_{2,\text{col }0}$, an 8-term dot product summing the widened hidden vector back down to one of the 4 output features, $=-0.064$.

**(e) Residual add #2** — end of the one block:

$$
x_1 = a + \mathrm{FFN}(\mathrm{LN}_2(a))=
\begin{bmatrix}
0.339&0.031&0.506&0.163\\
-0.204&0.296&0.196&0.166\\
0.060&0.304&-0.307&0.472
\end{bmatrix}
$$

> **In real GPT-2 you repeat Section 12.2–12.3 twelve times** (small) to forty-eight times (XL), each block reading and writing the same residual stream. We trace one block; the structure of the next is identical, just with its own weights.

### 12.4 Final LayerNorm and the weight-tied LM head (Sections 5.3 and 9)

After the last block, apply the once-only `ln_f` (the loose end Pre-LN leaves — Section 5.3):

$$
x_N'=\mathrm{LN}_f(x_1)=
\begin{bmatrix}
0.442&-1.276&1.373&-0.539\\
-1.675&0.964&0.435&0.277\\
-0.247&0.587&-1.500&1.160
\end{bmatrix}
$$

Then **unembed with the tied matrix** $W_e^\top$ (no new parameters, no bias — Section 9): $\text{logits}=x_N'\,W_e^\top \in \mathbb{R}^{3\times10}$. The result has **one column per vocabulary token** (10 columns now), so each row is a raw score for every possible next token:

$$
\text{logits}=
\begin{bmatrix}
-0.510&0.834&-0.315&-0.857&0.545&0.230&-0.064&0.726&-1.006&-0.457\\
0.065&-0.872&0.947&-0.274&-0.693&0.575&-0.744&0.135&0.184&0.989\\
0.591&-0.575&-0.107&1.189&-0.168&-0.631&-0.088&-0.403&0.718&0.414
\end{bmatrix}
$$

Because the head is **tied** to $W_e$ (Section 9), each logit is literally a *dot product of the final token vector with a vocabulary row* — i.e. a similarity score between "what this position now represents" and "what token $j$ looks like." Worked, the logit for `the` (id 1) at position 0 is $x'_{N,\text{row }0}\cdot W_{e,\text{row }1}$:

$$
(0.442)(0.50)+(-1.276)(-0.20)+(1.373)(0.30)+(-0.539)(0.10)=0.834
$$

Each row of `logits` is now turned into a probability distribution by the **softmax**, applied independently to each row. For a logit row $z=(z_0,\dots,z_9)$, the probability assigned to vocab id $j$ is

$$
P_j=\mathrm{softmax}(z)_j=\frac{e^{z_j}}{\sum_{k=0}^{9}e^{z_k}}
$$

Softmax does two things at once: $e^{z_j}$ makes every entry positive and amplifies larger logits, and dividing by the sum forces the ten numbers to add to 1. Let's compute **position 0's row** in full — its logits are $z=[-0.510,\,0.834,\,-0.315,\,-0.857,\,0.545,\,0.230,\,-0.064,\,0.726,\,-1.006,\,-0.457]$.

*Step 1 — exponentiate each of the 10 logits:*

$$
e^{z}=[\,0.600,\;2.303,\;0.730,\;0.424,\;1.725,\;1.259,\;0.938,\;2.067,\;0.366,\;0.633\,]
$$

*Step 2 — sum them:* $\;\sum_k e^{z_k}=0.600+2.303+0.730+0.424+1.725+1.259+0.938+2.067+0.366+0.633=11.045$.

*Step 3 — divide each by the sum:*

$$
P_{\text{pos }0}=\frac{1}{11.045}\,e^{z}=[\,0.054,\;0.209,\;0.066,\;0.038,\;0.156,\;0.114,\;0.085,\;0.187,\;0.033,\;0.057\,]
$$

Those ten numbers are non-negative and sum to 1 — a genuine distribution over the whole vocabulary `(<eot>, the, cat, sat, on, mat, dog, ran, red, box)`. Running the identical three steps on the other two logit rows gives the full per-position matrix $P$ (each row independently sums to 1):

$$
P=
\begin{bmatrix}
0.054&0.209&0.066&0.038&0.156&0.114&0.085&0.187&0.033&0.057\\
0.085&0.033&0.204&0.060&0.040&0.141&0.038&0.091&0.095&0.213\\
0.138&0.043&0.069&0.251&0.065&0.041&0.070&0.051&0.157&0.116
\end{bmatrix}
$$

with columns $\langle\text{eot}\rangle,\text{the},\text{cat},\text{sat},\text{on},\text{mat},\text{dog},\text{ran},\text{red},\text{box}$. Row $t$ is the model's predicted distribution for **the token that comes after position $t$** — three predictions produced in this one parallel pass.

> **Reading the shape of $P$ — and why the loss next uses only 3 of its 30 numbers.** This trips people up, so it is worth pinning down. $P$ is $3\times10$:
> - The **10 columns are the whole vocabulary** $\langle\text{eot}\rangle,\text{the},\text{cat},\text{sat},\text{on},\text{mat},\text{dog},\text{ran},\text{red},\text{box}$. A single row is a *complete* probability distribution over every possible next token, which is why each row sums to 1. (Most of these tokens — `mat, dog, ran, red, box` — never appear in our text; they are just always-present competing candidates.)
> - The **3 rows are the 3 positions** in the sequence. At *every* position the model scores *all* 10 tokens.
>
> So the model genuinely produces all $3\times10 = 30$ probabilities. But the loss does not need a "score" for the wrong tokens — at each position there is exactly **one** correct next token (we know the text is "the cat sat on"), and the only question the loss asks is *"how much probability did you put on that one correct token?"* That is **one number per row** — the column matching the true next token — so 3 rows give exactly 3 numbers:
> $$P_{0,\,\textbf{cat}}=0.066,\qquad P_{1,\,\textbf{sat}}=0.060,\qquad P_{2,\,\textbf{on}}=0.065.$$
> The other 27 entries are **not** discarded. (1) They were already used: $P_{0,\text{cat}}$ is only $0.066$ *because* softmax handed mass to `the`, `ran`, `on`, … instead — the wrong-token probabilities are baked into the right-token probability through the normalization. (2) They drive learning: the gradient in 12.5 is the full 10-wide vector $P - y_{\text{onehot}}$, which lowers every wrong token's logit and raises the correct one. The loss *value* is summarized by one probability per position; the loss *gradient* still touches all ten columns.

### 12.5 Training: computing the loss, explicitly

The forward pass produced three predicted distributions (the rows of $P$). Training scores each one against the **true** next token using the **cross-entropy loss**. The general cross-entropy between a true distribution $y$ and a predicted distribution $P$ is $-\sum_j y_j \ln P_j$; here the truth is a single known token, so $y$ is **one-hot** (a 1 on the correct id, 0 everywhere else) and the whole sum collapses to one term — the negative log of the probability the model gave the correct token:

$$
\mathcal{L}_t=-\sum_{j} y_{t,j}\,\ln P_{t,j}=-\ln P_{t,\,y_t}
$$

Why this is the right quantity: if the model put *all* its mass on the correct token ($P=1$) the loss is $-\ln 1 = 0$; the less mass it assigned, the larger $-\ln P$ becomes (growing without bound as $P\to 0$). Minimizing it therefore literally means "put more probability on the true next token." Computing it for each position — just read $P_{t,\,y_t}$ off the matrix above and take $-\ln$:

| position $t$ | input | true next token $y_t$ | $P_{t,\,y_t}$ | $\mathcal{L}_t=-\ln P_{t,y_t}$ |
|---|---|---|---|---|
| 0 | the | cat (id 2) | $P_{0,2}=0.066$ | $-\ln 0.066 = 2.718$ |
| 1 | cat | sat (id 3) | $P_{1,3}=0.060$ | $-\ln 0.060 = 2.813$ |
| 2 | sat | on (id 4)  | $P_{2,4}=0.065$ | $-\ln 0.065 = 2.733$ |

The loss for the whole sequence is the **average** over the $T=3$ positions:

$$
\mathcal{L}=\frac{1}{T}\sum_{t=1}^{T}\mathcal{L}_t=\frac{1}{3}\big(2.718+2.813+2.733\big)=2.755
$$

As a sanity check, a model guessing **uniformly** over the 10 tokens would assign $P=1/10$ to the truth and score $-\ln(1/10)=\ln 10\approx2.303$ everywhere. Our untrained net (2.755) is slightly *worse* than uniform — exactly what random weights should give. Training is the process of driving this number down.

**How that one scalar becomes weight updates.** $\mathcal{L}$ is a single number, so `loss.backward()` differentiates it with respect to every parameter and the optimizer steps each one in the direction that lowers $\mathcal{L}$. The first link in that backward chain — the gradient of the loss with respect to the **logits** — has a famously clean closed form for softmax-followed-by-cross-entropy:

$$
\frac{\partial \mathcal{L}_t}{\partial z_{t,j}}=P_{t,j}-y_{t,j}\qquad(\text{``predicted distribution minus the one-hot target''})
$$

For position 0 (true token `cat`, id 2), that gradient is the predicted row minus a one-hot at index 2:

$$
\frac{\partial \mathcal{L}_0}{\partial z_0}=
[\,0.054,\,0.209,\,0.066,\,0.038,\,0.156,\,0.114,\,0.085,\,0.187,\,0.033,\,0.057\,]-[\,0,0,1,0,0,0,0,0,0,0\,]
=[\,0.054,\,0.209,\,-0.934,\,0.038,\,0.156,\,0.114,\,0.085,\,0.187,\,0.033,\,0.057\,]
$$

Read the signs and you can watch learning about to happen: the **one negative entry sits on the correct token** `cat`, so gradient descent will *raise* its logit on the next step, while every wrong token has a **positive** gradient that *lowers* its logit. The magnitudes rank how wrong each was — `the` (0.209) and `ran` (0.187) are pushed down hardest because the model wrongly favored them most. This 10-vector is then propagated backward through the tied head $W_e^\top$, then `ln_f`, the decoder block, and finally the embedding tables, adjusting **every** weight we used ($W_e, W_p, W_Q, W_K, W_V, W_O, W_1, W_2$, both LayerNorms, and `ln_f`) so that the same input next time assigns a little more mass to `cat`. (With $\ln 10\approx2.30$ the uniform baseline, and 2.755 our current loss — see the note at the top of Section 12 on why an untrained net lands here.)

Notice what made the parallelism honest: the loss at position 0 used only column-0 information, position 1 only positions $\{0,1\}$, position 2 all three — precisely the structure the mask enforced in Section 12.2(d). **Three labels, one forward pass, zero leakage.**

### 12.6 Inference: the same machine, run autoregressively

At generation time there is no known continuation, so we keep **only the last row** of $P$ — the distribution for the token *after* everything seen so far (Section 11.2). After feeding "the cat sat", that is position 2's row:

$$
P_{\text{pos }2}=[\,0.138,\;0.043,\;0.069,\;\mathbf{0.251},\;0.065,\;0.041,\;0.070,\;0.051,\;0.157,\;0.116\,]
$$

Greedy decoding takes the argmax → id **3 = "sat"** (probability 0.251, the largest of the ten). We **append** it and feed the length-4 sequence $[1,2,3,3]$ back in to get the next token — and so on until `<eot>` or the context cap of $n_{ctx}$.

**Why this is cheap — the KV cache (Section 11.2), shown concretely.** When we extend $[1,2,3]\to[1,2,3,3]$, the $K$ and $V$ rows for positions 0, 1, 2 come out **bit-identical** to what we already computed in Section 12.2(b) — because a causal token's representation never depends on anything to its right. Verified directly on this toy:

```
K,V rows for positions 0-2 :  identical for T=3 and T=4   ✓
```

So a correct implementation stores each position's $(K,V)$ once and, per new token, computes only the **new** row's $Q,K,V$ and attends it against the cached stack — turning the naive $\mathcal{O}(T^2)$ re-encoding into $\mathcal{O}(T)$ work per step. Running that second step here yields $P_{\text{pos }3}=[0.131,0.070,0.042,\mathbf{0.275},0.095,0.031,0.100,0.051,0.134,0.070]$, i.e. greedy emits "sat" again (the weights are random, so the *content* is gibberish — but the *mechanism* is exactly production GPT-2). Section 12.7 does that second step **the naive way** — the whole four-token forward pass recomputed from scratch, every number shown — which is both the honest baseline KV caching optimizes *away* and the cleanest proof of why caching is exact.

**The one-line takeaway of the whole example:** *training* and *inference* run the **identical** stack of operations Section 12.1–12.4; the only difference is that training reads **all** rows of $P$ at once to compute $T$ losses in parallel (Section 12.5), while inference reads the **last** row, samples, appends, and repeats (Section 12.6). That symmetry — guaranteed by the causal mask — is the entire reason a next-token predictor trained in parallel can generate one token at a time without retraining.

### 12.7 One full cycle of *naive* inference — no KV cache, every number shown

Section 12.6 stopped at a shortcut: it *asserted* the next distribution and pointed at the KV cache as the reason you don't have to redo everything. This subsection does the opposite — it takes the honest, **cacheless** path and recomputes the *entire* forward pass on the extended sequence from scratch, exactly the way a first, un-optimized implementation would. This is "naive" inference: **hold nothing, reuse nothing, re-encode the whole prefix every step.** It is the $\mathcal{O}(T^2)$ baseline of Section 11.2 — the thing KV caching later deletes — and running it in full is the most direct way to *see* both why the redundant work exists and why removing it changes no output.

Greedy decoding in 12.6 appended `sat` (id 3) to `the cat sat`, so the model is now called on

$$
\text{ids} = [\,1,\,2,\,3,\,3\,]\quad(\text{the, cat, sat, sat}),\qquad T=4.
$$

Naive inference runs Sections 12.1–12.4 again, unchanged, on all **four** positions — not just the new one. We trace every step at $T=4$ and, at the end, check the two claims 12.6 made on credit: that positions 0–2 come back **bit-identical** to the $T=3$ pass (the wasted work), and that greedy again emits `sat`.

**The three weights §12 used but never printed.** The training walkthrough displayed the *outputs* of the attention output projection and the FFN (Sections 12.2f, 12.3b–d) without ever showing their weight matrices, and it only ever needed positional rows 0–2. To run position 3 we need all of them explicitly. Here they are — the residual-write $W_O$, the FFN up/down $W_1,W_2$, and the fourth positional row $W_p[3]$. (These complete the toy model's specification; multiplied against the Section 12.2–12.3 inputs they reproduce every number printed there.)

$$
W_p[3]=\big[\,0.10,\;0.00,\;-0.20,\;0.10\,\big]\qquad(\text{learned position-3 row, in the same small-constant style as rows 0–2})
$$

$$
W_O=\begin{bmatrix}
0.158&-0.008&0.049&-0.205\\
-0.016&0.195&-0.041&0.013\\
-0.079&0.115&0.224&0.028\\
-0.155&-0.109&0.044&0.188
\end{bmatrix}\;(4\times4),\qquad
W_2=\begin{bmatrix}
0.003&0.049&-0.001&-0.029\\
0.095&0.139&0.051&-0.028\\
-0.014&0.043&0.042&0.031\\
-0.069&0.125&0.072&0.030\\
0.003&0.049&-0.001&-0.029\\
-0.028&0.141&0.070&0.012\\
0.053&-0.008&-0.007&-0.011\\
-0.060&0.107&0.072&0.038
\end{bmatrix}\;(8\times4)
$$

$$
W_1=\begin{bmatrix}
0.050&-0.250&-0.050&0.225&0.050&-0.025&-0.125&0.125\\
0.150&0.050&-0.150&-0.075&0.150&-0.224&0.075&-0.075\\
-0.150&-0.050&0.150&0.025&-0.150&0.075&-0.025&0.125\\
-0.050&0.250&0.050&-0.175&-0.050&0.174&0.075&-0.175
\end{bmatrix}\;(4\times8)
$$

---

**(a) Input embedding + positional, all four positions (Section 12.1).** Look up rows $[1,2,3,3]$ of $W_e$ and rows $[0,1,2,3]$ of $W_p$ and add. Positions 0–2 are exactly as in 12.1; position 3 is `sat` (id 3) but now at absolute position 3, so it gets $W_p[3]$, **not** the $W_p[2]$ that the *same token* got at position 2 — the identical token lands at a different point in the stream:

$$
x_0=W_e[\text{ids}]+W_p[0{:}4]=
\begin{bmatrix}
0.500&-0.100&0.500&0.000\\
-0.200&0.300&0.200&0.100\\
0.000&0.300&-0.300&0.500\\
\mathbf{0.300}&\mathbf{0.100}&\mathbf{-0.600}&\mathbf{0.600}
\end{bmatrix}
$$

Worked, the new row 3 $=W_e[3]+W_p[3]=[0.20,0.10,-0.40,0.50]+[0.10,0.00,-0.20,0.10]=[0.300,0.100,-0.600,0.600]$ ✓. Compare row 2 (`sat` at position 2) $=[0.000,0.300,-0.300,0.500]$: **same token, different vector**, purely because the positional row differs — a concrete look at why "position is added, not concatenated" (Section 3.3).

**(b) Pre-LN #1 (Section 12.2a).** LayerNorm each row across its 4 features:

$$
\tilde x=\mathrm{LN}_1(x_0)=
\begin{bmatrix}
0.992&-1.172&0.992&-0.811\\
-1.603&1.069&0.534&0.000\\
-0.412&0.577&-1.402&1.237\\
\mathbf{0.453}&\mathbf{0.000}&\mathbf{-1.585}&\mathbf{1.132}
\end{bmatrix}
$$

Row 3 check: mean of $[0.3,0.1,-0.6,0.6]=0.1$, variance $=\tfrac14[(0.2)^2+0^2+(-0.7)^2+(0.5)^2]=0.195$, so $(0.3-0.1)/\sqrt{0.195}=0.453$, $(0.1-0.1)/\sqrt{0.195}=0.000$, etc. ✓ The top three rows are untouched from 12.2a.

**(c) Fused QKV (Section 12.2b) — and the first sighting of the wasted work.** Applying $W_Q,W_K,W_V$ (from 12.2b) to $\tilde x$:

$$
Q=\begin{bmatrix}
-0.081&-0.036&0.433&-0.559\\
-0.214&0.534&-0.534&0.000\\
0.223&-0.264&-0.198&0.709\\
\mathbf{0.317}&\mathbf{-0.521}&\mathbf{0.091}&\mathbf{0.657}
\end{bmatrix}\;
K=\begin{bmatrix}
-0.153&-0.397&-0.081&0.054\\
0.214&0.000&0.428&-0.267\\
-0.008&0.462&-0.074&0.148\\
\mathbf{-0.113}&\mathbf{0.430}&\mathbf{-0.294}&\mathbf{0.272}
\end{bmatrix}\;
V=\begin{bmatrix}
-0.478&0.415&-0.054&0.117\\
0.000&-0.374&0.053&0.321\\
0.586&-0.280&-0.049&-0.256\\
\mathbf{0.543}&\mathbf{-0.068}&\mathbf{-0.068}&\mathbf{-0.408}
\end{bmatrix}
$$

Worked, $Q_{3,0}=\tilde x_{\text{row }3}\!\cdot W_{Q,\text{col }0}=(0.453)(0.2)+(0.000)(0.1)+(-1.585)(0.0)+(1.132)(0.2)=0.317$ ✓.

**This is the load-bearing observation of the whole subsection.** Look at rows 0–2 of $K$ and $V$: they are **bit-for-bit** the $K,V$ from Section 12.2b. Nothing to their right can reach them — a causal token's projection depends only on itself and its left context (Section 11.2) — so re-deriving them was pure recomputation. We *just paid* $3/4$ of this matmul to reproduce numbers we already had. That reproduced block is exactly what the KV cache stores; only the bold **row 3** of $K,V$ is genuinely new.

$$
\boxed{\;K,V\text{ rows }0\text{–}2\ (T{=}4)\;=\;K,V\text{ rows }0\text{–}2\ (T{=}3)\quad\text{— identical, hence cacheable.}\;}
$$

**(d) Masked scaled dot-product, both heads, now $4\times4$ (Section 12.2d).** Heads split columns $0{:}2$ / $2{:}4$, scale by $\sqrt{d_k}=\sqrt2$, add the strictly-upper-triangular $-\infty$ mask. The mask is now $4\times4$: position 3 may attend to $\{0,1,2,3\}$, everyone else is unchanged.

$$
\text{scores}_0=\begin{bmatrix}
0.019&-\infty&-\infty&-\infty\\
-0.127&-0.032&-\infty&-\infty\\
0.050&0.034&-0.087&-\infty\\
\mathbf{0.112}&\mathbf{0.048}&\mathbf{-0.172}&\mathbf{-0.184}
\end{bmatrix}
\xrightarrow{\text{softmax}}
A_0=\begin{bmatrix}
1.000&0&0&0\\
0.476&0.524&0&0\\
0.350&0.345&0.305&0\\
\mathbf{0.291}&\mathbf{0.273}&\mathbf{0.219}&\mathbf{0.217}
\end{bmatrix}
$$

Worked, position 3 in head 0 (it attends to all four keys). Using the head-0 slices $Q_3^{(0)}=[0.317,-0.521]$ and $K_0^{(0)}=[-0.153,-0.397]$, $K_1^{(0)}=[0.214,0.000]$, $K_2^{(0)}=[-0.008,0.462]$, $K_3^{(0)}=[-0.113,0.430]$:

$$
\tfrac{Q_3^{(0)}\!\cdot K_0^{(0)}}{\sqrt2}=\tfrac{(0.317)(-0.153)+(-0.521)(-0.397)}{1.414}=0.112,\quad
\tfrac{Q_3^{(0)}\!\cdot K_2^{(0)}}{\sqrt2}=\tfrac{(0.317)(-0.008)+(-0.521)(0.462)}{1.414}=-0.172,
$$

and softmax over $[0.112,0.048,-0.172,-0.184]$ gives $[0.291,0.273,0.219,0.217]$ ✓ — position 3 spreads its attention over all four tokens, with no $-\infty$ term because it has no future. Rows 0–2 of $A_0$ are, again, identical to 12.2d: they never saw position 3, so the extra column changes nothing. Head 1 similarly:

$$
A_1=\begin{bmatrix}
1.000&0&0&0\\
0.548&0.452&0&0\\
0.352&0.279&0.369&0\\
\mathbf{0.248}&\mathbf{0.221}&\mathbf{0.260}&\mathbf{0.271}
\end{bmatrix}
$$

**(e) Weighted values, concat, output projection (Sections 12.2e–f).** $\text{head}_i=A_iV_i$, glued back to width 4, then $\times W_O$:

$$
\text{concat}=\begin{bmatrix}
-0.478&0.415&-0.054&0.117\\
-0.228&0.002&-0.005&0.209\\
0.011&-0.069&-0.022&0.036\\
\mathbf{0.107}&\mathbf{-0.058}&\mathbf{-0.033}&\mathbf{-0.077}
\end{bmatrix}
\xrightarrow{\;\cdot\,W_O\;}
\mathrm{MHSA}=\begin{bmatrix}
-0.096&0.066&-0.047&0.124\\
-0.068&-0.021&-0.003&0.086\\
-0.001&-0.020&0.000&0.003\\
\mathbf{0.032}&\mathbf{-0.008}&\mathbf{-0.003}&\mathbf{-0.038}
\end{bmatrix}
$$

Worked, the new concat row 3, head 0 part — weights $A_{0,\text{row }3}=[0.291,0.273,0.219,0.217]$ against head-0 value rows $V^{(0)}_0=[-0.478,0.415]$, $V^{(0)}_1=[0.000,-0.374]$, $V^{(0)}_2=[0.586,-0.280]$, $V^{(0)}_3=[0.543,-0.068]$:

$$
\text{head}_0[\text{row }3]=0.291[-0.478,0.415]+0.273[0.000,-0.374]+0.219[0.586,-0.280]+0.217[0.543,-0.068]=[0.107,-0.058]
$$

— the first two entries of concat-row 3; head 1's two entries $[-0.033,-0.077]$ are computed the same way. Then $\mathrm{MHSA}[\text{row }3]=\text{concat}[\text{row }3]\cdot W_O=[0.032,-0.008,-0.003,-0.038]$.

**(f) Residual add #1 (Section 12.2g).** Add onto the *un-normalized* stream $x_0$:

$$
a=x_0+\mathrm{MHSA}(\mathrm{LN}_1(x_0))=
\begin{bmatrix}
0.404&-0.034&0.453&0.124\\
-0.268&0.279&0.197&0.186\\
-0.001&0.280&-0.300&0.503\\
\mathbf{0.332}&\mathbf{0.092}&\mathbf{-0.603}&\mathbf{0.562}
\end{bmatrix}
$$

**(g) Pre-LN #2 → FFN(GELU) → residual add #2 (Section 12.3).** LayerNorm $a$, expand through $W_1$ to width 8, GELU, contract through $W_2$, add back. For the new position 3:

$$
\tilde a[\text{row }3]=\mathrm{LN}_2(a)_{\text{row }3}=[0.542,-0.008,-1.602,1.068]
$$
$$
(\tilde aW_1)[\text{row }3]=[0.212,\,0.211,\,-0.212,\,-0.105,\,0.212,\,0.055,\,0.051,\,-0.319]
$$
$$
\mathrm{GELU}[\text{row }3]=[0.124,\,0.123,\,-0.088,\,-0.048,\,0.124,\,0.029,\,0.027,\,-0.120]
$$
$$
\mathrm{FFN}[\text{row }3]=(\mathrm{GELU}\cdot W_2)[\text{row }3]=[0.025,\,0.010,\,-0.008,\,-0.019]
$$

Giving the end-of-block stream (rows 0–2 agree with Section 12.3e to within one unit in the last displayed digit — see the precision note below):

$$
x_1=a+\mathrm{FFN}(\mathrm{LN}_2(a))=
\begin{bmatrix}
0.340&0.031&0.506&0.163\\
-0.204&0.296&0.196&0.166\\
0.060&0.304&-0.307&0.472\\
\mathbf{0.357}&\mathbf{0.103}&\mathbf{-0.611}&\mathbf{0.542}
\end{bmatrix}
$$

(In real GPT-2 this is where you'd re-enter the block for layers $2\dots N$; the toy has one layer, so we go straight to the head.)

> **Precision note (read once, applies to (g)–(i)).** From this point on, rows 0–2 are computed from the *printed* (3-decimal) weights above, whereas Sections 12.3–12.4 displayed values computed from the author's full-precision originals. The two chains agree exactly through $\tilde x, Q, K, V, A$, concat, MHSA, and $a$ (those depend only on exactly-printed inputs), then drift by a few units in the third decimal through $\mathrm{LN}_2$ → FFN → `ln_f` — e.g. `ln_f` row 0 opens $0.446$ here vs $0.442$ in 12.4, because LayerNorm divides by a small standard deviation and amplifies last-digit differences. The drift washes out again at the softmax: the final $P$ rows 0–2 below reproduce Section 12.4's $P$ to every displayed digit. This is the Section-12 caveat made visible — and a useful micro-lesson in numerical sensitivity for free.

**(h) Final LayerNorm (Section 12.4).** Apply `ln_f`:

$$
x_N'=\mathrm{LN}_f(x_1)=
\begin{bmatrix}
0.446&-1.276&1.371&-0.540\\
-1.675&0.963&0.435&0.277\\
-0.246&0.586&-1.500&1.160\\
\mathbf{0.592}&\mathbf{0.012}&\mathbf{-1.618}&\mathbf{1.015}
\end{bmatrix}
$$

**(i) Weight-tied unembed → logits → softmax (Sections 12.4, 9).** $\text{logits}=x_N'\,W_e^\top\in\mathbb{R}^{4\times10}$, then row-wise softmax:

$$
\text{logits}=
\begin{bmatrix}
-0.510&0.835&-0.316&-0.857&0.546&0.229&-0.062&0.725&-1.005&-0.458\\
0.065&-0.872&0.947&-0.274&-0.692&0.575&-0.744&0.136&0.183&0.989\\
0.591&-0.574&-0.108&1.189&-0.168&-0.632&-0.088&-0.403&0.718&0.414\\
\mathbf{0.528}&\mathbf{-0.091}&\mathbf{-0.598}&\mathbf{1.274}&\mathbf{0.215}&\mathbf{-0.906}&\mathbf{0.257}&\mathbf{-0.406}&\mathbf{0.551}&\mathbf{-0.092}
\end{bmatrix}
$$

Worked, the new row's logit for `sat` (id 3), $x_{N,\text{row }3}'\cdot W_{e,\text{row }3}$:

$$
(0.592)(0.20)+(0.012)(0.10)+(-1.618)(-0.40)+(1.015)(0.50)=0.1184+0.0012+0.6472+0.5075=1.274\ \checkmark
$$

Softmax of the position-3 row $z=[0.528,-0.091,-0.598,1.274,0.215,-0.906,0.257,-0.406,0.551,-0.092]$, done the same three-step way as Section 12.4:

*exponentiate* — $e^{z}=[1.696,\,0.913,\,0.550,\,3.575,\,1.240,\,0.404,\,1.293,\,0.666,\,1.735,\,0.912]$;

*sum* — $\sum_k e^{z_k}=12.984$;

*divide* — giving the full per-position distribution (rows 0–2 reproduce Section 12.4's $P$ to the last displayed digit; row 3 is new):

$$
P=
\begin{bmatrix}
0.054&0.209&0.066&0.038&0.156&0.114&0.085&0.187&0.033&0.057\\
0.085&0.033&0.204&0.060&0.040&0.141&0.038&0.091&0.095&0.213\\
0.138&0.043&0.069&0.251&0.065&0.041&0.070&0.051&0.157&0.116\\
\mathbf{0.131}&\mathbf{0.070}&\mathbf{0.042}&\mathbf{0.275}&\mathbf{0.095}&\mathbf{0.031}&\mathbf{0.100}&\mathbf{0.051}&\mathbf{0.134}&\mathbf{0.070}
\end{bmatrix}
$$

**(j) Keep only the last row, decode (Section 11.2).** Inference discards rows 0–2 (they predict tokens we already have) and reads row 3 — the distribution for the token *after* `the cat sat sat`:

$$
P_{\text{pos }3}=[\,0.131,\;0.070,\;0.042,\;\mathbf{0.275},\;0.095,\;0.031,\;0.100,\;0.051,\;0.134,\;0.070\,]
$$

Greedy takes the argmax → id **3 = `sat`** (probability $0.275$). Append it → $[1,2,3,3,3]$ and, naively, you would now run this *entire pass again* at $T=5$, re-deriving rows 0–3 for the third time. (The content is gibberish — random weights — but the machine is exactly production GPT-2.)

---

**What this one cycle exposes.** Tallying the pass by position makes the $\mathcal{O}(T^2)$ waste literal. Of the four rows we pushed through LayerNorm, QKV, attention, and the FFN, **three (positions 0–2) came out identical to values we already computed at $T=3$** — verified at every stage above, in bold-vs-plain. Only the one bold row was new. Generating a single token cost a **four-row** forward pass to extract **one row** of new information:

| Quantity | Naive (this section) | With KV cache (Section 11.2) |
|---|---|---|
| Rows through QKV / attention / FFN, this step | all $T=4$ | only the new $1$ |
| $K,V$ rows computed | $4$ (3 recomputed, bold row new) | $1$ (rest read from cache) |
| Attention score entries | full $4\times4$ lower triangle | one new row vs. $4$ cached keys |
| Redundant work | rows 0–2, i.e. $3/4$ | none |
| Per-step cost | $\mathcal{O}(T^2)$ over a generation | $\mathcal{O}(T)$ |

And the equality in the boxed line of step (c) is *why the optimization is free*: because a causal token's $K,V$ never depend on anything to its right (the mask of Section 6.2, the immutability noted in Section 11.2), the rows the cache reuses are provably the same rows the naive pass recomputes — bit-for-bit, as we just watched. KV caching is not an approximation that trades accuracy for speed; it is the removal of arithmetic whose answer was already known. Naive inference is the baseline that makes that guarantee visible — and it is the *only* thing you must change to go from this chapter's math to a fast generator: same weights, same operations, same outputs, just stop recomputing the past.

### 12.8 The same cycle *with* the KV cache — and a proof it changes nothing

Section 12.7 established the claim informally: rows 0–2 came out identical, so recomputing them was waste. Now we *act* on that — run the identical generation step with a KV cache — and then **prove**, number by number, that the cached path yields the same `P_pos3` as the naive path, not approximately but exactly. The whole point of the cache rests on this equality; if it did not hold, caching would be a bug, not an optimization.

A cached generator splits the work into two phases:

- **Prefill** — run the prompt `[1,2,3]` once (this *is* Sections 12.1–12.4) and, as a side effect, **store** every position's $K,V$. That store is the cache.
- **Decode** — for each new token, embed and project **only that one token**, append its $K,V$ to the cache, and attend its single query against the whole cached stack. No earlier row is ever recomputed.

**The cache after prefill.** Running `[1,2,3]` produced, in Section 12.2b, exactly these $K$ and $V$ — we simply keep them instead of discarding them:

$$
K_{\text{cache}}=\begin{bmatrix}
-0.153&-0.397&-0.081&0.054\\
0.214&0.000&0.428&-0.267\\
-0.008&0.462&-0.074&0.148
\end{bmatrix},\qquad
V_{\text{cache}}=\begin{bmatrix}
-0.478&0.415&-0.054&0.117\\
0.000&-0.374&0.053&0.321\\
0.586&-0.280&-0.049&-0.256
\end{bmatrix}\quad(\text{rows }0,1,2)
$$

**Decode step: generate the token after `the cat sat sat`.** Greedy in 12.6 picked `sat` (id 3) for position 3. The cached decode processes **only** that one token — a **single-row** pass, $T_{\text{new}}=1$, against a growing cache.

**(a) Embed only the new token** (position-3 row, nothing else):

$$
x_0^{\text{new}}=W_e[3]+W_p[3]=[0.20,0.10,-0.40,0.50]+[0.10,0.00,-0.20,0.10]=[0.300,0.100,-0.600,0.600]
$$

**(b) Pre-LN + project the one row** to its query/key/value (LayerNorm and the QKV matmul are per-position, so a single row goes through untouched):

$$
\tilde x^{\text{new}}=[0.453,0.000,-1.585,1.132],\quad
Q_3=[0.317,-0.521,0.091,0.657],\;
K_3=[-0.113,0.430,-0.294,0.272],\;
V_3=[0.543,-0.068,-0.068,-0.408]
$$

**(c) Append to the cache.** $K_3,V_3$ are pushed onto the stored stack, which now spans all four positions — the top three rows are *read straight from memory*, never recomputed:

$$
K_{\text{cache}}\leftarrow\begin{bmatrix}K_{\text{cache}}\\ K_3\end{bmatrix}=\begin{bmatrix}
-0.153&-0.397&-0.081&0.054\\
0.214&0.000&0.428&-0.267\\
-0.008&0.462&-0.074&0.148\\
\mathbf{-0.113}&\mathbf{0.430}&\mathbf{-0.294}&\mathbf{0.272}
\end{bmatrix},\quad
V_{\text{cache}}\leftarrow\begin{bmatrix}
-0.478&0.415&-0.054&0.117\\
0.000&-0.374&0.053&0.321\\
0.586&-0.280&-0.049&-0.256\\
\mathbf{0.543}&\mathbf{-0.068}&\mathbf{-0.068}&\mathbf{-0.408}
\end{bmatrix}
$$

**(d) Attend the single new query against the full cache — and note there is no mask.** Only $Q_3$ participates; it scores against all four cached keys, per head, scaled by $\sqrt2$:

$$
\text{head 0: }\;\frac{Q_3^{(0)}K_{\text{cache}}^{(0)\top}}{\sqrt2}=[0.112,\,0.048,\,-0.172,\,-0.184]\;\xrightarrow{\text{softmax}}\;[0.291,\,0.273,\,0.219,\,0.217]
$$
$$
\text{head 1: }\;[0.020,\,-0.097,\,0.064,\,0.107]\;\xrightarrow{\text{softmax}}\;[0.248,\,0.221,\,0.260,\,0.271]
$$

**Why no $-\infty$ mask here** (the subtle correctness point): the naive pass needed the causal mask to stop position 3 from seeing positions $>3$. In the cache there simply *are* no such positions — the cache only ever holds keys for tokens already emitted, i.e. positions $\le$ the current one. The "no peeking ahead" law of Section 6.2 is enforced **structurally, by the contents of the cache**, rather than by adding $-\infty$. This is the operational face of the immutability from Section 11.2: a past $K,V$ never changes, and no future $K,V$ exists yet, so the cache is exactly the set of keys the mask would have kept.

**(e) Finish the one row** — weighted values → concat → output projection → residual → FFN → residual → `ln_f` → tied unembed → softmax, all applied to this single row:

$$
\text{concat}_3=[0.107,-0.058,-0.033,-0.077]\xrightarrow{W_O}\mathrm{MHSA}_3=[0.032,-0.008,-0.003,-0.038]\xrightarrow{+\,x_0^{\text{new}}}a_3=[0.332,0.092,-0.603,0.562]
$$
$$
x_{1,3}=[0.357,0.103,-0.611,0.542]\xrightarrow{\text{ln\_f}}[0.592,0.012,-1.618,1.015]\xrightarrow{W_e^\top,\,\text{softmax}}
$$
$$
P_{\text{pos }3}^{\text{cached}}=[\,0.131,\;0.070,\;0.042,\;\mathbf{0.275},\;0.095,\;0.031,\;0.100,\;0.051,\;0.134,\;0.070\,]
$$

Greedy → id **3 = `sat`** (probability $0.275$). We then append $K_3,V_3$ (already done) and, for the *next* token, repeat the decode with a five-deep cache — never re-touching positions 0–3.

---

**The proof that the cache changes nothing.** Set the cached row 3 beside the naive row 3 from Section 12.7 (its bold entries), quantity by quantity:

| Quantity (position 3) | Naive pass §12.7 | Cached decode §12.8 | Equal? |
|---|---|---|---|
| $x_0$ row | $[0.300,0.100,-0.600,0.600]$ | $[0.300,0.100,-0.600,0.600]$ | ✓ |
| $Q_3,K_3,V_3$ | as in 12.7c | as in 12.8b | ✓ |
| head-0 scores / weights | $[0.112,0.048,-0.172,-0.184]$ / $[0.291,0.273,0.219,0.217]$ | identical | ✓ |
| $\mathrm{MHSA}_3$ | $[0.032,-0.008,-0.003,-0.038]$ | $[0.032,-0.008,-0.003,-0.038]$ | ✓ |
| $x_{1,3}$ | $[0.357,0.103,-0.611,0.542]$ | $[0.357,0.103,-0.611,0.542]$ | ✓ |
| logits row 3 | $[0.528,-0.091,\dots,-0.092]$ | identical | ✓ |
| $P_{\text{pos }3}$ | $[0.131,\dots,\mathbf{0.275},\dots,0.070]$ | $[0.131,\dots,\mathbf{0.275},\dots,0.070]$ | ✓ |

Measured directly, the maximum absolute difference between the two `P_pos3` vectors is **exactly $0$** — not "within rounding," identically zero, because the two paths execute the *same arithmetic on the same inputs*. The equality is not luck; it follows from three facts already established:

1. **The new-token inputs are self-contained.** $x_0^{\text{new}}$, its LayerNorm, and $Q_3,K_3,V_3$ depend only on token id 3 and position 3 — never on any other position — so the single-row computation is bit-identical to naive row 3.
2. **The cached keys/values equal the recomputed ones.** Section 12.7(c) verified $K,V$ rows 0–2 at $T{=}4$ equal those at $T{=}3$; the cache *is* those rows. So $Q_3$ attends against the same four $(K,V)$ pairs in both paths — hence the same four scores, the same softmax, the same head outputs. (The absent mask changes nothing: at $T{=}4$ the naive mask kept exactly positions $0$–$3$, which is the entire cache.)
3. **Everything downstream is position-wise.** Concat, $W_O$, both residual adds, LayerNorm, the FFN, `ln_f`, and the unembed all act on one position at a time (only attention mixes across positions, Section 2). So computing row 3 alone yields the same row 3 as computing it inside a batch of four.

$$
\boxed{\;P_{\text{pos }3}^{\text{cached}}=P_{\text{pos }3}^{\text{naive}}\ \text{exactly}\quad\Longrightarrow\quad\text{identical token, at }\tfrac14\text{ the row-work this step.}\;}
$$

**What the cache actually bought.** The naive step (Section 12.7) pushed **four** rows through QKV, attention, and the FFN to read off **one** new distribution. The cached step pushed **one** row and reused three from memory:

| Work, this decode step | Naive §12.7 | KV cache §12.8 |
|---|---|---|
| Rows embedded + LayerNormed + QKV-projected | $4$ | $\mathbf{1}$ |
| New $K,V$ rows computed | $4$ (3 redundant) | $\mathbf{1}$ |
| Attention: query rows × key columns | $4\times4$ (masked) | $\mathbf{1\times4}$ (no mask) |
| FFN rows | $4$ | $\mathbf{1}$ |
| Output rows kept | $1$ (rows 0–2 discarded) | $\mathbf{1}$ |
| Result | $P_{\text{pos }3}$ | **same** $P_{\text{pos }3}$ |

Over a full generation the saving compounds: naive decoding is $\mathcal{O}(T^2)$ (every step re-encodes the whole prefix), cached decoding is $\mathcal{O}(T)$ per step (one new row against a cache of length $T$). And it costs nothing in quality — the boxed equality guarantees the cached generator emits the identical sequence the naive one would, token for token. That guarantee is a *direct gift of the causal mask*: because position $t$'s representation is provably independent of everything to its right (Section 6.2), its $K,V$ are safe to freeze and reuse forever. The same structural decision that made training parallel (Section 1.3) is what makes fast inference exact.

### 12.9 Epilogue to the cache: why managing it at scale is a nightmare — and how vLLM's PagedAttention fixed it

Sections 12.7–12.8 told the KV-cache story for **one** sequence with **unlimited** memory: append two little rows per layer per step, never look back. That story is complete as *mathematics*. As *engineering* it is barely the opening sentence, because a production server does not run one sequence — it runs **hundreds or thousands concurrently**, on a GPU whose memory is finite, and every one of those sequences owns a cache that **grows by an unpredictable amount at an unpredictable rate and dies at an unpredictable time**. Managing that zoo turned out to be *the* central systems problem of LLM serving, and the 2023 solution — **PagedAttention**, the idea inside **vLLM** (Kwon et al., SOSP 2023) — is a beautiful case of stealing a fifty-year-old operating-systems idea. This section is a conceptual walk from "what is a tensor in GPU memory, really" to "why the obvious cache layout collapses" to "how paging rescues it." No kernel code — just the ideas, with the same insistence on concrete numbers as the rest of the chapter.

#### 12.9.1 What "a tensor in HBM" actually is

A modern accelerator's main memory is **HBM** — *High-Bandwidth Memory* — stacks of DRAM dies sitting millimeters from the processor die on a shared substrate. Two properties of it drive everything below, and it pays to hold them apart:

- **Capacity** — how many bytes fit. Tens of GB (an A100 has 40 or 80 GB).
- **Bandwidth** — how many bytes per second can *move* between HBM and the compute units. On the order of 1.5–3 TB/s.

To the software, HBM is one **flat array of bytes** with addresses $0,1,2,\dots$ — there is no "shape" in the hardware. A *tensor* is nothing but a lightweight descriptor: a **base address**, a **shape**, and **strides** that turn an index tuple into an address. Our familiar per-layer key cache $K\in\mathbb{R}^{T\times d}$, stored row-major (row after row, the way this chapter always printed it), lives as one **contiguous run** of $T\cdot d$ scalars, and element $K[t,i]$ sits at

$$
\text{addr}(K[t,i]) = \text{base} + (t\cdot d + i)\cdot(\text{bytes per scalar}).
$$

That formula is the crux of the whole section. It is why contiguity *matters*: a matmul or attention routine reading $K$ doesn't "look tokens up" — it does exactly this arithmetic, marching a pointer through memory in long sequential runs, which is also the access pattern DRAM rewards with its full bandwidth. And it is why **appending is only cheap if someone reserved room**: token $t{+}1$'s key *must* land at $\text{base}+(t{+}1)\cdot d\cdot\text{bytes}$ — the very next bytes. If those bytes belong to something else, your options are ugly: allocate a bigger slab elsewhere and **copy the whole cache** (do that every step and you've re-created Section 12.7's quadratic waste, now in memory traffic), or accept that the cache must have been given its worst-case room **up front**. Hold that fork in the road; it becomes Section 12.9.3.

One more piece of the picture: GPU memory is handed out by an **allocator** in contiguous slabs. Like any allocator managing variable-sized requests, it can suffer **fragmentation** — free bytes that exist but are unusable: *internal* (a slab is bigger than what it actually stores) and *external* (free space shattered into gaps between live slabs, none big enough for the next request). File those two words away too.

#### 12.9.2 Why batch size is the whole game

Here is the economic fact that turns cache management from housekeeping into destiny. Consider serving a 13B-parameter model in fp16 and generating **one token for one sequence**. The decode step must multiply the single new token's vector through essentially *every* weight matrix — and the weights don't live in on-chip caches; all **26 GB** of them must stream from HBM through the compute units, to perform only about **2 FLOPs per parameter** (one multiply, one add). Two FLOPs per two-byte parameter is an *arithmetic intensity* of ~1 FLOP/byte. An A100 can sustain roughly **200 FLOPs per byte** of HBM traffic before compute becomes the bottleneck. At batch size 1, then, the GPU computes at **under 1% of its peak** — it is a 312-TFLOPS machine spending its life waiting on memory, a regime called **memory-bandwidth-bound**.

The fix is classic: **batching**. Serve $B$ sequences at once and each streamed weight byte is *reused* $B$ times — one load of $W_{qkv}$ feeds all $B$ sequences' projections. Throughput rises almost linearly in $B$ until you finally approach the compute roof (around $B\sim200$ on the arithmetic above). So:

$$
\text{serving throughput} \;\propto\; B
\qquad\text{and}\qquad
B \;=\; \Big\lfloor \frac{\text{HBM left after weights}}{\text{memory per sequence's KV cache}} \Big\rfloor.
$$

The weights are a fixed cost; **the KV cache is the marginal cost of each additional sequence**. Every byte of cache wasted is a slot of batch — hence a slice of throughput — burned. *KV-cache memory efficiency isn't a nicety; it is the throughput dial.*

#### 12.9.3 How big the cache is, and how awkwardly it grows

The cache stores, per token, per layer, one $d$-wide key row and one $d$-wide value row (Section 12.8c):

$$
\text{bytes per token} \;=\; \underbrace{2}_{K\text{ and }V}\times\; n_{layer}\times d \times(\text{bytes per scalar}).
$$

For our chapter's model, GPT-2 small in fp16: $2\times12\times768\times2 = 36{,}864$ B — **36 KiB per token**, hence a full 1024-token context costs **36 MiB per sequence**. Quaint. Now the 13B-class model (40 layers, $d=5120$): $2\times40\times5120\times2 = 800$ **KiB per token** — a single 2048-token sequence owns **~1.6 GB** of cache. On the A100-40GB serving that model, the weights take 26 GB; after activations and workspace maybe **12 GB remain for all caches combined**. That is the budget from which batch size — i.e., throughput — is carved.

And the object we must fit into it is memory-management hell by construction, on three counts:

1. **It grows.** One row per layer per step (Section 12.8c), append-only, for as long as the sequence lives.
2. **Its final size is unknowable.** Generation stops at $\langle\texttt{endoftext}\rangle$ *whenever the model happens to emit it* (Section 11.2) — 10 tokens or 2,000; the server cannot know in advance.
3. **Lifetimes churn.** Requests arrive and finish continuously; caches of every different length are born and freed all the time.

A growing, unknown-length, contiguity-demanding object (Section 12.9.1) with high churn: this is precisely the workload that breaks naive allocators.

#### 12.9.4 The naive solution — preallocate the full context — and how it kills performance

The pre-vLLM serving stacks took the fork Section 12.9.1 offered and chose the "reserve worst case up front" branch, because the attention code demanded contiguous $K$ and $V$ per sequence. Concretely: **when a request arrives, allocate its cache slab for the maximum possible length** — the full $n_{ctx}$ (or the request's max-tokens) — and let the sequence fill it left to right.

```
one sequence's preallocated slab (max length 2048), after generating to token 511:

  ┌───────────────┬─────────────────────────────────────────────────┐
  │ tokens 0..511 │ ///////////// reserved, never used ///////////// │
  │  (live K,V)   │ /////////////// tokens 512..2047 //////////////// │
  └───────────────┴─────────────────────────────────────────────────┘
        25% used                        75% dead weight
```

The waste comes in three distinct flavors, and it is worth separating them the way the vLLM paper did:

- **Internal fragmentation — the never-used tail.** The sequence above stopped at 512 tokens but reserved 2048; the remaining $1536\times800\,\text{KiB}\approx1.2$ GB were held for its entire lifetime and never touched. Since output length is unknowable (12.9.3), *every* sequence must be provisioned like the longest one.
- **Reservation slack — the not-yet-used middle.** Even the slots a sequence *will eventually* use sit idle-but-locked for most of its life: slot 2000 is reserved from step 0 but holds nothing until step 2000. No other sequence may use it in the meantime.
- **External fragmentation — the gaps between slabs.** Different requests reserve different-sized slabs; as they churn (12.9.3), the allocator's memory becomes a moth-eaten patchwork of free ranges, many too small to seat the next arrival even though the *total* free space would suffice.

Measured on real traces, the combined effect was brutal: existing serving systems kept only **20–38% of KV-cache memory actually filled with token data**. Read that against Section 12.9.2's equation: wasting ~70% of cache memory divides your feasible batch — and therefore your throughput — **by roughly three**, before any other inefficiency is counted. On our A100 example, naive full-context slabs at 1.6 GB each seat just **~7 concurrent sequences** in the 12 GB budget, while the *live tokens* those sequences actually hold might fit in a quarter of that. The GPU idles at a few percent of peak, not because the math is slow, but because the bookkeeping of one growing tensor per user is squandering the memory that batch size is made of.

(And the other fork — grow-on-demand by copying into ever-bigger slabs — is no rescue: it re-introduces quadratic memory traffic, doubles peak usage at every regrow, and *still* fragments externally. The dilemma is real; what's wrong is an assumption both branches share.)

#### 12.9.5 PagedAttention: virtual memory for the KV cache

The shared assumption is: *a sequence's cache must be one contiguous slab*. Operating systems faced this exact problem in the 1960s — processes demanding "contiguous" memory of unpredictable, growing size, fragmenting physical RAM — and solved it with **virtual memory and paging**: give each process the *illusion* of contiguity, back it with scattered fixed-size physical frames, and keep a per-process **page table** translating one to the other. PagedAttention is that idea, transplanted organ-for-organ:

| Operating system | vLLM |
|---|---|
| process | sequence (one request) |
| virtual address space | the sequence's logical cache: slots $0,1,2,\dots$ |
| physical frame (fixed size) | **physical block**: K,V storage for a fixed number of tokens (e.g. **16**) |
| page table | **block table**: per-sequence list *logical block → physical block* |
| page fault → allocate a frame | cache full → allocate one new block |

Mechanically: chop every sequence's cache into **logical blocks** of 16 consecutive tokens. Physical HBM is pre-carved into a pool of identical **physical blocks**. A sequence's block table maps one to the other, and the physical blocks can live *anywhere*:

```
sequence A: "the cat sat on ..."  (37 tokens = 3 blocks: 16+16+5)
sequence B: "once upon a ..."     (20 tokens = 2 blocks: 16+4)

     logical view (per sequence)              physical block pool in HBM
   A: [ blk0 ][ blk1 ][ blk2· ]            ┌─────┬─────┬─────┬─────┬─────┬─────┐
       │       │       │                   │ B:1 │ A:0 │free │ A:2 │ B:0 │ A:1 │
   table A: 0→#1, 1→#5, 2→#3               └─────┴─────┴─────┴─────┴─────┴─────┘
   table B: 0→#4, 1→#0                        #0    #1    #2    #3    #4    #5
                                            (interleaved, non-contiguous — and fine)
```

Now walk the three wastes of 12.9.4 through this design:

- **Never-used tail: gone.** Nothing is allocated ahead of need. A sequence that stops at 512 tokens allocated 32 blocks, not 2048 slots' worth. Allocation happens **on demand**: when token 513 arrives and block 32 is full, the engine grabs *one* free block from the pool and appends its id to the table — the OS's page-fault reflex.
- **Reservation slack: gone**, same reason — slot 2000 simply does not exist until step ~2000.
- **External fragmentation: structurally impossible.** Every block is the same size, so any free block satisfies any request. There are no awkward gaps because there are no awkward sizes — exactly why OSes chose fixed-size pages.

The only waste left is the **partially-filled last block**: on average ~8 of 16 token-slots, per sequence — versus the *hundreds to thousands* of slots the naive scheme wasted. Waste per sequence is now **bounded by a constant** (one block) instead of scaling with $n_{ctx}$; measured utilization went from 20–38% to **~96%**.

**What the attention computation itself must change** — conceptually one thing. Section 12.8(d)'s decode attends the single query $Q_t$ against "the cached $K$ stack," which the naive kernel reads with the one-shot address arithmetic of 12.9.1 — possible only because the stack was contiguous. The paged kernel instead takes the **block table as an extra input** and does the OS's two-step address translation: for logical block $j$, look up its physical id $p=\text{table}[j]$, jump the pointer to block $p$'s base, and read 16 K-rows; compute those 16 dot products; accumulate softmax terms; next $j$. Attention, remember, is *order-respecting but location-indifferent* — the softmax cares which token a score belongs to (position was baked into the residual stream back in Section 12.1), not where its bytes sleep. Two design points make the indirection nearly free:

- **The overhead is one integer lookup per 16 tokens**, amortized to noise against the 16 × $d$-wide row loads it steers — and decode was bandwidth-bound anyway (12.9.2), so a little extra address logic hides completely behind the memory traffic.
- **Blocks are internally contiguous**, so *within* a block the hardware still sees the long sequential reads DRAM rewards. That is the block-size dial: too small (say 1 token) and you're doing scattered reads plus a huge table; too large (say 1024) and you're crawling back toward internal fragmentation. Sixteen-ish is the sweet spot where both pathologies are negligible.

#### 12.9.6 The bonus round: pages can be *shared*

Virtual memory came with a famous free gift — multiple processes mapping the same physical page — and the transplant inherits it. Since the block table is just indirection, **two sequences may point at the same physical block**, with a reference count. Immediately profitable, because serving is full of *identical prefixes*:

- **Parallel sampling.** "Give me 5 completions of this prompt": all 5 sequences map the *same* prompt blocks — stored once, refcount 5 — and only their divergent continuations own private blocks.
- **Beam search.** Beams share almost their entire past by construction; paging shares those blocks instead of copying whole caches, cutting cache memory by up to ~half on beam workloads.
- **Shared system prompts.** A fixed instruction prefix used by every request can be cached once, permanently.

Writes are handled with the OS's own trick, **copy-on-write**: blocks are append-only except the last (Section 12.8c — past $K,V$ are *immutable*), so sharing is trivially safe for full blocks; when a sequence must append into a *shared*, partially-filled block, the engine copies **that one 16-token block** (not the megabytes of cache), decrements the refcount, and appends into the private copy.

Paging even rehabilitates **eviction**. When the pool runs dry mid-flight, the scheduler can preempt a sequence and reclaim its blocks — either *swapping* them to CPU RAM and back, or simply **dropping them and recomputing later**: the entire cache can be rebuilt in one parallel prefill pass over the tokens so far, because — full circle — that is exactly the training-mode forward pass of Section 12.5, and Section 12.8 proved it reproduces the cache bit-for-bit. Cheap, safe restarts are a corollary of the causal mask.

#### 12.9.7 The payoff, and the moral

Stacking it up: near-zero waste (≈96% utilization) → 2–5× more sequences resident in the same HBM → correspondingly larger batches feeding a bandwidth-starved GPU (12.9.2) → **2–4× higher serving throughput than the best prior systems at equal latency** (and an order of magnitude over naive per-request serving), from *pure memory bookkeeping* — not one weight, logit, or probability differs, by the same immutability argument that made Section 12.8's cache exact. Practically every serving stack since has adopted paged KV management, and the architectural lineage in Section 18 attacks the same bottleneck from the other side: MQA/GQA shrink $2\times n_{layer}\times d$ itself by sharing $K,V$ across heads, shrinking the very rows being paged.

> **The takeaway, in the spirit of Section 4.** The residual stream was the right mental model for the *math* of GPT-2; for *serving* it, the right mental model is that **each request is a little process whose address space is its KV cache**. Once you see that, fifty years of operating-systems wisdom — paging, page tables, copy-on-write, swapping — applies verbatim, and the "nightmare" dissolves into the one problem computer science has already solved most thoroughly. The enabling deep fact is GPT-2's own: causal attention makes every cache entry **write-once, read-many** (Sections 6.2, 11.2, 12.8) — and write-once data is exactly what pages, shares, and swaps safely.

---

## 13. The model family and a worked parameter count

GPT-2 shipped as four sizes, all sharing $d_k=64$ and the same block design — they differ only in depth, width, and head count:

| Name | $n_{layer}$ | $d$ ($n_{embd}$) | $h$ | $n_{ctx}$ | Params |
|---|---|---|---|---|---|
| Small  | 12 | 768  | 12 | 1024 | ~124 M |
| Medium | 24 | 1024 | 16 | 1024 | ~355 M |
| Large  | 36 | 1280 | 20 | 1024 | ~774 M |
| XL     | 48 | 1600 | 25 | 1024 | ~1558 M |

(The original paper rounds these to 117M / 345M / 762M / 1542M; the numbers above are the careful counts.)

### 13.1 Counting the 124M model from scratch

With $|V|=50257$, $d=768$, $n_{ctx}=1024$, $n_{layer}=12$, $d_{ff}=4d=3072$:

**Embeddings (shared with the output via tying):**
$$
\underbrace{50257\times 768}_{W_e = 38{,}597{,}376} + \underbrace{1024\times 768}_{W_p = 786{,}432} = 39{,}383{,}808
$$

**Per block:**
- two LayerNorms: $2\times(2\times768) = 3{,}072$
- attention QKV (fused) + bias: $768\times 2304 + 2304 = 1{,}771{,}776$
- attention output proj + bias: $768\times 768 + 768 = 590{,}592$
- FFN up + bias: $768\times 3072 + 3072 = 2{,}362{,}368$
- FFN down + bias: $3072\times 768 + 768 = 2{,}360{,}064$

$$
\text{block} = 3072 + 1{,}771{,}776 + 590{,}592 + 2{,}362{,}368 + 2{,}360{,}064 = 7{,}087{,}872
$$

**Twelve blocks:** $12 \times 7{,}087{,}872 = 85{,}054{,}464$

**Final LayerNorm:** $2\times 768 = 1{,}536$

**Output projection:** $0$ (tied to $W_e$).

$$
\boxed{\,39{,}383{,}808 + 85{,}054{,}464 + 1{,}536 = 124{,}439{,}808 \approx 124\text{M}\,}
$$

This is the famous "124M" — and notice that nearly a third of it lives in the embedding table, which is why weight tying matters so much at this scale.

### 13.2 The scaling result

The reason there are four sizes is the empirical finding running through the GPT line: holding architecture fixed and scaling depth/width/data **monotonically improves** the language-modeling loss, with no sign of saturation at these sizes. GPT-2 was the existence proof; the scaling-law papers that followed made it quantitative. Architecturally, the takeaway is that the decoder-only block is a *scalable primitive* — you grow capability by stacking more of the same, which is far harder to do with the warmup-fragile Post-LN encoder–decoder.

---

## 14. What changed from GPT-1 (for lineage)

GPT-1 (2018) already established decoder-only LM pretraining, but used **Post-LN** like the 2017 paper. GPT-2's deltas over GPT-1 are exactly the stabilizers in this document: **move LayerNorm to Pre-LN**, **add the final `ln_f`**, **apply the $1/\sqrt N$ residual init**, plus the **byte-level BPE** vocabulary and a larger context (1024 vs 512) and much greater scale. In other words, GPT-1 proved the *paradigm*; GPT-2 proved it *scales*, and the changes that made it scale are architectural-numerical, not conceptual.

---

## 15. Why decoder-only won — the architectural philosophy

Pulling the threads together:

1. **One pattern, maximal reuse.** By keeping only masked self-attention, every parameter is spent on the same operation at every layer and every position. There is no encoder/decoder asymmetry to balance, no cross-attention bridge to tune. Uniformity is what makes the block a clean scaling primitive.

2. **Conditioning is free.** Because context and continuation share one causal stream, any task expressible as "given this text, produce that text" is *already* in-distribution — no architectural change, no task-specific head. This is the seed of in-context learning that GPT-3 would make famous.

3. **The training signal is dense.** Every token is a label. A length-$T$ sequence yields $T$ supervised predictions per forward pass — far more signal per example than seq2seq's single target sequence. The causal mask is what makes this density free of leakage.

4. **The residual stream is the real object.** Read Sections 4, 5, and 8 together: Pre-LN keeps the stream's gradient path clean, `ln_f` keeps its final scale clean, and $1/\sqrt N$ keeps its forward variance clean. The 2017 Transformer's depth limit was largely a residual/normalization problem; solve that, and depth — hence capability — is unlocked.

The 2017 paper's thesis was "attention is all you need." GPT-2's implicit thesis is sharper: **a deep stack of *masked* self-attention, trained to predict the next token, is all you need** — the encoder, the cross-attention, and the very idea of a fixed task were scaffolding you can remove.

---

## 16. Side-by-side summary

| Aspect | 2017 Transformer (decoder side) | GPT-2 |
|---|---|---|
| Macro-structure | Encoder + Decoder | Decoder-only |
| Attention types | self (bi), self (causal), cross | causal self only |
| Objective | $P(\text{target}\mid\text{source})$ | $P(w_t\mid w_{<t})$ |
| LayerNorm placement | Post-LN (after residual add) | Pre-LN (before sub-layer) + final `ln_f` |
| Positional info | fixed sinusoidal | learned table, max 1024 |
| FFN activation | ReLU | GELU (tanh approx.) |
| Residual init | standard | $\times\,1/\sqrt N$ on residual writes |
| Output embedding | tied to input (shared pre-softmax weights) | tied to input $W_e^\top$ — *same idea, not a GPT-2 novelty* |
| Tokenizer | word/sub-word, fixed | byte-level BPE, $|V|=50257$, no UNK |
| Depth shipped | 6 | 12 / 24 / 36 / 48 |
| Training warmup | essential (Post-LN fragility) | far less critical (Pre-LN stability) |

---

## 17. The Prefix LM — the fourth door, worked end to end

Section 1.4 presented the 2018–19 architecture space as three doors: encoder-only (sees everything, cannot speak), encoder–decoder (speaks across an imposed boundary), decoder-only (speaks, sees only the past). There was always a **fourth door**, quieter than the others: keep the *single* stack and the *single* stream of the decoder-only design, but let the mask be smarter. If the sequence naturally splits into a **prompt you were given** and a **continuation you must produce**, then nothing about honesty requires the prompt tokens to hide from each other — the prompt is fully known before generation begins, exactly like the encoder's source in Section 1.1. Only the *continuation* needs causality. So:

> **Prefix LM:** one decoder-only tower, one token stream of length $T$ with a designated **prefix length $P$**; attention is **bidirectional inside the prefix** (positions $0\dots P{-}1$ all see each other) and **causal from there on** (position $t \ge P$ sees everything $\le t$).

This is UniLM's mixed-mask training (Dong et al., 2019) and the "prefix LM" objective ablated in the T5 paper (Raffel et al., 2020); it resurfaces today in multimodal models that let *image* tokens attend bidirectionally as a sealed island (PaliGemma, Gemma 3) inside an otherwise causal stream. Its pitch is seductive: BERT-quality *understanding* of the prompt (door 1's gift), GPT-quality *generation* of the continuation (door 3's gift), zero new parameters — because the entire change is **one matrix that contains no weights**.

That last point deserves its own sentence, because it is the deepest lesson of this section: **GPT-2, BERT's encoder, and the prefix LM are the same block diagram — Sections 2 through 9 verbatim — differing only in the mask $M$.** Architecture, in the 2017 lineage, is substantially *mask policy*. Which also means we can do something unusually clean below: run the prefix LM through the **identical toy model of Section 12 — same $W_e, W_p, W_Q, W_K, W_V, W_O, W_1, W_2$, to the digit** — and watch precisely which numbers a mask flip touches, and which it provably cannot.

### 17.1 The mask, precisely

Replace Section 6.2's strictly-causal $M$ with

$$
M_{ij}=\begin{cases}0 & j\le i \quad\text{(causal part: the past is always visible)}\\ 0 & i<P \text{ and } j<P \quad\text{(prefix part: prompt sees all of prompt)}\\ -\infty & \text{otherwise}\end{cases}
$$

i.e. the causal lower triangle **plus a fully-unmasked $P\times P$ block in the top-left corner**. For our worked sequence — "the cat sat", ids $[1,2,3]$, with prompt **"the cat"** as the prefix ($P=2$) and "sat" the first continuation token:

$$
M=\begin{bmatrix}0&\mathbf{0}&-\infty\\0&0&-\infty\\0&0&0\end{bmatrix}
\qquad\text{vs causal}\qquad
M^{\text{GPT-2}}=\begin{bmatrix}0&-\infty&-\infty\\0&0&-\infty\\0&0&0\end{bmatrix}
$$

**Exactly one entry differs**: $M_{01}$, position 0's view of position 1. Row 1 already saw $\{0,1\}$ under causality; row 2 (continuation) is causal in both. That single unfrozen entry is the entire architectural delta — and the whole forward pass below is the story of what that one number does and does not reach.

### 17.2 The full forward pass, end to end

Same drill as Section 12, every step. **Bold** marks values that changed relative to the GPT-2 pass; everything unbolded is *identical to Section 12's numbers, and identical for a reason we state as we go*.

**(a) Embeddings (= Section 12.1, unchanged).** The input does not know about masks:

$$
x_0 = W_e[\text{ids}]+W_p[0{:}3]=\begin{bmatrix}0.500&-0.100&0.500&0.000\\-0.200&0.300&0.200&0.100\\0.000&0.300&-0.300&0.500\end{bmatrix}
$$

**(b) Pre-LN #1 (= Section 12.2a, unchanged)** — LayerNorm is per-row; no cross-token contact yet:

$$
\tilde x=\mathrm{LN}_1(x_0)=\begin{bmatrix}0.992&-1.172&0.992&-0.811\\-1.603&1.069&0.534&0.000\\-0.412&0.577&-1.402&1.237\end{bmatrix}
$$

**(c) Q, K, V (= Section 12.2b, unchanged — and note *why*).** The projections act row-by-row; **the mask has not entered the computation yet**. This is worth pausing on: $Q,K,V$ are mask-independent, so the prefix LM's keys and values at layer 1 are *bit-identical* to GPT-2's:

$$
Q=\begin{bmatrix}-0.081&-0.036&0.433&-0.559\\-0.214&0.534&-0.534&0.000\\0.223&-0.264&-0.198&0.709\end{bmatrix},\;
K=\begin{bmatrix}-0.153&-0.397&-0.081&0.054\\0.214&0.000&0.428&-0.267\\-0.008&0.462&-0.074&0.148\end{bmatrix},\;
V=\begin{bmatrix}-0.478&0.415&-0.054&0.117\\0.000&-0.374&0.053&0.321\\0.586&-0.280&-0.049&-0.256\end{bmatrix}
$$

**(d) Scores + the prefix mask — the one place the architectures diverge.** Per head, $\frac{Q_hK_h^\top}{\sqrt2}+M$. The newly-alive entry is score $(0,1)$: position 0 ("the") now scores its right-hand neighbor ("cat"). Worked, in both heads, from the head slices $Q_0^{(0)}=[-0.081,-0.036]$, $K_1^{(0)}=[0.214,0.000]$, $Q_0^{(1)}=[0.433,-0.559]$, $K_1^{(1)}=[0.428,-0.267]$:

$$
s^{(0)}_{01}=\frac{(-0.081)(0.214)+(-0.036)(0.000)}{1.414}=-0.012,\qquad
s^{(1)}_{01}=\frac{(0.433)(0.428)+(-0.559)(-0.267)}{1.414}=0.237
$$

(the second lands on a display boundary: $0.2366$ from the rounded inputs shown, $0.2362$ in full precision — the Section 12 rounding caveat at work). The masked score matrices — compare Section 12.2d, where the $(0,1)$ slots held $-\infty$:

$$
\text{scores}_0=\begin{bmatrix}0.019&\mathbf{-0.012}&-\infty\\-0.127&-0.032&-\infty\\0.050&0.034&-0.087\end{bmatrix},\qquad
\text{scores}_1=\begin{bmatrix}-0.046&\mathbf{0.236}&-\infty\\0.031&-0.162&-\infty\\0.038&-0.194&0.085\end{bmatrix}
$$

Softmax each row. Row 0 is now a genuine two-way contest instead of a forced $[1,0,0]$. Worked, head 0: $e^{0.019}=1.019$, $e^{-0.012}=0.988$, sum $2.007$, weights $[0.508,0.492]$; head 1: $e^{-0.046}=0.955$, $e^{0.236}=1.266$, sum $2.221$, weights $[0.430,0.570]$:

$$
A_0=\begin{bmatrix}\mathbf{0.508}&\mathbf{0.492}&0\\0.476&0.524&0\\0.350&0.345&0.305\end{bmatrix},\qquad
A_1=\begin{bmatrix}\mathbf{0.430}&\mathbf{0.570}&0\\0.548&0.452&0\\0.352&0.279&0.369\end{bmatrix}
$$

Read the changed row: in head 1, "the" now spends **57% of its attention on "cat"** — a token to its *right*, unthinkable under the causal mask. Rows 1 and 2 are untouched: row 1's visible set $\{0,1\}$ is the same under both masks, and row 2 is past the prefix, hence causal in both. *One mask entry changed, one attention row changed.*

**(e) Weighted values → concat → output projection.** Only row 0 can differ (its attention row is the only one that did). Worked, row 0, head 0: $0.508\,[-0.478,0.415]+0.492\,[0.000,-0.374]=[-0.243,0.027]$ ($0.026$ in full precision — another boundary digit); head 1: $0.430\,[-0.054,0.117]+0.570\,[0.053,0.321]=[0.007,0.233]$:

$$
\text{concat}=\begin{bmatrix}\mathbf{-0.243}&\mathbf{0.026}&\mathbf{0.007}&\mathbf{0.233}\\-0.228&0.002&-0.005&0.209\\0.011&-0.069&-0.022&0.036\end{bmatrix}
\xrightarrow{\;\cdot W_O\;}
\mathrm{MHSA}=\begin{bmatrix}\mathbf{-0.075}&\mathbf{-0.017}&\mathbf{-0.001}&\mathbf{0.094}\\-0.068&-0.021&-0.003&0.086\\-0.001&-0.020&0.000&0.003\end{bmatrix}
$$

(Row 0 of concat previously equaled $V_0$ outright, because $[1,0,0]$ attention *is* the identity copy; now it is a genuine mixture of "the" and "cat".)

**(f) Residual add #1:**

$$
a=x_0+\mathrm{MHSA}=\begin{bmatrix}\mathbf{0.425}&\mathbf{-0.117}&\mathbf{0.499}&\mathbf{0.094}\\-0.268&0.279&0.197&0.186\\-0.001&0.280&-0.300&0.503\end{bmatrix}
$$

**(g) Pre-LN #2 → FFN(GELU) → residual add #2.** The FFN is position-wise (Section 2), so rows 1–2 sail through with Section 12.3's exact values; row 0, worked in full:

$$
\tilde a_0=\mathrm{LN}_2(a)_0=[0.799,-1.372,1.097,-0.524]
$$
$$
(\tilde a W_1)_0=[-0.304,-0.455,0.304,0.402,-0.304,0.279,-0.269,0.431]
$$
$$
\mathrm{GELU}_0=[-0.116,-0.148,0.188,0.264,-0.116,0.170,-0.106,0.288]
$$
$$
\mathrm{FFN}_0=(\mathrm{GELU}\cdot W_2)_0=[-0.063,0.065,0.053,0.039]
$$
$$
x_1=a+\mathrm{FFN}(\mathrm{LN}_2(a))=\begin{bmatrix}\mathbf{0.361}&\mathbf{-0.052}&\mathbf{0.552}&\mathbf{0.133}\\-0.204&0.296&0.196&0.166\\0.060&0.304&-0.307&0.472\end{bmatrix}
$$

**(h) Final LayerNorm and the tied head:**

$$
x_N'=\mathrm{LN}_f(x_1)=\begin{bmatrix}\mathbf{0.494}&\mathbf{-1.317}&\mathbf{1.328}&\mathbf{-0.505}\\-1.675&0.963&0.435&0.277\\-0.246&0.586&-1.500&1.160\end{bmatrix}
$$

Worked, row 0's logit for `on` (id 4), $x'_{N,0}\cdot W_{e,4}$: $(0.494)(0.30)+(-1.317)(-0.30)+(1.328)(0.10)+(-0.505)(0.20)=0.575$ ✓.

$$
\text{logits}=\begin{bmatrix}\mathbf{-0.498}&\mathbf{0.859}&\mathbf{-0.359}&\mathbf{-0.817}&\mathbf{0.575}&\mathbf{0.188}&\mathbf{-0.048}&\mathbf{0.726}&\mathbf{-1.008}&\mathbf{-0.480}\\0.065&-0.872&0.947&-0.274&-0.692&0.575&-0.744&0.136&0.183&0.989\\0.591&-0.574&-0.108&1.189&-0.168&-0.632&-0.088&-0.403&0.718&0.414\end{bmatrix}
$$

**(i) Softmax, per row:**

$$
P=\begin{bmatrix}\mathbf{0.055}&\mathbf{0.213}&\mathbf{0.063}&\mathbf{0.040}&\mathbf{0.160}&\mathbf{0.109}&\mathbf{0.086}&\mathbf{0.186}&\mathbf{0.033}&\mathbf{0.056}\\0.085&0.033&0.204&0.060&0.040&0.141&0.038&0.091&0.095&0.213\\0.138&0.043&0.069&0.251&0.065&0.041&0.070&0.051&0.157&0.116\end{bmatrix}
$$

**The audit trail of one mask bit.** Trace what just happened end to end: unmasking $M_{01}$ changed score $(0,1)$ → attention row 0 → concat row 0 → MHSA row 0 → $a$ row 0 → FFN row 0 → $x_1$ row 0 → logits row 0 → $P$ row 0 — **a single row's pipeline, from mask to probabilities** — while rows 1 and 2 came out *identical to Section 12.4 in every digit*, because nothing in a one-block network lets row 0's changed representation reach them: attention is the only cross-token operation (Section 2), and it already happened. Keep that last clause in mind; it is about to become both a plot twist (17.4) and a systems catastrophe (17.5).

### 17.3 Training the prefix LM: who is allowed to have a loss?

Section 12.5 scored all three positions. The prefix LM **cannot** — and seeing why is the best way to internalize what the mask does. The rule from Section 1.3 was that position $t$'s prediction of token $t{+}1$ is honest only if position $t$ never sees $w_{\ge t+1}$. Check each row against the new mask:

- **Row 0** attends $\{0,1\}$ — it *sees token 1*, the very token it would be asked to predict. Its "prediction" of `cat` is contaminated: a trained model would simply copy the answer it can already read, and the loss would collapse to zero while teaching nothing. **Excluded.**
- **Row 1** attends $\{0,1\}$, all $\le 1$ — its prediction of token 2 (`sat`) is clean. **Included.** (The *last* prefix position is always the first legitimate predictor: it sees exactly the whole prompt and nothing more.)
- **Row 2** is causal — clean. **Included.**

So the prefix LM trains on **the continuation only**: loss at positions $P{-}1,\dots,T{-}1$, predicting tokens $P,\dots,T$. Computing it from $P$ above (rows 1 and 2 being unchanged, these are literally Section 12.5's own numbers):

| position $t$ | input | true next token | $P_{t,y_t}$ | $\mathcal{L}_t$ |
|---|---|---|---|---|
| 0 | the | *(cat — visible to row 0)* | — | **excluded from loss** |
| 1 | cat | sat (id 3) | $0.060$ | $2.813$ |
| 2 | sat | on (id 4) | $0.065$ | $2.733$ |

$$
\mathcal{L}=\tfrac12(2.813+2.733)=2.773
$$

And here is the economic fine print, connecting back to Section 1.5: GPT-2 extracted $T$ supervised predictions from this sequence; the prefix LM extracts $T-P+1$ — here $2$ of $3$. The prefix bought bidirectional *understanding* at the price of **supervision density**: every token you place inside the prefix is a token that no longer generates gradient. On a corpus split half-and-half, the prefix LM trains on half the labels per FLOP of the causal LM — the same tax BERT paid at 15% masking (Section 1.4), in milder form. Nothing is free; the mask giveth context and taketh away labels.

### 17.4 The plot twist: with one layer, generation never noticed

Look again at the final $P$: row 2 — the row that *generates* (Section 12.6) — is **unchanged**. Greedy still emits `sat` at $0.251$. In a **one-block** model, the bidirectional prefix made zero difference to the continuation. Why: row 2 consumed the prefix through $K,V$ rows 0–1, which were computed in step (c) — *before the mask acted* — and are therefore mask-independent. Row 0's enriched representation exists, but nothing downstream ever attends to it again.

Bidirectionality pays only **through depth**: at layer 2, the prefix's $K,V$ are projections of $x_1$ — which *does* carry row 0's mask-touched values — so the continuation finally tastes the two-way prefix. Watch it happen: run the same block a second time (weights reused, purely illustratively) on both models' $x_1$ and compare what layer 2 sees:

$$
\text{layer-2 } K_{\text{row }0}:\quad
\underbrace{[-0.201,-0.456,\ 0.131,\ 0.064]}_{\text{causal GPT-2}}
\;\;\text{vs}\;\;
\underbrace{[\mathbf{-0.213},\mathbf{-0.448},\ \mathbf{0.116},\ \mathbf{0.080}]}_{\text{prefix LM}}
$$

The prefix's *keys themselves* now differ, so row 2's layer-2 scores, weights, and output all shift, and the final distributions part ways (position-2 rows: $[0.138,0.046,0.063,0.258,\dots]$ vs $[0.137,0.046,0.063,0.258,\dots]$ — a hair's difference under these tiny random weights, but structurally guaranteed and compounding with every additional layer). The general law, worth stating because Section 17.5 is built on it:

> **In a prefix LM, a prefix token's $K,V$ at layer $\ell\ge2$ depend on the *entire* prefix.** At layer 1 they depend only on the token itself (step (c) — projections precede the mask); from layer 2 onward, each prefix position has already attended bidirectionally, so *every* prefix token's cache rows are functions of *all* prefix tokens. Contrast GPT-2, where token $t$'s $K,V$ at every layer depend only on tokens $\le t$ (Section 11.2).

### 17.5 Why KV-cache management is hard for a prefix LM

Sections 11.2, 12.8 and 12.9 rested on one theorem: *a token's $K,V$, once computed, are immutable — no future token can change them.* Every serving optimization we built is a corollary of it: append-only caches (12.8c), block sharing and copy-on-write (12.9.6), drop-and-recompute preemption (12.9.6). The boxed law of 17.4 says the prefix LM **breaks the theorem** — not everywhere, but exactly where serving makes its money.

Be precise about what still works and what dies:

**Still fine: a frozen prefix, and all decoding.** For a single request whose prompt is fixed, the prefix's $K,V$ are computed once in prefill and never change (the prefix isn't growing, so nothing invalidates); the continuation is causal, so decode-side caching is Section 12.8 verbatim. Paged storage (12.9.5) works untouched — pages don't care what mask produced their contents.

**Dead: growing the prefix.** Suppose the conversation continues — the model's answer plus a new user turn now join the *prompt* for the next round, and a prefix LM naturally wants the new turn inside the bidirectional region. In GPT-2, extending the prompt is an *append*: old $K,V$ rows are untouched (the immutability theorem), and only the new tokens are prefilled. In the prefix LM, the moment the prefix grows from $P$ to $P'$, **position 0 is entitled to attend to the new tokens** — so its layer-2+ representation changes, so *its* $K,V$ change, so everything attending to it changes. The entire prefix cache, all $P$ positions, all layers above the first, is **invalidated by an append**. Our toy shows it directly — grow the prefix from $P=2$ to $P=3$ (let "sat" join the bidirectional region, as a new turn would):

$$
\begin{array}{lcc}
& P=2 & P=3\\
\text{layer-1 } K_{\text{row }0} & [-0.153,-0.397,-0.081,0.054] & [-0.153,-0.397,-0.081,0.054]\;\;\text{(identical — pre-mask)}\\
\text{layer-1 } A_{0,\text{row }0} & [0.508,0.492,0] & [\mathbf{0.340},\mathbf{0.330},\mathbf{0.330}]\\
x_1\text{ row }0 & [0.361,-0.052,0.552,0.133] & [\mathbf{0.425},\mathbf{-0.061},\mathbf{0.559},\mathbf{0.049}]\\
\text{layer-2 } K_{\text{row }0} & [-0.213,-0.448,0.116,0.080] & [\mathbf{-0.161},\mathbf{-0.441},\mathbf{0.029},\mathbf{0.038}]
\end{array}
$$

Token 0's layer-2 key moved by up to $0.088$ — an entire cached row, for a token that was *not touched*, invalidated because a token arrived two positions to its right. Multiply by every prefix position and every layer: **appending $k$ tokens forces re-prefilling all $P+k$**, an $\mathcal{O}(P^2)$ attention bill *per conversational turn*, where GPT-2 pays $\mathcal{O}(Pk)$ once for the new tokens only.

**Dead: cross-request prefix sharing — the crown jewel of 12.9.6.** Two requests share a 1,000-token system prompt and diverge afterward. Under GPT-2, tokens 0–999 have *identical* $K,V$ in both requests (they cannot see the divergent part), so their blocks are shared with a refcount — the single biggest memory win in real serving, where thousands of requests share one system prompt. Under a prefix LM where each request's full prompt is bidirectional, token 0 of request A has attended to A's continuation of the prompt, and token 0 of request B to B's: **their $K,V$ differ from layer 2 up**. The "shared" prefix shares nothing. Longest-common-prefix caching (radix-tree caches à la SGLang), copy-on-write forking mid-prompt, sealed system-prompt pages — all of it silently keys on causal immutability, and all of it is invalid the moment the mask looks rightward.

**Dead: chunked prefill.** Serving engines like to prefill long prompts in chunks (say 512 tokens at a time) to interleave with ongoing decodes. Causally, chunk 1's $K,V$ are final before chunk 2 is touched — stream and forget. Bidirectionally, chunk 1's representations *depend on chunk 2*; you cannot finalize anything until the whole prefix is resident. Prefill becomes an all-or-nothing monolith.

Note the shape of the disaster: the prefix LM's *math* is barely different — one mask block — but it dissolves the **write-once, read-many** property that Section 12.9's entire memory system was built on. The nightmare isn't computing the cache; it's that the cache stopped being immutable.

### 17.6 How it is solved, conceptually

Every real remedy is some way of *restoring immutability by decree* — deciding, at the system level, which regions are frozen and never letting the mask's promises exceed the freeze:

1. **Seal the prefix (the encoder framing).** Declare the prefix immutable *per request*: prefill it once, bidirectionally, and never extend it — everything that arrives later (including subsequent turns) is appended **causally**, even though a "pure" prefix LM would have re-widened the window. This is just admitting what the prefix LM secretly is — door 2's encoder grafted into door 3's stream — and treating the prefix exactly like an encoder output: computed once, cached as a sealed read-only segment (paging-friendly, shareable within the request), consumed by the continuation's self-attention — which at that point is cross-attention in all but name (Section 1.2's observation, run in reverse). You trade a little modeling purity (later turns get causal treatment only) for the entire 12.8/12.9 machinery back.

2. **Share only exact-match wholes.** Cross-request reuse survives in one degraded form: if two requests have the **byte-identical complete prefix** (the same sealed system prompt, the same document), their prefix caches are identical and can be shared — hash the whole prefix, cache the whole segment. What is lost is *partial* sharing: match-the-first-800-tokens is worthless, because token 0's cache already depends on tokens 800+. Whole-prefix caching still pays handsomely for the "one system prompt, thousands of requests" pattern; it pays nothing for organic prompt overlap.

3. **Shrink the bidirectional region to what is truly static.** The modern multimodal compromise: make *only* the intrinsically order-free island bidirectional — the image tokens in PaliGemma, an embedded document — and keep all *text*, including the user's prompt, causal. The bidirectional islands are static by nature (an image never grows a new patch mid-conversation), so their caches are born immutable, and the causal remainder keeps every serving trick. This is the mask following the *data's* actual mutability structure rather than an ideal.

4. **Pay the re-prefill.** Where bidirectional prompts genuinely matter, accept the $\mathcal{O}(P^2)$-per-turn recomputation — noting, from Sections 11.1 and 12.9.2, that prefill is *parallel and compute-bound*, the one regime where the GPU is actually earning its FLOPs. Expensive, not pathological; drop-and-recompute preemption (12.9.6) even becomes the *normal* path rather than the fallback.

And the meta-lesson, completing the arc this chapter has drawn three times (Sections 1.5, 12.9.7, and here): the prefix LM loses in deployment not because its *modeling* is worse — T5's ablations had it competitive, and bidirectional prompt encoding is genuinely a better prior for understanding — but because its **serving economics** are worse: fewer training labels per token (17.3), no incremental multi-turn prefill, no partial prefix sharing, monolithic prefill. GPT-2's strictly causal mask looked, in 2019, like a modeling compromise made for training-time parallelism. It turned out to also be *the* enabling invariant of the entire inference industry: **write-once $K,V$**. The mask you choose on the pretraining whiteboard silently writes your datacenter's memory-management contract, years in advance. That — not any single trick — is why the fourth door, elegant as it is, stayed mostly shut.

---

## 18. Suggested next steps on the theory ladder

Since you are building toward modern LLMs and care about architecture specifically, the natural continuations from GPT-2, each a self-contained delta on this base, are: **rotary / ALiBi positional schemes** (fixing the learned-table extrapolation wall of Section 3.3); **RMSNorm** (a cheaper LayerNorm variant, Section 5.4); **SwiGLU/GLU FFN variants** (replacing Section 7's GELU MLP); **multi-query / grouped-query attention** (changing Section 6's $K,V$ sharing for inference); and **the Pre-LN→sandwich-norm/DeepNorm** lineage (further taming Section 5's deep-stack stability). Each one is "GPT-2 with exactly one component swapped," which is the cleanest way to keep studying architecture in isolation.