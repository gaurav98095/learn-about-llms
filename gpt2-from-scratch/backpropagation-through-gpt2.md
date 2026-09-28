# Backpropagation Through GPT-2

### A patient, complete derivation for software engineers whose calculus has gone rusty

---

## Preface: who this is for and how to read it

You can read PyTorch code. You know that `loss.backward()` fills in `.grad` on every parameter. You have a vague memory that this involves the chain rule, and a much sharper memory that the last time you saw a partial derivative was in a lecture hall you would rather not revisit.

This book takes you from that state to being able to derive, by hand, every gradient in GPT-2 — and then to using those derivations to explain *why* GPT-2 is shaped the way it is. Why is the attention score divided by $\sqrt{d_k}$? Why did everyone move LayerNorm to the front of the block? Why is there a residual connection there at all? These are not aesthetic choices. Each one is a fact about the backward pass, and once you can do the backward pass, each one becomes obvious.

**Three promises.**

1. **Nothing is skipped.** When a step says "hence", you will be able to see why. Where a line of algebra takes five sub-steps, all five are written out.
2. **Nothing is dumbed down.** We will do real matrix calculus. But we will build the machinery from scratch, in Chapter 0 and Chapter 1, assuming only that you once knew what a derivative was.
3. **Everything is verified.** Every formula in this book was implemented in NumPy and checked against numerical finite-difference gradients to a precision of about $10^{-10}$. The reference implementation is in Appendix A. Where I claim a design decision has a particular effect, there is a measurement.

**How to read it.**

Chapters 0 and 1 are the toolbox. If you can already differentiate $Y = XW + b$ with respect to $W$ from memory and explain why the answer is $X^\top\,dY$ and not $dY\,X^\top$, skim them. If you cannot, do not skip them — everything after Chapter 2 is the same four moves applied over and over, and if the four moves are shaky, the rest will feel like magic.

Chapter 2 lays out GPT-2's forward pass. Chapters 3–9 walk backwards through it, one operation at a time. Chapter 10 assembles the whole thing. Chapter 11 is the payoff: the design decisions, proven.

Read with a pen. Genuinely. The derivations are written so that you can close the book at any "hence" and reproduce the next line yourself.

---

## The method: one recipe, applied thirty times

Before any mathematics, here is the entire method of this book. Every single gradient in GPT-2 is obtained by these four steps in this order. Learn the steps now; the rest of the book is repetition.

> ### The Recipe
>
> **Step 1 — Write the forward pass *locally*, with indices.**
> Forget tensors. Write down how a *single number* in the output is computed from the input numbers. Nail the summation: which index is being summed over, and what is its range?
>
> **Step 2 — Write the *local* derivatives.**
> Differentiate that one output number with respect to one input number. Most of these are trivial once Step 1 is right.
>
> **Step 3 — Decide what to sum over.**
> The gradient of $J$ with respect to an input number is a *sum* over every place that number was used. Two rules cover almost everything:
> - **Shared-parameter rule:** if the quantity is a *weight* (reused for every token in every sequence), sum over the batch and sequence dimensions.
> - **Fan-out rule:** if the quantity is an *activation* that got spread into many outputs, sum over the output dimension it fanned into.
>
> **Step 4 — Lift to tensor form using shapes.**
> You now have an indexed sum. Write down the shape of every tensor involved and the shape the answer must have. There is almost always exactly one arrangement of matmuls and transposes that produces the right shape. That arrangement is the answer.

Step 4 is not a cheat. It is a *check*. The index expression from Step 3 is the ground truth; shapes tell you how to write it as a matmul without re-deriving it. When the shapes disagree with your instinct, your instinct is wrong.

---

## Notation and shape reference

Keep this page bookmarked.

| Symbol | Meaning | Typical GPT-2 small value |
|---|---|---|
| $B$ | batch size | 8 |
| $S$ | sequence length (context) | 1024 |
| $N$ | $B \cdot S$ — total tokens in the batch, used when we flatten | 8192 |
| $d$ | model / embedding dimension ($d_{\text{model}}$) | 768 |
| $n_h$ | number of attention heads | 12 |
| $d_k$ | per-head dimension, $d_k = d / n_h$ | 64 |
| $V$ | vocabulary size | 50257 |
| $L$ | number of transformer blocks | 12 |
| $J$ | the scalar loss | — |

**The single most important notational convention in this book:**

$$dX \;\;\equiv\;\; \frac{\partial J}{\partial X}$$

and $dX$ **always has exactly the same shape as $X$**. If $W$ is $(768, 3072)$ then $dW$ is $(768, 3072)$. If $h$ is $(8, 1024, 768)$ then $dh$ is $(8, 1024, 768)$.

This is not a mathematical necessity — it is a convention (sometimes called the *denominator layout* or the *gradient* rather than the *Jacobian*), and it is the one every deep learning framework uses. It is what makes the parameter update `W -= lr * dW` type-check. Hold onto it: whenever you are lost in a derivation, ask "what shape must this be?" and the convention will pull you back.

**Index convention.** Subscripts index into a tensor. $X_{b,s,i}$ is the single number at batch $b$, position $s$, channel $i$. When I write $\sum_{k=1}^{d}$ I mean a real, honest sum over $d$ terms — not a matrix product, not broadcasting. Every time you see a sum, imagine a `for` loop.

**A note on $\odot$.** $A \odot B$ is the elementwise (Hadamard) product: `A * B` in NumPy. It is *not* a matrix product. I will use $\cdot$ or juxtaposition for matrix products.

---
---

# Chapter 0 — The calculus you have forgotten

We start further back than you probably need. That is deliberate: the errors people make in matrix calculus are almost never matrix errors. They are scalar-calculus errors wearing a costume.

## 0.1 A derivative is an exchange rate

Forget limits and tangent lines. Here is the only intuition you need:

> The derivative $\dfrac{\partial y}{\partial x}$ is the answer to: **"if I nudge $x$ up by one tiny unit, how many tiny units does $y$ move?"**

It is an exchange rate. If $\frac{\partial y}{\partial x} = 3$, then $x$ dollars buy $y$ at three-to-one: bump $x$ by $0.001$ and $y$ goes up by about $0.003$.

That is all. Three consequences fall out immediately:

- **Sign matters.** A negative derivative means $y$ goes *down* when $x$ goes up.
- **Derivatives are local.** The exchange rate at $x=2$ can be completely different from the rate at $x=5$. It is the rate *right here*.
- **Units matter.** $\frac{\partial J}{\partial W}$ is "loss units per weight unit". This is why gradient descent works: if you know the exchange rate, you know which way to move the weight to buy yourself less loss.

**Worked check.** Let $y = x^2$. At $x=3$, $y=9$. Nudge: $x = 3.001 \Rightarrow y = 9.006001$. So $y$ moved $0.006001$ when $x$ moved $0.001$; the ratio is $6.001 \approx 6$. And indeed $\frac{dy}{dx} = 2x = 6$ at $x=3$. The tiny leftover $0.000001$ is the part that vanishes as the nudge shrinks — that is the whole content of "taking a limit".

Everything in this book is that computation, done symbolically, in bulk.

## 0.2 Partial derivatives: nudge one thing, hold the rest still

When $y$ depends on several inputs, say $y = f(x_1, x_2, x_3)$, the *partial* derivative $\frac{\partial y}{\partial x_2}$ asks the same question with one extra clause:

> "If I nudge $x_2$ by one tiny unit **and hold $x_1$ and $x_3$ frozen**, how much does $y$ move?"

The curly $\partial$ instead of $d$ is just a flag that says "there are other variables and I am freezing them".

**Mechanically**: treat every other variable as if it were the number $7$. Differentiate normally.

**Example.** $y = 3x_1 x_2 + x_2^2 - 5x_3$.

$$\frac{\partial y}{\partial x_1} = 3x_2 \qquad \frac{\partial y}{\partial x_2} = 3x_1 + 2x_2 \qquad \frac{\partial y}{\partial x_3} = -5$$

For $\frac{\partial y}{\partial x_1}$: the term $3x_1x_2$ is "constant $\times\, x_1$" where the constant is $3x_2$, so its derivative is $3x_2$. The terms $x_2^2$ and $-5x_3$ contain no $x_1$ at all, so nudging $x_1$ does not move them — their derivative is $0$. This is the part people forget: **terms without the variable contribute exactly zero**, and in tensor calculus this fact does enormous amounts of work, because most terms in a big sum do not contain the index you are differentiating with respect to.

## 0.3 The chain rule, single path

Suppose $x \to u \to y$: $u$ depends on $x$, and $y$ depends on $u$.

$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$$

**Why this is obviously true, in exchange-rate language.** Suppose rupees convert to dollars at 80 rupees per dollar, and dollars convert to euros at 0.9 euros per dollar. How many euros per rupee? You multiply the rates: $\frac{1}{80} \times 0.9$. Rates along a chain compose by multiplication. That is the chain rule.

**Worked example.** $y = (3x+1)^2$. Let $u = 3x+1$, so $y = u^2$.

- $\frac{\partial y}{\partial u} = 2u = 2(3x+1)$
- $\frac{\partial u}{\partial x} = 3$
- $\frac{\partial y}{\partial x} = 2(3x+1)\cdot 3 = 18x + 6$

Check by expanding first: $y = 9x^2 + 6x + 1$, so $\frac{dy}{dx} = 18x+6$. ✓

**This is already backpropagation.** Read the chain rule right-to-left and it says: to get the gradient at $x$, take the gradient at $u$ and multiply by the local derivative $\frac{\partial u}{\partial x}$. A neural network is $x \to u_1 \to u_2 \to \cdots \to J$. Backprop starts with $\frac{\partial J}{\partial J} = 1$ at the right end and multiplies its way leftwards, one local derivative at a time.

## 0.4 The multivariable chain rule — the rule that actually matters

Here is the one that people forget, and it is the one that generates nearly every summation you will see in this book.

Suppose $x$ influences $y$ through **several intermediate variables at once**:

```
        ┌──> u₁ ──┐
   x ───┼──> u₂ ──┼──> J
        └──> u₃ ──┘
```

Then:

$$\boxed{\;\frac{\partial J}{\partial x} \;=\; \sum_{k} \frac{\partial J}{\partial u_k}\cdot\frac{\partial u_k}{\partial x}\;}$$

**In words: when a variable fans out into several places, its gradient is the *sum* of the gradients coming back from every place it went.**

**Why a sum and not a product?** Because the effects are *simultaneous and independent*. Nudge $x$ by $\varepsilon$. Then $u_1$ moves by $\frac{\partial u_1}{\partial x}\varepsilon$, and $u_2$ moves by $\frac{\partial u_2}{\partial x}\varepsilon$, and $u_3$ moves by $\frac{\partial u_3}{\partial x}\varepsilon$ — all at once. Each of those movements independently pushes $J$ around, and small effects on the same quantity add up. So the total movement of $J$ is the sum of the three contributions.

Compare: the chain rule *multiplies* when you go **through** something in sequence, and *adds* when something **splits** and rejoins. Series → multiply. Parallel → add. If you remember electrical circuits, it is the same bookkeeping.

**Worked example.** Let $u = x^2$, $v = 3x$, and $J = uv$.

Path-wise:
- $\frac{\partial J}{\partial u} = v = 3x$, and $\frac{\partial u}{\partial x} = 2x$. Contribution: $3x \cdot 2x = 6x^2$.
- $\frac{\partial J}{\partial v} = u = x^2$, and $\frac{\partial v}{\partial x} = 3$. Contribution: $x^2 \cdot 3 = 3x^2$.
- Total: $\frac{\partial J}{\partial x} = 6x^2 + 3x^2 = 9x^2$.

Check directly: $J = x^2 \cdot 3x = 3x^3$, so $\frac{dJ}{dx} = 9x^2$. ✓

Had you only followed one path you would have got $6x^2$ — a wrong answer that looks perfectly reasonable. **Almost every hand-derived backprop bug is a missing path.**

## 0.5 The Fan-Out Law, stated once so we can cite it forever

I will refer to this constantly, so let us name it.

> **The Fan-Out Law.** If a quantity $x$ is *used* in $m$ different downstream places, then
> $$\frac{\partial J}{\partial x} = \sum_{\text{all } m \text{ uses}} (\text{gradient arriving from that use})$$
> Equivalently: **in the forward pass, "copy" becomes "sum" in the backward pass.**

This single law explains, in advance:

- why the gradient for a **weight matrix** sums over batch and sequence (one weight is reused for every token);
- why the gradient for a **bias** sums over batch and sequence (one bias is added to every token);
- why the gradient at a **residual junction** adds two incoming gradients (the residual stream was copied into two branches);
- why anything **broadcast** in the forward pass gets **summed** over the broadcast axis in the backward pass;
- why the gradient of the **tied embedding matrix** in GPT-2 has two separate contributions.

Learn this law and half the book is already done.

## 0.6 A complete worked example, with numbers

Let us do one full graph by hand so that when we do it with tensors it feels familiar.

Forward:

$$a = 2, \quad b = 3$$
$$c = a\,b, \qquad e = a + c, \qquad J = e^2$$

Note $a$ fans out: it is used in $c$ **and** in $e$. Expect a sum.

**Forward numbers:** $c = 6$, $e = 2 + 6 = 8$, $J = 64$.

**Backward, one node at a time.** We always compute $\frac{\partial J}{\partial \text{node}}$ starting from the output.

1. $\dfrac{\partial J}{\partial J} = 1$. (Trivially: nudge $J$ by one unit and $J$ moves one unit.)

2. $J = e^2 \Rightarrow \dfrac{\partial J}{\partial e} = 2e = 16$.

3. $e = a + c$. The local derivative $\frac{\partial e}{\partial c} = 1$, so
   $\dfrac{\partial J}{\partial c} = \dfrac{\partial J}{\partial e}\cdot 1 = 16$.

4. $c = ab$. So $\frac{\partial c}{\partial b} = a = 2$, giving
   $\dfrac{\partial J}{\partial b} = 16 \times 2 = 32$.

5. **Now $a$ — the fan-out node.** Two paths:
   - via $e$ directly: $\frac{\partial e}{\partial a} = 1$, contribution $16 \times 1 = 16$;
   - via $c$: $\frac{\partial c}{\partial a} = b = 3$, contribution $16 \times 3 = 48$.

   $$\frac{\partial J}{\partial a} = 16 + 48 = 64$$

**Verify numerically.** $J(a) = (a + 3a)^2 = 16a^2$, so $\frac{dJ}{da} = 32a = 64$ at $a=2$. ✓ And a finite-difference check: $a = 2.001 \Rightarrow c = 6.003,\ e=8.004,\ J = 64.064016$. Change in $J$ is $0.064016$ for a change of $0.001$ in $a$, ratio $64.016 \approx 64$. ✓

Sit with step 5. That plus-sign is the whole of backpropagation's bookkeeping.

## 0.7 The derivative rules we will actually use

You need six. Here they are with the derivations we will lean on later.

| Function | Derivative | Where it shows up in GPT-2 |
|---|---|---|
| $x^n$ | $n x^{n-1}$ | variance in LayerNorm; GELU's cubic term |
| $e^x$ | $e^x$ | softmax |
| $\ln x$ | $1/x$ | cross-entropy loss |
| $\tanh x$ | $1 - \tanh^2 x$ | GELU (tanh approximation) |
| $u \cdot v$ | $u'v + uv'$ (product rule) | GELU; softmax quotient |
| $u/v$ | $\dfrac{u'v - uv'}{v^2}$ (quotient rule) | softmax; normalisation |

Two of these deserve a moment.

**The quotient rule is the engine of the softmax derivative.** We will use it in exactly the form $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$, and the subtlety will be that $v$ (the sum of exponentials) depends on *every* input, which is why the softmax Jacobian is not diagonal. Chapter 3 does this carefully.

**$\frac{d}{dx}\ln x = 1/x$ is why cross-entropy is so well behaved.** The log's derivative is a *reciprocal*, and the softmax's derivative contains a *product* of probabilities. When you compose them, the reciprocal cancels the product and you get the astonishingly clean $\hat{P} - Y$. This cancellation is not luck; it is the reason this particular loss is paired with this particular activation. Chapter 3, again.

**A derivation you should see once — $\frac{d}{dx}\tanh x = 1 - \tanh^2 x$.**

$\tanh x = \frac{e^x - e^{-x}}{e^x + e^{-x}}$. Let $u = e^x - e^{-x}$ and $v = e^x + e^{-x}$. Then $u' = e^x + e^{-x} = v$ and $v' = e^x - e^{-x} = u$. By the quotient rule:

$$\frac{d}{dx}\tanh x = \frac{u'v - uv'}{v^2} = \frac{v\cdot v - u \cdot u}{v^2} = \frac{v^2 - u^2}{v^2} = 1 - \frac{u^2}{v^2} = 1 - \tanh^2 x$$

Notice the shape of that result: **the derivative is expressible in terms of the output**. That is a recurring gift. It means the backward pass can reuse what the forward pass already computed instead of recomputing it — which is exactly what a "cache" is for. Softmax, sigmoid, and tanh all have this property; GELU almost does.

## 0.8 The Kronecker delta, and how sums collapse

This is a piece of notation that will save you enormous confusion later.

$$\delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j\end{cases}$$

That is it — it is the identity matrix, written with indices. Its purpose is to express "the derivative of a thing with respect to itself is 1, and with respect to anything else is 0":

$$\frac{\partial x_i}{\partial x_j} = \delta_{ij}$$

**Why it matters: it collapses sums.** The single most useful identity in this entire book is

$$\sum_{k} \delta_{ik}\, f_k = f_i$$

Read it out loud: "sum over $k$ of ($1$ when $k=i$, else $0$) times $f_k$". Every term is zero except the one where $k = i$, which survives with coefficient 1. So the whole sum collapses to a single term.

**Worked use.** Suppose $y_j = \sum_{k=1}^{d} x_k W_{kj}$ and we want $\frac{\partial y_j}{\partial x_i}$.

$$\frac{\partial y_j}{\partial x_i} = \sum_{k=1}^{d} \frac{\partial x_k}{\partial x_i} W_{kj} = \sum_{k=1}^{d} \delta_{ki} W_{kj} = W_{ij}$$

The $d$-term sum collapsed to one term. This mechanical move — *write the delta, then collapse the sum* — is how essentially every local derivative in Chapters 3–9 gets computed. If you find yourself staring at a monstrous sum wondering where to start, the answer is always: expand the derivative into a delta and let it eat the sum.

## 0.9 So what *is* backpropagation?

A neural network is a directed acyclic graph of simple operations. Backpropagation is one specific, efficient way to compute the gradient of the final scalar with respect to every node in that graph.

The naive alternative — for each of the 124 million parameters, nudge it and re-run the forward pass — would take 124 million forward passes. Backpropagation gets all 124 million gradients in **one** backward pass, at roughly twice the cost of one forward pass.

The trick is *ordering*. Because the loss is a **scalar**, it is enormously cheaper to propagate gradients from the output backwards (each step is a vector–matrix product) than from the inputs forwards (each step would be a matrix–matrix product). This is the same reason you'd compute $\mathbf{v}^\top ABC$ left-to-right rather than right-to-left.

Operationally, backprop is:

1. **Forward pass**: compute every intermediate value and *cache* whatever the backward pass will need.
2. **Seed**: set $\frac{\partial J}{\partial J} = 1$.
3. **Backward pass**: visit nodes in reverse topological order. At each node, you already have the gradient of $J$ with respect to that node's *output*; multiply by the node's *local* derivatives to get gradients with respect to its inputs and parameters; and **accumulate** (`+=`, never `=`) into any node that fans out.

That "accumulate, never assign" instruction is the Fan-Out Law in code. When you write GPT-2's backward pass by hand and it silently trains to a worse loss than PyTorch, the bug is a `=` where you needed a `+=`.

---
---

# Chapter 1 — Matrix calculus without tears

Matrix calculus has a bad reputation, mostly because it is usually taught as a table of identities to memorise ("$\frac{\partial}{\partial X}\text{tr}(AX) = A^\top$", and so on). That approach fails the moment you meet a shape you have not memorised — which, in a transformer, is immediately.

We do it differently. **We never differentiate a matrix.** We differentiate a *single number* with respect to another *single number*, using only Chapter 0, and then we look at the shapes to see what matrix operation reassembles those numbers. That procedure never fails and requires memorising nothing.

## 1.1 The convention: gradients wear the same clothes as their variables

We fix, once and forever:

$$dX \equiv \frac{\partial J}{\partial X}, \qquad \text{shape}(dX) = \text{shape}(X)$$

Concretely, if $W$ has shape $(d_{\text{in}}, d_{\text{out}}) = (768, 3072)$, then $dW$ is a $(768, 3072)$ matrix whose entry $(i,j)$ is the ordinary scalar partial derivative $\frac{\partial J}{\partial W_{ij}}$.

Two things follow that you should internalise:

- **`W -= lr * dW` is well-typed.** This is the entire reason for the convention.
- **The "gradient" is not the "Jacobian".** If $y$ is a vector of size $m$ and $x$ a vector of size $n$, the Jacobian $\frac{\partial y}{\partial x}$ is an $m \times n$ matrix. But we never build that. We only ever have *the gradient of a scalar*, which has the shape of its variable. This is why backprop is cheap: we never materialise Jacobians, we only ever apply them.

The one place Jacobians appear in this book is Chapter 11, where the *structure* of a Jacobian (its rank, its singular values) is what proves a design decision. Even there, we never build one in code.

## 1.2 Index notation: the escape hatch that always works

Whenever a tensor expression confuses you, drop to indices. In index notation there are no matrices, only numbers and `for` loops.

Standard matrix product $C = A\cdot B$, in indices:

$$C_{xy} = \sum_{k} A_{xk} B_{ky}$$

Read this as three loops:

```python
for x in range(rows_A):
    for y in range(cols_B):
        C[x, y] = 0
        for k in range(cols_A):          # the summed index
            C[x, y] += A[x, k] * B[k, y]
```

The summed index $k$ is the "inner" dimension — it appears twice on the right and not at all on the left. **Free indices** ($x$, $y$) appear on both sides. That is the pattern to look for: an index that appears on the right but not the left is being summed over.

This one convention lets you write any tensor operation unambiguously, including ones with batch dimensions that no matrix-calculus identity table covers.

## 1.3 The Recipe, in full

Now we can state the method precisely. This is the procedure for the rest of the book.

**Step 1 — Local forward pass.** Write one output element as an explicit sum over input elements. Ask: *which index am I summing, and over what range?* Getting this wrong is the only way to get the final answer wrong.

**Step 2 — Local derivatives.** Differentiate that output element with respect to one input element. Use Chapter 0.8: expand into a Kronecker delta, then let it collapse the sum.

**Step 3 — Apply the Fan-Out Law: what do I sum over?** The gradient of $J$ with respect to an input element is the sum of contributions from every output element that used it. In practice:

- **If the quantity is a shared parameter** (a weight, a bias, an embedding row, a LayerNorm gain) — it is reused for every token in the batch, so **sum over the batch and sequence dimensions** $(b, s)$. This is the shared-parameter rule.
- **If the quantity is an activation** — it was used to compute every one of the $d_{\text{out}}$ outputs at its own position, so **sum over the output dimension**. This is the fan-out rule.
- **If the quantity was broadcast** — sum over every axis it was broadcast along.

**Step 4 — Lift to tensors by shape.** List the shape of each participating tensor and the required shape of the answer. Then find the unique arrangement of matmul / transpose / sum that produces it. Verify it matches your Step-3 index expression, and you are done.

Let us now run the recipe on the three "atoms" that GPT-2 is built out of.

## 1.4 Atom 1 — the linear layer $Y = XW + b$

This is the workhorse. Q, K, V projections, the attention output projection, both MLP layers, and the unembedding are all this one operation.

**Setup.** $X$ has shape $(N, d_{\text{in}})$, $W$ has shape $(d_{\text{in}}, d_{\text{out}})$, $b$ has shape $(d_{\text{out}},)$, and $Y$ has shape $(N, d_{\text{out}})$. Here $N$ is however many rows there are — in GPT-2 it will be $B\times S$ tokens.

### Step 1 — local forward

$$Y_{n,j} = \sum_{i=1}^{d_{\text{in}}} X_{n,i}\, W_{i,j} \;+\; b_j$$

Read carefully: to make output element $(n,j)$ we sweep $i$ across the **input** dimension. Row $n$ of $X$ is used; column $j$ of $W$ is used; $b_j$ is used. The summed index is $i$, ranging over $d_{\text{in}}$.

Note what is *not* summed: $n$. Each row of $X$ produces its own row of $Y$, independently. That independence is why the batch dimension will behave the way it does.

### Step 2 — local derivatives

Three of them.

**(a) With respect to a weight $W_{p,q}$:**

$$\frac{\partial Y_{n,j}}{\partial W_{p,q}} = \frac{\partial}{\partial W_{p,q}}\left(\sum_i X_{n,i}W_{i,j} + b_j\right) = \sum_i X_{n,i}\,\delta_{ip}\delta_{jq} = X_{n,p}\,\delta_{jq}$$

The $\delta_{ip}$ collapsed the sum (only the $i=p$ term survives), and the $\delta_{jq}$ survives as a condition: **$W_{p,q}$ only affects outputs in column $q$.** That is worth pausing on. Column $q$ of $W$ produces column $q$ of $Y$ and nothing else.

**(b) With respect to a bias $b_q$:**

$$\frac{\partial Y_{n,j}}{\partial b_q} = \delta_{jq}$$

**(c) With respect to an input $X_{m,p}$:**

$$\frac{\partial Y_{n,j}}{\partial X_{m,p}} = \sum_i \delta_{nm}\delta_{ip} W_{i,j} = \delta_{nm} W_{p,j}$$

The $\delta_{nm}$ says: **row $m$ of $X$ only affects row $m$ of $Y$.** Rows do not mix in a linear layer. (This is exactly what attention *does* do and a linear layer *does not* — a fact we will lean on.)

### Step 3 — what do we sum over?

**For $W$:** $W$ is a shared parameter. Element $W_{p,q}$ was used for **every row $n$** and (from Step 2a) affects only column $q$. So we sum over $n$ and over $j$, but the $\delta_{jq}$ kills all $j$ except $j = q$:

$$dW_{p,q} = \sum_{n=1}^{N}\sum_{j=1}^{d_{\text{out}}} dY_{n,j}\cdot X_{n,p}\,\delta_{jq} = \sum_{n=1}^{N} dY_{n,q}\, X_{n,p}$$

**This is the shared-parameter rule in action: the sum over $N$ is the sum over the batch.** Every token in every sequence contributes a vote for how $W_{p,q}$ should change, and the votes add.

**For $b$:** identically,

$$db_q = \sum_{n=1}^{N} dY_{n,q}$$

**For $X$:** $X_{m,p}$ is an activation. It stayed in row $m$ (the $\delta_{nm}$) but fanned out into **all $d_{\text{out}}$ columns** of that row. So we sum over $j$:

$$dX_{m,p} = \sum_{j=1}^{d_{\text{out}}} dY_{m,j}\, W_{p,j}$$

**This is the fan-out rule: the sum runs over the output dimension.**

Notice how cleanly the two rules separated. The weight summed over $N$ (batch); the activation summed over $d_{\text{out}}$ (outputs). This is the distinction to hold onto — it is the third step of the Recipe, and it is where hand-derived backprop usually goes wrong.

### Step 4 — lift to tensors

**For $dW$.** Required shape: $(d_{\text{in}}, d_{\text{out}})$. We have $X$: $(N, d_{\text{in}})$ and $dY$: $(N, d_{\text{out}})$. The index expression $dW_{p,q} = \sum_n X_{n,p}\,dY_{n,q}$ sums over $n$, which is the *first* axis of both. To contract first-with-first we transpose $X$:

$$\boxed{\,dW = X^\top\, dY\,} \qquad (d_{\text{in}}, N)\cdot(N, d_{\text{out}}) \to (d_{\text{in}}, d_{\text{out}})\ \checkmark$$

**For $db$.** Required shape $(d_{\text{out}},)$; sum $dY$ over its row axis:

$$\boxed{\,db = \sum_{n} dY_{n,:}\,} \qquad \texttt{dY.sum(axis=0)}$$

**For $dX$.** Required shape $(N, d_{\text{in}})$. We have $dY$: $(N, d_{\text{out}})$ and $W$: $(d_{\text{in}}, d_{\text{out}})$. The expression $dX_{m,p} = \sum_j dY_{m,j}W_{p,j}$ contracts the *second* axis of $dY$ with the *second* axis of $W$:

$$\boxed{\,dX = dY\, W^\top\,} \qquad (N, d_{\text{out}})\cdot(d_{\text{out}}, d_{\text{in}}) \to (N, d_{\text{in}})\ \checkmark$$

### Why "shape-first" works, and how to use it honestly

Look at what happened. Given $dY$, $X$, $W$ and the requirement that $dW$ be $(d_{\text{in}}, d_{\text{out}})$, there is essentially *one* way to combine the available tensors. Try the alternatives: $dY^\top X$ gives $(d_{\text{out}}, N)\cdot(N, d_{\text{in}}) = (d_{\text{out}}, d_{\text{in}})$ — the transpose of what we want. $X\,dY$ does not even conform. So the shape constraint alone nearly determines the answer.

Use this, but use it as a *check*, not a *derivation*. The index expression from Step 3 is the truth. Shapes tell you how to write that truth efficiently. There is one classic situation where shapes alone will mislead you — when two candidate arrangements both conform because a dimension happens to be square (very common in attention, where $d = d_{\text{model}}$ appears on both sides of many matrices, and where $S \times S$ score matrices are square). In those cases only the index expression saves you. We will hit exactly this in Chapter 8.

**Memorise these three.** They will appear a dozen times:

$$dW = X^\top dY, \qquad db = \textstyle\sum_{\text{rows}} dY, \qquad dX = dY\,W^\top$$

A mnemonic that actually reflects the mathematics: *to get the gradient of the weight, sandwich the incoming gradient against the saved input, transposing the one whose batch axis needs to be contracted; to push the gradient back to the input, multiply by the weight with its two dimensions swapped, because you are travelling from the output side to the input side.*

## 1.5 Atom 2 — an elementwise activation

**Setup.** $A = \phi(H)$ where $\phi$ is applied elementwise (GELU, tanh, ReLU, sigmoid). $A$ and $H$ have identical shape.

**Step 1 — local forward.** There is no sum at all:

$$A_{n,j} = \phi(H_{n,j})$$

**Step 2 — local derivative.**

$$\frac{\partial A_{n,j}}{\partial H_{m,p}} = \phi'(H_{n,j})\,\delta_{nm}\delta_{jp}$$

**Step 3 — sum over what?** Formally $H_{m,p}$ is an activation, so the fan-out rule says sum over the output dimension:

$$dH_{m,p} = \sum_{j} dA_{m,j}\,\phi'(H_{m,j})\,\delta_{jp} = dA_{m,p}\,\phi'(H_{m,p})$$

**The sum collapses to one term**, because an elementwise function has no fan-out: each input touches exactly one output. This is why elementwise ops are the cheapest thing in a backward pass, and it is worth seeing the sum collapse rather than just asserting the answer — the same machinery handles the trivial case and the hard case.

**Step 4 — tensor form.**

$$\boxed{\,dH = dA \odot \phi'(H)\,}$$

Elementwise multiply. Note the argument: $\phi'(H)$, the derivative evaluated at the **pre-activation** $H$. A classic bug is evaluating at $A$. (For tanh you *can* write $1 - A^2$ because of the identity in §0.7 — but that is a special property of tanh, not a general licence.)

## 1.6 Atom 3 — broadcasting

**Setup.** $Y = X + b$ where $X$ is $(N, d)$ and $b$ is $(d,)$. NumPy silently copies $b$ across all $N$ rows.

This is the Fan-Out Law in its purest form. $b_j$ was used $N$ times. Therefore:

$$db_j = \sum_{n=1}^{N} dY_{n,j} \qquad\Longrightarrow\qquad \boxed{\,db = \texttt{dY.sum(axis=0)}\,}$$

> **The Broadcasting Rule.** *Every axis you broadcast along in the forward pass, you sum along in the backward pass.* Forward `expand` ⇔ backward `sum`. Forward `sum` ⇔ backward `expand`.

They are exact duals. This rule alone will resolve most "why is there a `.sum()` here?" questions in GPT-2's backward pass, including the ones in LayerNorm, which is where they get genuinely confusing.

## 1.7 The full worked example: a two-layer network, end to end

Now we run the Recipe on a complete little network. This example is deliberately the *same shape* as GPT-2's output head (linear → activation → linear → softmax → cross-entropy), so when we get to Chapter 3 it will feel like something you have already done.

### The network

$$H = XW_1 + b_1 \qquad (N, d_h)$$
$$A = \tanh(H) \qquad (N, d_h)$$
$$Z = AW_2 + b_2 \qquad (N, C)$$
$$\hat{P} = \text{softmax}(Z) \qquad (N, C)$$
$$J = -\frac{1}{N}\sum_{n=1}^{N}\sum_{c=1}^{C} Y_{n,c}\ln \hat{P}_{n,c}$$

with $X$: $(N,3)$, $W_1$: $(3,4)$, $W_2$: $(4,2)$, $N = 2$, $C = 2$. $Y$ is one-hot.

### Concrete numbers

$$X = \begin{bmatrix} 1.0 & -2.0 & 0.5 \\ 0.0 & 1.0 & 2.0\end{bmatrix}\quad
W_1 = \begin{bmatrix} 0.2 & -0.3 & 0.5 & 0.1 \\ 0.4 & 0.1 & -0.2 & 0.3 \\ -0.1 & 0.5 & 0.2 & -0.4\end{bmatrix}\quad
b_1 = \begin{bmatrix}0.1 \\ 0.0 \\ -0.1 \\ 0.2\end{bmatrix}^\top$$

$$W_2 = \begin{bmatrix} 0.3 & -0.2 \\ 0.1 & 0.4 \\ -0.5 & 0.2 \\ 0.2 & 0.1 \end{bmatrix}\quad b_2 = \begin{bmatrix}0.05 & -0.05\end{bmatrix}\quad \text{labels} = [0, 1]$$

**Forward, computed by hand for one entry then given in full.**

$H_{1,1} = X_{1,1}W_{1,11} + X_{1,2}W_{1,21} + X_{1,3}W_{1,31} + b_{1,1}$
$\phantom{H_{1,1}} = (1.0)(0.2) + (-2.0)(0.4) + (0.5)(-0.1) + 0.1 = 0.2 - 0.8 - 0.05 + 0.1 = -0.55$ ✓

$$H = \begin{bmatrix}-0.55 & -0.25 & 0.90 & -0.50 \\ 0.30 & 1.10 & 0.10 & -0.30\end{bmatrix} \qquad A = \begin{bmatrix}-0.5005 & -0.2449 & 0.7163 & -0.4621 \\ 0.2913 & 0.8005 & 0.0997 & -0.2913\end{bmatrix}$$

$$Z = \begin{bmatrix}-0.5752 & 0.0492 \\ 0.1093 & 0.2027\end{bmatrix} \qquad \hat{P} = \begin{bmatrix}0.3488 & 0.6512 \\ 0.4767 & 0.5233\end{bmatrix}$$

$$J = -\tfrac{1}{2}\big(\ln 0.3488 + \ln 0.5233\big) = -\tfrac{1}{2}(-1.0533 - 0.6476) = 0.8504$$

### Backward, by the Recipe

**(i) The loss and softmax together.** Chapter 3 derives this in full detail; for now take the result, which is the single most famous simplification in deep learning:

$$dZ = \frac{1}{N}\left(\hat{P} - Y\right) = \frac{1}{2}\begin{bmatrix}0.3488 - 1 & 0.6512 - 0 \\ 0.4767 - 0 & 0.5233 - 1\end{bmatrix} = \begin{bmatrix}-0.3256 & 0.3256 \\ 0.2383 & -0.2383\end{bmatrix}$$

Two sanity facts you can already read off: **each row sums to zero** (softmax outputs are constrained to sum to 1, so the gradient must live in the subspace that preserves that), and **the correct class has a negative gradient** (push its logit up), while wrong classes have positive gradient (push theirs down). Good.

**(ii) Second linear layer.** Atom 1 with $X \to A$, $W \to W_2$, $Y \to Z$:

$$dW_2 = A^\top dZ, \qquad db_2 = \sum_n dZ_{n,:}, \qquad dA = dZ\,W_2^\top$$

Compute one entry of $dW_2$ by hand, to see the batch sum happen:

$$dW_{2,\,11} = \sum_{n=1}^{2} A_{n,1}\,dZ_{n,1} = (-0.5005)(-0.3256) + (0.2913)(0.2383) = 0.1630 + 0.0694 = 0.2324\ \checkmark$$

That sum over $n$ **is** the shared-parameter rule. Both training examples voted; the votes added.

$$dW_2 = \begin{bmatrix}0.2324 & -0.2324 \\ 0.2705 & -0.2705 \\ -0.2095 & 0.2095 \\ 0.0810 & -0.0810\end{bmatrix} \qquad db_2 = \begin{bmatrix}-0.0873 & 0.0873\end{bmatrix}$$

And one entry of $dA$, to see the fan-out rule happen:

$$dA_{1,1} = \sum_{j=1}^{2} dZ_{1,j}\,W_{2,\,1j} = (-0.3256)(0.3) + (0.3256)(-0.2) = -0.0977 - 0.0651 = -0.1628\ \checkmark$$

That sum ran over $j$, the **output** dimension — because $A_{1,1}$ was used to compute both output logits.

$$dA = \begin{bmatrix}-0.1628 & 0.0977 & 0.2279 & -0.0326 \\ 0.1192 & -0.0715 & -0.1668 & 0.0238\end{bmatrix}$$

**(iii) The tanh.** Atom 2, using $\tanh'(h) = 1 - \tanh^2(h) = 1 - A^2$:

$$dH = dA \odot (1 - A^2)$$

$$dH_{1,1} = (-0.1628)\times(1 - (-0.5005)^2) = (-0.1628)(1 - 0.2505) = (-0.1628)(0.7495) = -0.1220\ \checkmark$$

$$dH = \begin{bmatrix}-0.1220 & 0.0918 & 0.1110 & -0.0256 \\ 0.1091 & -0.0257 & -0.1652 & 0.0218\end{bmatrix}$$

**(iv) First linear layer.** Atom 1 again:

$$dW_1 = X^\top dH = \begin{bmatrix}-0.1220 & 0.0918 & 0.1110 & -0.0256 \\ 0.3531 & -0.2093 & -0.3871 & 0.0730 \\ 0.1571 & -0.0055 & -0.2749 & 0.0308\end{bmatrix}$$

$$db_1 = \begin{bmatrix}-0.0130 & 0.0661 & -0.0542 & -0.0038\end{bmatrix} \qquad dX = dH\,W_1^\top = \begin{bmatrix}0.0010 & -0.0695 & 0.0906 \\ -0.0509 & 0.0806 & -0.0655\end{bmatrix}$$

### Verification

Every one of these numbers was checked against central finite differences,

$$\frac{\partial J}{\partial \theta_i} \approx \frac{J(\theta + \varepsilon e_i) - J(\theta - \varepsilon e_i)}{2\varepsilon}, \qquad \varepsilon = 10^{-6}$$

with a **maximum absolute error of $1.0\times10^{-10}$** across all 26 parameters. The derivation is correct.

> **Do this yourself, once.** Write the finite-difference checker before you write the backward pass. It is twenty lines, it is the only thing standing between you and a subtly wrong gradient that still trains — badly — and it is how every formula in the rest of this book was validated.

## 1.8 Batched tensors: what changes when there are three dimensions?

In GPT-2 activations are $(B, S, d)$, not $(N, d)$. Almost nothing changes, and it is worth being precise about why.

A linear layer treats $(B, S)$ as **passive** or **outer** dimensions: it acts on the last axis only, independently for each $(b,s)$ pair. So:

- Everything from §1.4 holds with $n$ replaced by the pair $(b,s)$.
- The weight gradient sums over **both**: $dW_{p,q} = \sum_{b}\sum_{s} X_{b,s,p}\,dY_{b,s,q}$.
- In code, this is either `np.tensordot(X, dY, axes=((0,1),(0,1)))` or — equivalently and more commonly — flatten $(B,S,d) \to (BS, d)$ and use the 2-D rule.

This flattening is why the notes and the code use $N = B\cdot S$ so freely. It is not an approximation; it is exact, because $B$ and $S$ genuinely are passive for any per-token operation.

**The exceptions matter.** Two operations in GPT-2 do *not* treat $S$ as passive:

1. **Attention**, which deliberately mixes across $S$. There, $S$ becomes an active dimension and you cannot flatten it.
2. **LayerNorm**, which mixes across $d$ but is independent per $(b,s)$ — so $S$ stays passive there, but $d$ becomes active, which is the reverse of usual and is exactly why LayerNorm's backward pass is the awkward one.

Knowing which dimensions are active and which are passive, for each operation, is most of the battle. Keep a mental table.

## 1.9 Batched matrix multiplication

Attention needs one more atom: matmul where *both* operands are data (no fixed weight matrix), with leading batch dimensions.

**Setup.** $C = A\cdot B$ where $A$ is $(B, n_h, m, k)$ and $B$ is $(B, n_h, k, p)$, giving $C$ of shape $(B, n_h, m, p)$. The leading two axes are passive; the matmul happens on the last two.

**Step 1:** $\;C_{b,h,x,y} = \sum_{r=1}^{k} A_{b,h,x,r}\,B_{b,h,r,y}$

**Step 2–3:** Both $A$ and $B$ are now activations, so both get the fan-out treatment.

$$dA_{b,h,x,r} = \sum_{y=1}^{p} dC_{b,h,x,y}\,B_{b,h,r,y}, \qquad dB_{b,h,r,y} = \sum_{x=1}^{m} dC_{b,h,x,y}\,A_{b,h,x,r}$$

**Step 4:**

$$\boxed{\,dA = dC\cdot B^\top, \qquad dB = A^\top\cdot dC\,}$$

where $\top$ transposes only the **last two** axes, leaving the batch axes alone (`A.transpose(0,1,3,2)` in NumPy, or `A.mT` in modern PyTorch).

Compare to Atom 1: $dX = dY\,W^\top$ and $dW = X^\top dY$. **Identical structure.** The only difference is that in Atom 1 the "batch sum" for $dW$ was a genuine sum over the batch, because $W$ was shared across the batch; here $B$ is *not* shared across the batch, so there is no sum over $b$ — the batch axis stays. That single distinction is the difference between a parameter gradient and an activation gradient, and it is why Step 3 of the Recipe is a separate step.

## 1.10 The atomic backward rules

Every derivation in Chapters 3–9 is these rules, composed.

| Forward | Backward | Rule invoked |
|---|---|---|
| $Y = XW + b$ | $dX = dY W^\top$; $\;dW = X^\top dY$; $\;db = \sum_{\text{batch}} dY$ | fan-out (dX), shared-param (dW, db) |
| $A = \phi(H)$ elementwise | $dH = dA \odot \phi'(H)$ | sum collapses |
| $C = A + B$ | $dA = dC$; $\;dB = dC$ | addition copies gradient |
| $y = x$ used twice | $dx = dy_1 + dy_2$ | Fan-Out Law |
| broadcast along axis $k$ | `sum` along axis $k$ | Broadcasting Rule |
| `sum` along axis $k$ | broadcast along axis $k$ | Broadcasting Rule (dual) |
| $C = A\cdot B$ (both data) | $dA = dC\,B^\top$; $\;dB = A^\top dC$ | fan-out on both |
| $C = A^\top$ | $dA = dC^\top$ | relabelling |
| $Y = \texttt{concat}(A, B)$ | $dA, dB = \texttt{split}(dY)$ | relabelling |
| $Y = \texttt{reshape}(X)$ | $dX = \texttt{reshape}(dY, X.\text{shape})$ | relabelling |
| $Y = \alpha X$, $\alpha$ constant | $dX = \alpha\, dY$ | scaling |
| $Y = X + \text{const}$ | $dX = dY$ | constants vanish |

The last two look trivial and are not: the $\frac{1}{\sqrt{d_k}}$ scaling and the causal mask are exactly these two rows, and Chapter 11 shows that both of them have consequences for training that are anything but trivial.

---
---

# Chapter 2 — GPT-2's forward pass, written out completely

Before we can walk backwards we need an unambiguous map of what we are walking backwards through. This chapter is that map. There is no calculus here; there is a lot of shape bookkeeping, and getting it right now saves you from every confusion later.

## 2.1 The whole model, in order

Input: a batch of token IDs, shape $(B, S)$, integers in $[0, V)$.

**Embedding.**

$$h^{(0)} = E[\text{tokens}] + W_{\text{pos}}[0{:}S] \qquad (B, S, d)$$

$E$ is the token embedding matrix, shape $(V, d)$ — the lookup $E[\text{tokens}]$ selects one row per token. $W_{\text{pos}}$ is the learned positional embedding, shape $(S_{\max}, d)$, added by broadcasting across the batch.

**Then $L$ identical transformer blocks.** GPT-2 uses **pre-normalisation**: LayerNorm sits *inside* each branch, before the sublayer, and the residual addition is the last thing that happens.

For $\ell = 0 \ldots L-1$:

$$\tilde{h} = \text{LN}_1(h^{(\ell)}) \qquad (B,S,d)$$
$$Q = \tilde{h}W_q + b_q, \quad K = \tilde{h}W_k + b_k, \quad V = \tilde{h}W_v + b_v \qquad \text{each } (B,S,d)$$
$$Q_h, K_h, V_h = \text{split\_heads}(Q, K, V) \qquad \text{each } (B, n_h, S, d_k)$$
$$P = Q_h K_h^\top \qquad (B, n_h, S, S)$$
$$\text{scores} = \frac{P}{\sqrt{d_k}} \qquad (B, n_h, S, S)$$
$$\text{scores}^{\text{masked}} = \text{scores} + M, \qquad M_{ij} = \begin{cases}0 & j \le i\\ -\infty & j > i\end{cases}$$
$$A = \text{softmax}_{\text{last axis}}(\text{scores}^{\text{masked}}) \qquad (B, n_h, S, S)$$
$$\text{head} = A\,V_h \qquad (B, n_h, S, d_k)$$
$$\text{concat} = \text{merge\_heads}(\text{head}) \qquad (B, S, d)$$
$$\text{attn\_out} = \text{concat}\cdot W_o + b_o \qquad (B,S,d)$$
$$h^{(\ell+\frac12)} = h^{(\ell)} + \text{attn\_out} \qquad \textbf{residual}$$

$$\tilde{h}_2 = \text{LN}_2(h^{(\ell+\frac12)})$$
$$f_1 = \tilde{h}_2 W_1 + b_1 \qquad (B,S,4d)$$
$$f_{\text{act}} = \text{GELU}(f_1) \qquad (B,S,4d)$$
$$f_2 = f_{\text{act}} W_2 + b_2 \qquad (B,S,d)$$
$$h^{(\ell+1)} = h^{(\ell+\frac12)} + f_2 \qquad \textbf{residual}$$

**Final norm and unembedding.**

$$H_{\text{out}} = \text{LN}_f(h^{(L)}) \qquad (B,S,d)$$
$$Z = H_{\text{out}}E^\top \qquad (B,S,V)$$
$$\hat{P} = \text{softmax}_{\text{last axis}}(Z) \qquad (B,S,V)$$
$$J = -\frac{1}{B\cdot S}\sum_{b=1}^{B}\sum_{s=1}^{S}\sum_{k=1}^{V} Y_{b,s,k}\ln \hat{P}_{b,s,k}$$

Note $Z = H_{\text{out}}E^\top$ uses the **same** $E$ as the input embedding. This is **weight tying**, and it has a consequence for the backward pass that we handle in Chapter 4.

## 2.2 The shape table

Print this out.

| Tensor | Shape | Kind |
|---|---|---|
| tokens | $(B,S)$ | integer input |
| $E$ | $(V,d)$ | parameter (tied: used twice) |
| $W_{\text{pos}}$ | $(S_{\max}, d)$ | parameter |
| $h^{(\ell)}$ | $(B,S,d)$ | activation (the *residual stream*) |
| $\gamma, \beta$ | $(d,)$ | parameters (LayerNorm gain/bias) |
| $W_q, W_k, W_v, W_o$ | $(d,d)$ | parameters |
| $b_q, b_k, b_v, b_o$ | $(d,)$ | parameters |
| $Q, K, V$ | $(B,S,d)$ | activations |
| $Q_h, K_h, V_h$ | $(B,n_h,S,d_k)$ | activations |
| $P$, scores, $A$ | $(B,n_h,S,S)$ | activations |
| head | $(B,n_h,S,d_k)$ | activation |
| concat | $(B,S,d)$ | activation |
| $W_1$ | $(d, 4d)$ | parameter |
| $W_2$ | $(4d, d)$ | parameter |
| $f_1, f_{\text{act}}$ | $(B,S,4d)$ | activations |
| $Z, \hat{P}, Y$ | $(B,S,V)$ | activations / labels |

## 2.3 The residual stream: the spine of the model

Notice something structural. Every block reads from $h$, computes something, and **adds** it back. $h^{(\ell)}$ is never overwritten — it is only ever incremented:

$$h^{(L)} = h^{(0)} + \sum_{\ell} \text{attn\_out}^{(\ell)} + \sum_{\ell} f_2^{(\ell)}$$

The residual stream is a running total, and each block writes a correction into it. This one observation will do a *lot* of work in Chapter 11: it means the path from the loss all the way back to the embeddings includes a **pure identity route**, along which the gradient travels completely unmodified.

## 2.4 What the cache must hold

The backward pass needs values from the forward pass. Cache exactly these per block:

- $h^{(\ell)}$ (the block input) and $h^{(\ell+\frac12)}$ (the mid-block residual)
- LayerNorm's $\hat{x}$ and $\sigma$ for both norms (**not** $\mu$ and $\sigma^2$ separately — see Chapter 5)
- $\tilde{h}$, $\tilde{h}_2$ (the normalised inputs, needed for $dW$)
- $Q_h, K_h, V_h$
- $A$ (the post-softmax attention weights — **not** the pre-softmax scores)
- concat, $f_1$, $f_{\text{act}}$

Plus globally: $\hat{P}$, $H_{\text{out}}$, LN$_f$'s $\hat{x}$ and $\sigma$, and the token IDs.

Two things there are worth flagging now because they are recurring themes. We cache $A$ rather than the scores because **the softmax backward pass is expressible entirely in terms of its own output** (§0.7's gift). And we cache $\hat{x}$ rather than $\mu, \sigma^2$ because the LayerNorm backward pass turns out to be expressible in terms of $\hat{x}$ and $\sigma$ alone.

## 2.5 The route we will walk

We go strictly in reverse:

$$J \to \hat{P} \to Z \to \{H_{\text{out}}, E\} \to h^{(L)} \to \underbrace{\text{block } L\!-\!1 \to \cdots \to \text{block } 0}_{\text{Chapters 5–8, repeated}} \to h^{(0)} \to \{E, W_{\text{pos}}\}$$

Chapters 3 and 4 handle the head. Chapter 5 does LayerNorm — the hardest single derivation in the book, so we do it in extreme slow motion. Chapters 6–8 do the block. Chapter 9 closes the loop at the embeddings.

---
---

# Chapter 3 — The loss and the softmax

**Goal of this chapter:** derive $dZ = \frac{\partial J}{\partial Z}$, the gradient of the loss with respect to the raw logits.

The answer is famously simple. The derivation is not — it is the longest chain of small steps in the book — and it is worth doing in full, because the cancellation at the end is the single best illustration of *why* particular losses are paired with particular output activations.

## 3.1 Setup and the dummy-variable trick

$$Z \in \mathbb{R}^{B\times S\times V}, \qquad \hat{P} \in \mathbb{R}^{B\times S\times V}, \qquad Y \in \mathbb{R}^{B\times S\times V}\ \text{(one-hot)}$$

$$J = -\frac{1}{B\cdot S}\sum_{b=1}^{B}\sum_{s=1}^{S}\sum_{k=1}^{V} Y_{b,s,k}\ln\!\left(\hat{P}_{b,s,k}\right)$$

We want $\dfrac{\partial J}{\partial Z_{b,s,i}}$ — the derivative with respect to **one specific logit**, at batch element $b$, position $s$, vocabulary index $i$.

By the multivariable chain rule (§0.4), $Z_{b,s,i}$ influences $J$ through the probabilities. Which probabilities? *All $V$ of them at that position*, because softmax's denominator contains every logit. So:

$$\frac{\partial J}{\partial Z_{b,s,i}} = \sum_{k=1}^{V}\frac{\partial J}{\partial \hat{P}_{b,s,k}}\cdot\frac{\partial \hat{P}_{b,s,k}}{\partial Z_{b,s,i}}$$

**A notational hygiene point that prevents real errors.** In the loss formula, $b, s, k$ are *bound* summation variables. We are also using $b, s, i$ as *free* indices for the specific logit we are differentiating. Reusing letters here causes genuine mistakes. So rewrite the loss with fresh dummy letters:

$$J = -\frac{1}{B\cdot S}\sum_{i'}\sum_{j'}\sum_{m} Y_{i',j',m}\ln\!\left(\hat{P}_{i',j',m}\right)$$

Now the free indices $(b,s,k)$ are unambiguous.

## 3.2 Step 1 of 2 — the loss with respect to one probability

$$\frac{\partial J}{\partial \hat{P}_{b,s,k}} = -\frac{1}{B\cdot S}\sum_{i'}\sum_{j'}\sum_{m}\frac{\partial}{\partial \hat{P}_{b,s,k}}\Big[Y_{i',j',m}\ln \hat{P}_{i',j',m}\Big]$$

Look at a single term of that triple sum. $Y_{i',j',m}$ is a *constant* (it is data), so it comes out front:

$$Y_{i',j',m}\cdot\frac{1}{\hat{P}_{i',j',m}}\cdot\frac{\partial \hat{P}_{i',j',m}}{\partial \hat{P}_{b,s,k}}$$

using $\frac{d}{dx}\ln x = 1/x$ and the chain rule. And that last factor is a triple Kronecker delta:

$$\frac{\partial \hat{P}_{i',j',m}}{\partial \hat{P}_{b,s,k}} = \begin{cases}1 & \text{if } i'=b,\ j'=s,\ m=k\\ 0 & \text{otherwise}\end{cases}$$

because $\hat{P}_{b,s,k}$ is just one entry of a big tensor, and every *other* entry is, as far as this derivative is concerned, an unrelated variable.

So the triple sum has exactly **one** surviving term (§0.8 — the deltas eat the sums):

$$\boxed{\;\frac{\partial J}{\partial \hat{P}_{b,s,k}} = -\frac{1}{B\cdot S}\cdot\frac{Y_{b,s,k}}{\hat{P}_{b,s,k}}\;}$$

Hold onto that $1/\hat{P}$. It is about to be cancelled by something, and the cancellation is the punchline.

## 3.3 Step 2 of 2 — the softmax Jacobian

$$\hat{P}_{b,s,k} = \frac{e^{Z_{b,s,k}}}{\sum_{j=1}^{V} e^{Z_{b,s,j}}}$$

We need $\dfrac{\partial \hat{P}_{b,s,k}}{\partial Z_{b,s,i}}$.

**Why we cannot just say "zero unless $k=i$".** This is the trap, and the notes flag it explicitly. For an elementwise function we would be done. But softmax is not elementwise: the denominator $\sum_j e^{Z_{b,s,j}}$ contains **every** logit at that position. Nudge $Z_{b,s,i}$ and *every* $\hat{P}_{b,s,k}$ moves — the one with $k=i$ goes up, all the others go down, because the total must stay at 1. So the Jacobian is a full $V\times V$ matrix and we need two cases.

To reduce clutter, fix $(b,s)$ and drop it: write $P_k = e^{Z_k}/\sum_j e^{Z_j}$.

Apply the quotient rule with $u = e^{Z_k}$ and $v = \sum_j e^{Z_j}$:

$$\frac{\partial P_k}{\partial Z_i} = \frac{u'v - uv'}{v^2}, \qquad u' = \frac{\partial e^{Z_k}}{\partial Z_i}, \quad v' = \frac{\partial}{\partial Z_i}\sum_j e^{Z_j} = e^{Z_i}$$

Note $v' = e^{Z_i}$ *always* — regardless of $k$ — because the sum always contains the $j=i$ term. Only $u'$ depends on the case.

### Case A: the diagonal, $k = i$

$u = e^{Z_i}$, so $u' = e^{Z_i}$.

$$\frac{\partial P_i}{\partial Z_i} = \frac{e^{Z_i}\left(\sum_j e^{Z_j}\right) - e^{Z_i}e^{Z_i}}{\left(\sum_j e^{Z_j}\right)^2}$$

Split the fraction deliberately — factor out one copy of $\frac{e^{Z_i}}{\sum_j e^{Z_j}}$:

$$= \frac{e^{Z_i}}{\sum_j e^{Z_j}}\cdot\frac{\sum_j e^{Z_j} - e^{Z_i}}{\sum_j e^{Z_j}} = P_i\left(1 - P_i\right)$$

$$\boxed{\;\frac{\partial P_i}{\partial Z_i} = P_i(1-P_i)\;}$$

### Case B: off-diagonal, $k \neq i$

Now $u = e^{Z_k}$ contains no $Z_i$ at all, so $u' = 0$. Only the denominator moves:

$$\frac{\partial P_k}{\partial Z_i} = \frac{0\cdot v - e^{Z_k}e^{Z_i}}{v^2} = -\frac{e^{Z_k}}{\sum_j e^{Z_j}}\cdot\frac{e^{Z_i}}{\sum_j e^{Z_j}} = -P_kP_i$$

$$\boxed{\;\frac{\partial P_k}{\partial Z_i} = -P_kP_i \quad (k\neq i)\;}$$

**Read what this says.** Raising logit $i$ raises $P_i$ by $P_i(1-P_i)$ and lowers every other $P_k$ by $P_kP_i$. The total change is $P_i(1-P_i) - \sum_{k\neq i}P_kP_i = P_i(1-P_i) - P_i(1-P_i) = 0$. Probability is conserved, exactly as it must be. That the two cases are *forced* to be consistent this way is a good sign your algebra is right.

Both cases at once: $\frac{\partial P_k}{\partial Z_i} = P_k(\delta_{ki} - P_i)$.

## 3.4 Putting them together

Now we split the sum over $k$ into the diagonal term and the rest:

$$\frac{\partial J}{\partial Z_{b,s,i}} = \underbrace{\frac{\partial J}{\partial \hat{P}_{b,s,i}}\cdot\frac{\partial \hat{P}_{b,s,i}}{\partial Z_{b,s,i}}}_{k=i} + \underbrace{\sum_{k\neq i}\frac{\partial J}{\partial \hat{P}_{b,s,k}}\cdot\frac{\partial \hat{P}_{b,s,k}}{\partial Z_{b,s,i}}}_{k\neq i}$$

Substitute everything from §3.2 and §3.3:

$$= -\frac{1}{B\cdot S}\left[\frac{Y_{b,s,i}}{\hat{P}_{b,s,i}}\hat{P}_{b,s,i}\big(1-\hat{P}_{b,s,i}\big) \;+\; \sum_{k\neq i}\frac{Y_{b,s,k}}{\hat{P}_{b,s,k}}\Big(-\hat{P}_{b,s,k}\hat{P}_{b,s,i}\Big)\right]$$

**Here is the cancellation.** In the first term, $\hat{P}_{b,s,i}$ appears in the denominator and the numerator — it cancels. In the sum, $\hat{P}_{b,s,k}$ cancels the same way. The $1/\hat{P}$ from the logarithm annihilates the $\hat{P}$ from the softmax:

$$= -\frac{1}{B\cdot S}\left[Y_{b,s,i}\big(1-\hat{P}_{b,s,i}\big) - \hat{P}_{b,s,i}\sum_{k\neq i}Y_{b,s,k}\right]$$

Expand and regroup around $\hat{P}_{b,s,i}$:

$$= -\frac{1}{B\cdot S}\left[Y_{b,s,i} - \hat{P}_{b,s,i}\Big(\underbrace{Y_{b,s,i} + \sum_{k\neq i}Y_{b,s,k}}_{\text{this is the whole sum over }k}\Big)\right]$$

That bracketed sum is $\sum_{k=1}^{V} Y_{b,s,k}$ — the sum of the entire one-hot vector — which is **exactly 1**. (It is 1 in the true label's slot and 0 elsewhere.)

$$= -\frac{1}{B\cdot S}\Big[Y_{b,s,i} - \hat{P}_{b,s,i}\cdot 1\Big]$$

$$\boxed{\;\frac{\partial J}{\partial Z_{b,s,i}} = \frac{1}{B\cdot S}\Big(\hat{P}_{b,s,i} - Y_{b,s,i}\Big)\;}$$

## 3.5 Tensor form

Every index is free — nothing was summed away — so the tensor form is immediate:

$$\boxed{\;dZ = \frac{1}{B\cdot S}\left(\hat{P} - Y\right) \qquad (B,S,V)\;}$$

```python
probs = softmax(logits, axis=-1)                 # (B,S,V)
Y = one_hot(targets, V)                          # (B,S,V)
dlogits = (probs - Y) / (B * S)
```

## 3.6 Why you should care that it is this clean

**It is the *prediction error*.** $dZ$ is literally "what you predicted minus what was true", scaled by the number of tokens. Gradient descent on the logits is exactly "push the true token's logit up in proportion to how much probability mass you failed to give it, and push every other token's logit down in proportion to how much you wrongly gave it."

**The cancellation is engineered, not lucky.** Softmax's derivative manufactures a factor of $\hat{P}$; cross-entropy's derivative manufactures a factor of $1/\hat{P}$. They were made for each other. Use softmax with mean-squared error instead and the $\hat{P}(1-\hat{P})$ survives — and since $\hat{P}(1-\hat{P}) \to 0$ whenever the model is confident, a *confidently wrong* model gets almost no gradient and cannot recover. Cross-entropy has no such saturation: however confident and wrong you are, $\hat{P} - Y$ stays order 1. This is the whole reason nobody trains classifiers with MSE.

**Two invariants to assert in your code.**

1. $\sum_{i} dZ_{b,s,i} = 0$ for every $(b,s)$. Because $\sum_i \hat{P} = 1$ and $\sum_i Y = 1$. (Measured on the reference implementation: max $|{\sum_i dZ}|$ is at the double-precision round-off floor, $\sim10^{-17}$.) This is not decoration — it says the gradient never tries to change the *overall scale* of the logits, only their differences, which is correct because softmax is invariant to adding a constant to all logits.
2. $|dZ| \le \frac{1}{B\cdot S}$ everywhere, since $\hat{P}, Y \in [0,1]$. The logit gradient is bounded, always. That is a genuinely valuable stability property.

**Practical note.** In real implementations `softmax` subtracts $\max_i Z_{b,s,i}$ before exponentiating. This is not an approximation: softmax is exactly invariant under adding a constant to all logits, since $\frac{e^{Z_k - c}}{\sum_j e^{Z_j - c}} = \frac{e^{-c}e^{Z_k}}{e^{-c}\sum_j e^{Z_j}}$. It prevents $e^{Z}$ overflowing. And because it is an exact invariance, it changes nothing in the backward pass.

---
---

# Chapter 4 — The unembedding, and weight tying

**Goal:** given $dZ$, obtain $dH_{\text{out}}$ and $dE$.

## 4.1 Forward

$$Z = H_{\text{out}}E^\top, \qquad H_{\text{out}} \in \mathbb{R}^{B\times S\times d}, \quad E \in \mathbb{R}^{V\times d}, \quad Z \in \mathbb{R}^{B\times S\times V}$$

### Step 1 — local forward

$$Z_{b,s,i} = \sum_{j=1}^{d}\left(H_{\text{out}}\right)_{b,s,j}\,E_{i,j}$$

Check this against the shapes: $E^\top$ is $(d, V)$, so $Z_{b,s,i} = \sum_j (H_{\text{out}})_{b,s,j}(E^\top)_{j,i}$, and $(E^\top)_{j,i} = E_{i,j}$. ✓ The summed index is $j$, over the embedding dimension $d$. Geometrically: the logit for token $i$ is the **dot product** of the final hidden state with token $i$'s embedding vector.

## 4.2 Gradient with respect to $H_{\text{out}}$

### Step 2 — local derivative

$$\frac{\partial Z_{b,s,i}}{\partial (H_{\text{out}})_{b,s,j}} = E_{i,j}$$

(Trivially, by the delta-collapse of §0.8; and note the $(b,s)$ indices must match, since row $(b,s)$ of $H_{\text{out}}$ only affects row $(b,s)$ of $Z$.)

### Step 3 — sum over what?

$(H_{\text{out}})_{b,s,j}$ is an **activation**. Fan-out rule: it was used to compute *every one of the $V$ logits* at position $(b,s)$. So sum over the output dimension $V$:

$$\frac{\partial J}{\partial (H_{\text{out}})_{b,s,j}} = \sum_{i=1}^{V}\frac{\partial J}{\partial Z_{b,s,i}}\cdot\frac{\partial Z_{b,s,i}}{\partial (H_{\text{out}})_{b,s,j}} = \sum_{i=1}^{V} dZ_{b,s,i}\,E_{i,j}$$

### Step 4 — tensor form

Required shape $(B,S,d)$. Available: $dZ$ is $(B,S,V)$, $E$ is $(V,d)$. The sum contracts $dZ$'s last axis with $E$'s first axis — that is a plain matmul:

$$\boxed{\;dH_{\text{out}} = dZ\cdot E\;} \qquad (B,S,V)\cdot(V,d) \to (B,S,d)\ \checkmark$$

**No transpose.** The forward used $E^\top$; the backward uses $E$. This is the general $dX = dY\,W^\top$ rule with $W = E^\top$, so $W^\top = E$. If that feels slippery, the index expression above is the ground truth and it is unambiguous.

## 4.3 Gradient with respect to $E$

Now $E$ is a **shared parameter** — the same embedding matrix serves every position in every sequence. Shared-parameter rule: sum over $b$ and $s$.

### Steps 2–3

$$\frac{\partial Z_{b,s,i}}{\partial E_{p,q}} = \delta_{ip}\left(H_{\text{out}}\right)_{b,s,q}$$

$$dE_{p,q} = \sum_{b=1}^{B}\sum_{s=1}^{S}\sum_{i=1}^{V} dZ_{b,s,i}\,\delta_{ip}\,(H_{\text{out}})_{b,s,q} = \sum_{b}\sum_{s} dZ_{b,s,p}\,(H_{\text{out}})_{b,s,q}$$

### Step 4

Required shape $(V, d)$. Flatten $(B,S) \to N$: $dZ$ becomes $(N, V)$, $H_{\text{out}}$ becomes $(N,d)$. We contract over $N$:

$$\boxed{\;dE = (dZ)^\top\cdot H_{\text{out}}\;} \qquad (V,N)\cdot(N,d)\to(V,d)\ \checkmark$$

```python
dH_out = dlogits @ E                                            # (B,S,d)
dE     = np.tensordot(dlogits, H_out, axes=((0,1),(0,1)))       # (V,d)
```

## 4.4 The two flavours of "transpose the input"

Compare this with the MLP's $W_2$ gradient, which we will derive in Chapter 7 as $dW_2 = f_{\text{act}}^\top\,df_2$. The notes remark on exactly this and it is worth spelling out, because it confuses everyone once.

- For a standard weight $W$ of shape $(d_{\text{in}}, d_{\text{out}})$, forward is $Y = XW$ and the rule is $dW = X^\top dY$.
- For the embedding $E$ of shape $(V, d)$, forward is $Z = H E^\top$ and the rule is $dE = dZ^\top H$.

They look like mirror images, and they are — because $E$ is stored **transposed relative to how it is used**. In the unembedding, $E$'s $V$ axis plays the role of $d_{\text{out}}$ and its $d$ axis plays the role of $d_{\text{in}}$. Substitute $W = E^\top$ into $dW = X^\top dY$: you get $dE^\top = H^\top dZ$, and transposing both sides gives $dE = dZ^\top H$. ✓ Same rule, different storage layout.

**The lesson: never pattern-match on the shape of a stored parameter. Match on its role in the local forward equation.**

## 4.5 Weight tying: the same matrix, two gradients

GPT-2 uses the **same** $E$ for the input lookup and the output projection. So $E$ fans out into two completely different parts of the graph. The Fan-Out Law (§0.5) is unambiguous:

$$\boxed{\;dE^{\text{total}} = \underbrace{dE^{\text{from unembedding}}}_{\text{this chapter}} + \underbrace{dE^{\text{from lookup}}}_{\text{Chapter 9}}\;}$$

In code, this is precisely the `+=` versus `=` distinction from §0.9:

```python
grads["wte"] += dE_from_unembedding          # Chapter 4
...
np.add.at(grads["wte"], tokens, dh0)         # Chapter 9 — scatter-add
```

Write `=` instead of `+=` in either place and the model still trains — just measurably worse, with no error message. This is the single most common silent bug in hand-written transformer backprop.

**Why tie at all?** Three reasons, and the third is a backward-pass reason. (1) $V\times d$ is $38.6$M parameters for GPT-2 small — around 31% of the model — so tying is a large saving. (2) It enforces a sensible symmetry: the vector that *represents* a token and the vector that *predicts* it should be related. (3) Every token in the vocabulary now receives gradient on **every** step through the unembedding path — whereas the lookup path only gives gradient to tokens that actually appear in the batch. Rare tokens would otherwise be updated very seldom. Tying gives them a steady signal.

---
---

# Chapter 5 — LayerNorm backward

This is the hardest derivation in the book, for one reason: **the normalising statistics depend on the very values you are differentiating**. Change $x_{i,3}$ and you change the mean $\mu_i$, which changes *every* $\hat{x}_{i,j}$ — including $\hat{x}_{i,3}$ itself, and also $\hat{x}_{i,7}$, which has nothing directly to do with channel 3. Three paths, all real, all needing the multivariable chain rule.

We will go extremely slowly. If any chapter deserves a second pass, it is this one.

## 5.1 Forward pass

LayerNorm acts **per token, across the embedding dimension**. Batch and sequence are passive; $d$ is active. So flatten $(B,S) \to N$ and consider a single row $i$ (one token), with channels $j \in \{1,\ldots,d\}$:

$$\mu_i = \frac{1}{d}\sum_{k=1}^{d} x_{i,k}$$
$$\sigma_i^2 = \frac{1}{d}\sum_{k=1}^{d}\left(x_{i,k} - \mu_i\right)^2$$
$$\hat{x}_{i,j} = \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$
$$y_{i,j} = \gamma_j\,\hat{x}_{i,j} + \beta_j$$

$\gamma$ and $\beta$ have shape $(d,)$ and are **broadcast** across all $N$ rows.

Note carefully: $\mu_i$ and $\sigma_i^2$ are per-row **scalars**, computed from *all $d$ channels of that row*. That coupling is the entire difficulty.

## 5.2 The easy gradients: $\gamma$ and $\beta$

$\gamma$ and $\beta$ are shared parameters broadcast over $(B, S)$. By the Broadcasting Rule (§1.6): sum over the broadcast axes.

$$\frac{\partial y_{i,j}}{\partial \gamma_q} = \hat{x}_{i,j}\,\delta_{jq}, \qquad \frac{\partial y_{i,j}}{\partial \beta_q} = \delta_{jq}$$

$$\boxed{\;d\gamma_j = \sum_{i=1}^{N} dy_{i,j}\,\hat{x}_{i,j}, \qquad d\beta_j = \sum_{i=1}^{N} dy_{i,j}\;}$$

In $(B,S,d)$ form:

```python
dgamma = (dy * xhat).sum(axis=(0, 1))    # (d,)
dbeta  = dy.sum(axis=(0, 1))             # (d,)
```

And the gradient with respect to the normalised value, which we will need constantly:

$$\frac{\partial y_{i,j}}{\partial \hat{x}_{i,j}} = \gamma_j \qquad\Longrightarrow\qquad \boxed{\;d\hat{x}_{i,j} = dy_{i,j}\,\gamma_j\;} \qquad\text{i.e.}\quad d\hat{x} = dy\odot\gamma$$

That is straightforward. Now the hard part.

## 5.3 The three paths from $x_{i,j}$

Draw the graph for a single row. The variable $x_{i,j}$ reaches the loss three ways:

```
                   ┌───────────────────────────────────> x̂ᵢⱼ  ────┐
                   │                (direct)                       │
   xᵢⱼ ────────────┼──────> σᵢ² ──────> x̂ᵢ,₁ … x̂ᵢ,d ──────────────┼──> J
                   │              (all d of them)                  │
                   └──────> μᵢ  ──────> x̂ᵢ,₁ … x̂ᵢ,d ──────────────┘
                                  (all d of them)
```

So, by the multivariable chain rule:

$$dx_{i,j} = \underbrace{\frac{\partial J}{\partial \hat{x}_{i,j}}\frac{\partial \hat{x}_{i,j}}{\partial x_{i,j}}}_{\textbf{Path 1: direct}} + \underbrace{\sum_{k=1}^{d}\frac{\partial J}{\partial \hat{x}_{i,k}}\frac{\partial \hat{x}_{i,k}}{\partial \sigma_i^2}\frac{\partial \sigma_i^2}{\partial x_{i,j}}}_{\textbf{Path 2: variance}} + \underbrace{\sum_{k=1}^{d}\frac{\partial J}{\partial \hat{x}_{i,k}}\frac{\partial \hat{x}_{i,k}}{\partial \mu_i}\frac{\partial \mu_i}{\partial x_{i,j}}}_{\textbf{Path 3: mean}}$$

The sums over $k$ in paths 2 and 3 are the fan-out: $\mu_i$ and $\sigma_i^2$ each feed into **all $d$** normalised values in the row, so we must collect gradient from all $d$.

Miss either sum and your LayerNorm gradient will be wrong in a way that still trains — slowly, to a worse optimum. Let us do each path.

## 5.4 Path 1 — the direct route

Treating $\mu_i$ and $\sigma_i^2$ as frozen:

$$\frac{\partial \hat{x}_{i,j}}{\partial x_{i,j}} = \frac{\partial}{\partial x_{i,j}}\left[\frac{x_{i,j}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}\right] = \frac{1}{\sqrt{\sigma_i^2+\epsilon}}$$

$$\boxed{\;\text{Path 1} = \frac{d\hat{x}_{i,j}}{\sqrt{\sigma_i^2+\epsilon}}\;}$$

## 5.5 Path 2 — through the variance

Two pieces.

**Piece A: how $\hat{x}_{i,k}$ depends on $\sigma_i^2$.** Write $\hat{x}_{i,k} = (x_{i,k}-\mu_i)(\sigma_i^2+\epsilon)^{-1/2}$ and differentiate with respect to $\sigma_i^2$ using the power rule (treating $(x_{i,k}-\mu_i)$ as constant):

$$\frac{\partial \hat{x}_{i,k}}{\partial \sigma_i^2} = (x_{i,k}-\mu_i)\cdot\left(-\tfrac{1}{2}\right)(\sigma_i^2+\epsilon)^{-3/2} = -\frac{1}{2}\,(x_{i,k}-\mu_i)\,(\sigma_i^2+\epsilon)^{-3/2}$$

So, collecting from all $d$ channels:

$$d\sigma_i^2 = \sum_{k=1}^{d} d\hat{x}_{i,k}\cdot\left[-\tfrac{1}{2}(x_{i,k}-\mu_i)(\sigma_i^2+\epsilon)^{-3/2}\right]$$

**Simplify using $\hat{x}$.** Note $(x_{i,k}-\mu_i) = \hat{x}_{i,k}\sqrt{\sigma_i^2+\epsilon}$. Substituting:

$$d\sigma_i^2 = -\frac{1}{2}\sum_{k} d\hat{x}_{i,k}\,\hat{x}_{i,k}\,\frac{\sqrt{\sigma_i^2+\epsilon}}{(\sigma_i^2+\epsilon)^{3/2}} = -\frac{1}{2(\sigma_i^2+\epsilon)}\sum_{k} d\hat{x}_{i,k}\,\hat{x}_{i,k}$$

This is why we cache $\hat{x}$: it absorbs the $(x-\mu)$ terms and one power of the standard deviation.

**Piece B: how $\sigma_i^2$ depends on $x_{i,j}$.**

$$\sigma_i^2 = \frac{1}{d}\sum_{k=1}^{d}(x_{i,k}-\mu_i)^2 \qquad\Longrightarrow\qquad \frac{\partial \sigma_i^2}{\partial x_{i,j}} = \frac{2}{d}(x_{i,j}-\mu_i)$$

> **A caveat worth stating.** Strictly, $\mu_i$ *also* depends on $x_{i,j}$, so a fully rigorous derivative of $\sigma^2$ would include that. It works out that the extra term is $-\frac{2}{d}\left(\frac{1}{d}\sum_k (x_{i,k}-\mu_i)\right) = 0$, because deviations from the mean sum to zero *by construction*. So the simple answer is exactly right. We are structuring the graph as $x \to (\mu, \sigma^2) \to \hat{x}$ with $\mu$ and $\sigma^2$ treated as siblings, which the notes do too, and the zero-sum identity is what makes that legitimate. Worth knowing why it's legitimate rather than just doing it.

$$\boxed{\;\text{Path 2} = d\sigma_i^2\cdot\frac{2}{d}(x_{i,j}-\mu_i)\;}$$

## 5.6 Path 3 — through the mean

**Piece A:**

$$\frac{\partial \hat{x}_{i,k}}{\partial \mu_i} = \frac{-1}{\sqrt{\sigma_i^2+\epsilon}} \qquad\Longrightarrow\qquad d\mu_i = \sum_{k=1}^{d} d\hat{x}_{i,k}\cdot\frac{-1}{\sqrt{\sigma_i^2+\epsilon}} = \frac{-1}{\sqrt{\sigma_i^2+\epsilon}}\sum_{k} d\hat{x}_{i,k}$$

**Piece B:**

$$\mu_i = \frac{1}{d}\sum_k x_{i,k} \qquad\Longrightarrow\qquad \frac{\partial \mu_i}{\partial x_{i,j}} = \frac{1}{d}$$

$$\boxed{\;\text{Path 3} = \frac{d\mu_i}{d}\;}$$

## 5.7 Assembling the three paths

Write $s \equiv \sqrt{\sigma_i^2+\epsilon}$ for brevity, so $\sigma_i^2 + \epsilon = s^2$.

$$dx_{i,j} = \underbrace{\frac{d\hat{x}_{i,j}}{s}}_{P_1} \;+\; \underbrace{\left[-\frac{1}{2s^2}\sum_k d\hat{x}_{i,k}\hat{x}_{i,k}\right]\cdot\frac{2}{d}\left(x_{i,j}-\mu_i\right)}_{P_2} \;+\; \underbrace{\frac{1}{d}\left[\frac{-1}{s}\sum_k d\hat{x}_{i,k}\right]}_{P_3}$$

Simplify $P_2$, using $(x_{i,j}-\mu_i) = \hat{x}_{i,j}\,s$ once more. The $2$ and the $\frac12$ cancel:

$$P_2 = -\frac{1}{s^2}\cdot\frac{1}{d}\cdot \hat{x}_{i,j}\,s\sum_k d\hat{x}_{i,k}\hat{x}_{i,k} = -\frac{\hat{x}_{i,j}}{d\,s}\sum_k d\hat{x}_{i,k}\hat{x}_{i,k}$$

Now every term carries a $\frac{1}{s}$. Factor out $\frac{1}{d\,s}$:

$$dx_{i,j} = \frac{1}{d\,s}\left[\,d\cdot d\hat{x}_{i,j} \;-\; \hat{x}_{i,j}\sum_{k=1}^{d} d\hat{x}_{i,k}\,\hat{x}_{i,k} \;-\; \sum_{k=1}^{d} d\hat{x}_{i,k}\right]$$

$$\boxed{\;dx_{i,j} = \frac{1}{d\sqrt{\sigma_i^2+\epsilon}}\left[\,d\cdot d\hat{x}_{i,j} \;-\; \sum_{k=1}^{d} d\hat{x}_{i,k} \;-\; \hat{x}_{i,j}\sum_{k=1}^{d} d\hat{x}_{i,k}\hat{x}_{i,k}\right]\;}$$

That is the LayerNorm backward pass. Three terms, one from each path.

## 5.8 Tensor form

Shapes: $dy, x, \hat{x} \in (B,S,d)$; $\gamma,\beta \in (d,)$; $\sigma^2 \in (B,S,1)$.

$$d\hat{x} = dy\odot\gamma \qquad (B,S,d)$$
$$\textstyle\sum_k d\hat{x}_{i,k} \;\to\; \texttt{sum}(d\hat{x}) \text{ over last axis} \qquad (B,S,1)$$
$$\textstyle\sum_k d\hat{x}_{i,k}\hat{x}_{i,k} \;\to\; \texttt{sum}(d\hat{x}\odot\hat{x}) \text{ over last axis} \qquad (B,S,1)$$

$$\boxed{\;dx = \frac{1}{d\cdot\text{std}}\odot\Big(\underbrace{d\cdot d\hat{x}}_{(B,S,d)} - \underbrace{\texttt{sum}(d\hat{x})}_{(B,S,1)} - \underbrace{\hat{x}\odot \texttt{sum}(d\hat{x}\odot\hat{x})}_{(B,S,d)}\Big)\;}$$

The two $(B,S,1)$ terms broadcast back across $d$ — which, by the Broadcasting Rule, is exactly the dual of the sum that produced them. Forward: one statistic broadcast to $d$ channels. Backward: $d$ channels summed to one statistic, then broadcast again. The symmetry is not a coincidence; it is the Fan-Out Law showing its face.

```python
def layernorm_backward(dy, xhat, std, gamma):
    d = xhat.shape[-1]
    dgamma = (dy * xhat).sum(axis=(0, 1))
    dbeta  = dy.sum(axis=(0, 1))
    dxhat  = dy * gamma
    sum_dxhat      = dxhat.sum(-1, keepdims=True)
    sum_dxhat_xhat = (dxhat * xhat).sum(-1, keepdims=True)
    dx = (d * dxhat - sum_dxhat - xhat * sum_dxhat_xhat) / (d * std)
    return dx, dgamma, dbeta
```

## 5.9 Reading the formula: LayerNorm is a projection

This formula has structure worth naming, because Chapter 11 uses it twice.

Rewrite it. In vector form for one row, with $\mathbf{1}$ the all-ones vector and noting $\hat{x}^\top\hat{x} = d$ (the normalised vector has unit variance and zero mean, so its squared norm is exactly $d$):

$$dx = \frac{1}{s}\left[\,d\hat{x} - \frac{1}{d}\mathbf{1}\left(\mathbf{1}^\top d\hat{x}\right) - \frac{1}{d}\hat{x}\left(\hat{x}^\top d\hat{x}\right)\right] = \frac{1}{s}\left(I - \frac{\mathbf{1}\mathbf{1}^\top}{d} - \frac{\hat{x}\hat{x}^\top}{d}\right)d\hat{x}$$

The matrix in brackets is $I$ minus two **orthogonal projectors**: one onto the all-ones direction $\mathbf{1}$, one onto the $\hat{x}$ direction. And $\mathbf{1} \perp \hat{x}$, because $\hat{x}$ has zero mean by construction. So this is an orthogonal projection that **annihilates two directions**, scaled by $1/s$.

Three consequences you can verify by inspection:

1. $\mathbf{1}^\top dx = 0$: **the incoming gradient's mean component is removed.** LayerNorm refuses to pass back any gradient that would just shift all channels of a token by a constant — sensibly, since such a shift has no effect on the output.
2. $\hat{x}^\top dx = 0$: **the component that would just rescale the token is also removed.** Again sensible: LayerNorm's output is invariant to the input's scale.
3. **The Jacobian has rank exactly $d-2$**, with two zero singular values and the remaining $d-2$ all equal to $1/s$.

I verified (3) numerically. Building $\partial\hat{x}/\partial x$ by finite differences for $d = 8, 16, 64$:

| $d$ | rank | two smallest singular values | largest singular value | $1/s$ |
|---|---|---|---|---|
| 8 | 6 ( $=d-2$ ) | $3.4\times10^{-10}$, $1.1\times10^{-10}$ | 0.6791 | 0.6791 |
| 16 | 14 | $4.6\times10^{-10}$, $8.9\times10^{-12}$ | 0.5331 | 0.5331 |
| 64 | 62 | $1.2\times10^{-9}$, $7.2\times10^{-11}$ | 0.5508 | 0.5508 |

and $J\mathbf{1} = 0$, $J\hat{x} = 0$ to $10^{-9}$ in all cases. The closed form matches the numerical Jacobian to $4\times10^{-10}$.

**This is the fact behind pre-norm vs post-norm.** LayerNorm's Jacobian is *rank-deficient* and *scaled by $1/\sigma$*. Put one in the path of every gradient and you multiply by a rank-deficient matrix $L$ times. Put it inside a residual branch instead and the identity path routes around it entirely. Chapter 11.2 makes that argument precisely.

**One more practical consequence.** The $1/s$ factor means **LayerNorm automatically damps gradients into high-variance tokens**. A token whose activations have blown up gets $\sigma$ large, hence $1/\sigma$ small, hence a small gradient. It is a built-in, free, per-token adaptive learning rate. That is a large part of why transformers are as trainable as they are.

---
---

# Chapter 6 — Residual connections

The shortest chapter, and the most consequential.

## 6.1 Forward

$$h^{(\ell+1)} = h^{(\ell+\frac12)} + f_2$$

## 6.2 Backward

Start from the general fact. If $C = A + B$ then

$$\frac{\partial J}{\partial A} = \frac{\partial J}{\partial C}\cdot\frac{\partial C}{\partial A} = \frac{\partial J}{\partial C}\cdot 1 = \frac{\partial J}{\partial C}$$

and identically for $B$. So:

$$\boxed{\;dh^{(\ell+\frac12)} = dh^{(\ell+1)}, \qquad df_2 = dh^{(\ell+1)}\;}$$

**Addition copies the gradient, unchanged, into both branches.** No scaling, no transposition, nothing.

## 6.3 The other half: the join is a fan-out

Here is the part people get wrong. Going *forward*, $h^{(\ell+\frac12)}$ is used in two places: it goes into the LayerNorm-MLP branch, **and** it goes straight into the addition. That is a fan-out. So going *backward*, gradients from both must be summed:

$$\boxed{\;dh^{(\ell+\frac12)}_{\text{total}} = \underbrace{dh^{(\ell+1)}}_{\text{via the skip}} + \underbrace{dh^{(\ell+\frac12)}_{\text{from MLP}}}_{\text{via LN}_2\to W_1\to\text{GELU}\to W_2}\;}$$

The notes state this explicitly, and it is worth restating: **you compute $dh$ from the skip path early and must remember to add it back after the branch gradient has been computed.** In code this is why the residual gradient is held in a temporary:

```python
d_res = dh              # the skip path — gradient passes straight through
df2   = dh              # the branch path — same value, different destiny
...                     # walk back through W2, GELU, W1, LN2
dh_mid = dh_from_branch + d_res       # <-- the Fan-Out Law
```

Forget that `+ d_res` and the model still runs. It will just train as though it had no residual connections, and you will spend a day wondering why the loss plateaus.

## 6.4 Why this matters more than it looks

Unroll the whole stack:

$$h^{(L)} = h^{(0)} + \sum_{\ell=0}^{L-1}\Big(\text{attn\_out}^{(\ell)} + f_2^{(\ell)}\Big)$$

Differentiate the residual stream at layer $\ell$ with respect to the one below:

$$\frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}} = I + \frac{\partial \mathcal{F}(h^{(\ell)})}{\partial h^{(\ell)}}$$

**There is an exact identity matrix in there.** The end-to-end Jacobian is

$$\frac{\partial h^{(L)}}{\partial h^{(0)}} = \prod_{\ell}\left(I + J_{\mathcal{F}}^{(\ell)}\right) = I + \sum_{\ell}J_{\mathcal{F}}^{(\ell)} + \sum_{\ell<m}J_{\mathcal{F}}^{(m)}J_{\mathcal{F}}^{(\ell)} + \cdots$$

The leading term is $I$, regardless of depth, regardless of weight scale, regardless of what the branches do. **The gradient can never be destroyed by the residual path**, because there is always a route back to the embeddings along which it is multiplied by nothing at all. Chapter 11.3 turns this into a measurement.

---
---

# Chapter 7 — The MLP block

$$\tilde{h}_2 = \text{LN}_2(h^{(\ell+\frac12)}) \;\to\; f_1 = \tilde{h}_2W_1 + b_1 \;\to\; f_{\text{act}} = \text{GELU}(f_1) \;\to\; f_2 = f_{\text{act}}W_2 + b_2$$

with $W_1: (d, 4d)$ and $W_2: (4d, d)$. We walk backwards, so we start with $W_2$.

## 7.1 The second linear layer

### Step 1 — local forward

$$(f_2)_{b,s,j} = \sum_{m=1}^{d_{\text{in}}} (f_{\text{act}})_{b,s,m}\,(W_2)_{m,j} + (b_2)_j, \qquad d_{\text{in}} = 4d,\ j \in [1, d]$$

### Steps 2–3 — for $W_2$ (shared parameter → sum over batch and sequence)

$$\frac{\partial (f_2)_{b,s,j}}{\partial (W_2)_{p,q}} = (f_{\text{act}})_{b,s,p}\,\delta_{jq}$$

$$d(W_2)_{p,q} = \sum_{b}\sum_{s}\sum_{j} (df_2)_{b,s,j}\,(f_{\text{act}})_{b,s,p}\,\delta_{jq} = \sum_{b}\sum_{s}(df_2)_{b,s,q}\,(f_{\text{act}})_{b,s,p}$$

**The caveat that the notes flag.** $W_2$ has shape $(d_{\text{in}}, d_{\text{out}})$, so its **rows** index the input dimension and its **columns** index the output dimension. The gradient arriving from above indexes the *output* dimension $j$. So when you assemble the matmul, the incoming gradient must land on $W_2$'s **column** axis and the saved activation on its **row** axis. Get this backwards and you produce $dW_2^\top$ — which, if $d_{\text{in}} = d_{\text{out}}$, will not even raise a shape error. (For $W_1$ and $W_2$ in the MLP the dimensions differ, so you would catch it; for $W_q, W_k, W_v, W_o$, which are all $(d,d)$, **you would not**. This is exactly the case §1.4 warned about.)

### Step 4 — tensor form

$$\boxed{\;dW_2 = f_{\text{act}}^\top\, df_2\;} \qquad (4d, N)\cdot(N, d) \to (4d, d)\ \checkmark$$
$$\boxed{\;db_2 = \sum_{b,s} (df_2)_{b,s,:}\;} \qquad (d,)$$
$$\boxed{\;df_{\text{act}} = df_2\, W_2^\top\;} \qquad (B,S,d)\cdot(d,4d)\to(B,S,4d)\ \checkmark$$

For the third one, the index derivation (fan-out rule — sum over the $d_{\text{out}}$ dimension):

$$\frac{\partial (f_2)_{b,s,j}}{\partial (f_{\text{act}})_{b,s,i}} = (W_2)_{i,j} \qquad\Longrightarrow\qquad (df_{\text{act}})_{b,s,i} = \sum_{j=1}^{d_{\text{out}}} (df_2)_{b,s,j}\,(W_2)_{i,j}$$

Contracting $df_2$'s last axis with $W_2$'s **second** axis is $df_2 W_2^\top$. ✓

## 7.2 GELU

GPT-2 uses the tanh approximation:

$$\text{GELU}(x) = 0.5\,x\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right)\right)$$

### Local derivative

It is elementwise, so (Atom 2, §1.5) the sum collapses and we only need the scalar derivative. Let

$$u(x) = \sqrt{\tfrac{2}{\pi}}\left(x + 0.044715\,x^3\right), \qquad t = \tanh(u)$$

so $\text{GELU}(x) = 0.5\,x\,(1+t)$. Apply the **product rule** to $0.5x \cdot (1+t)$:

$$\text{GELU}'(x) = 0.5(1+t) + 0.5x\cdot\frac{dt}{dx}$$

and $\frac{dt}{dx} = (1-t^2)\cdot u'(x)$ by the chain rule and §0.7, with

$$u'(x) = \sqrt{\tfrac{2}{\pi}}\left(1 + 3\cdot 0.044715\,x^2\right)$$

Therefore:

$$\boxed{\;\text{GELU}'(x) = 0.5\left(1+\tanh u\right) + 0.5\,x\left(1-\tanh^2 u\right)\sqrt{\tfrac{2}{\pi}}\left(1 + 0.134145\,x^2\right)\;}$$

### Tensor form

$$\boxed{\;df_1 = df_{\text{act}} \odot \text{GELU}'(f_1)\;}$$

Note the argument: $f_1$, the **pre-activation**, which is why it must be cached.

```python
def gelu_grad(x):
    u  = np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
    t  = np.tanh(u)
    du = np.sqrt(2/np.pi) * (1 + 3*0.044715 * x**2)
    return 0.5*(1 + t) + 0.5*x*(1 - t**2)*du
```

### A remark for Chapter 11

Unlike ReLU, $\text{GELU}'$ is never exactly zero for $x < 0$ — it is small and negative around $x \approx -1$, crosses zero near $x \approx -0.75$, and decays smoothly. There are no dead units. Chapter 11.6 explains why that matters more for a transformer than for a CNN.

## 7.3 The first linear layer

Identical structure to §7.1, with $\tilde{h}_2 \to f_1$ and $W_1: (d, 4d)$:

$$\boxed{\;dW_1 = \tilde{h}_2^\top\,df_1, \qquad db_1 = \sum_{b,s}(df_1)_{b,s,:}, \qquad d\tilde{h}_2 = df_1\,W_1^\top\;}$$

Shapes: $(d, N)\cdot(N, 4d) \to (d, 4d)$ ✓ ; $(4d,)$ ✓ ; $(B,S,4d)\cdot(4d,d)\to(B,S,d)$ ✓

## 7.4 And back through LN$_2$

Feed $d\tilde{h}_2$ into the LayerNorm backward of Chapter 5, which returns $dh_{\text{from MLP}}$, $d\gamma_2$, $d\beta_2$. Then apply §6.3:

$$dh^{(\ell+\frac12)} = dh_{\text{from MLP}} + d_{\text{res}}$$

The MLP is complete. In code, the whole sub-block:

```python
d_res = dh; df2 = dh
dfa,  dW2, db2 = linear_backward(df2, f_act, W2)
df1  = dfa * gelu_grad(f1)
dhn,  dW1, db1 = linear_backward(df1, h_norm2, W1)
dh_mid, dg2, dbeta2 = layernorm_backward(dhn, *ln2_cache)
dh_mid = dh_mid + d_res              # Fan-Out Law
```

---
---

# Chapter 8 — Attention

Attention has the most moving parts, but every part is an atom you already know. We walk backwards through:

$$\text{output projection} \to \text{concat/split} \to \text{head} = AV \to \text{softmax} \to \text{mask} \to \text{scale} \to QK^\top \to \text{QKV projections}$$

## 8.1 The output projection

$$\text{MHSA}_{\text{out}} = \text{concat}\cdot W_o + b_o$$

### Step 1

$$(\text{MHSA}_{\text{out}})_{b,s,j} = \sum_{m=1}^{d}(\text{concat})_{b,s,m}(W_o)_{m,j} + (b_o)_j$$

### Steps 2–4

Pure Atom 1. $W_o$ and $b_o$ are shared parameters (sum over $b,s$); concat is an activation (sum over the output dimension $j$).

$$\boxed{\;dW_o = \text{concat}^\top\, d\text{MHSA}_{\text{out}}, \qquad db_o = \sum_{b,s} d\text{MHSA}_{\text{out}}, \qquad d\text{concat} = d\text{MHSA}_{\text{out}}\,W_o^\top\;}$$

## 8.2 Concat and split: pure relabelling

$$\text{concat} = \big[\text{head}_0 \,\|\, \text{head}_1 \,\|\, \cdots \,\|\, \text{head}_{n_h-1}\big]$$

Concatenation moves numbers without changing them. Its gradient is a **slice**:

$$\boxed{\;d(\text{head}_i) = d(\text{concat})[\,\cdot,\cdot,\; i\cdot d_k : (i+1)\cdot d_k\,]\;}$$

shape $(B, S, d_k)$ per head, or in one tensor operation:

```python
dhead = dconcat.reshape(B, S, nh, dk).transpose(0, 2, 1, 3)     # (B,nh,S,dk)
```

That is the exact inverse of `merge_heads`. **Every reshape/transpose in the forward pass becomes its inverse reshape/transpose in the backward pass, with no arithmetic.** If your head-splitting is wrong in the backward pass, the model trains but the heads get scrambled — a bug that produces no error and no obvious symptom, so it is worth a unit test.

## 8.3 The value-weighted sum: $\text{head}_i = A_i V_i$

Per head $i$: $A_i \in (B,S,S)$, $V_i \in (B,S,d_k)$, $\text{head}_i \in (B,S,d_k)$.

### Step 1 — local forward, with the dimensions named

$$(\text{head}_i)_{b,s,m} = \sum_{c=1}^{S}(A_i)_{b,s,c}\,(V_i)_{b,c,m}$$

The notes make an observation here that is worth adopting permanently. Map this onto the linear-layer template $Y = XW$:

| Template | Here |
|---|---|
| $N$ (rows) | $S$ (query positions) |
| $d_{\text{in}}$ (summed) | $S$ (key positions) — the index $c$ |
| $d_{\text{out}}$ | $d_k$ |
| passive/outer | $B$ and the head index |

**$S$ appears as both the row count and the summed dimension.** That is why blind shape-matching fails here — several wrong arrangements conform. The index expression above is the only reliable guide.

### Steps 2–3

Both $A_i$ and $V_i$ are activations, so both get the fan-out treatment (§1.9).

**For $V_i$:** the element $(V_i)_{b,c,m}$ is used by **every query position $s$** (each of the $S$ queries attends to key $c$). So sum over $s$:

$$d(V_i)_{b,c,m} = \sum_{s=1}^{S} d(\text{head}_i)_{b,s,m}\,(A_i)_{b,s,c}$$

**For $A_i$:** the element $(A_i)_{b,s,c}$ is used to produce **all $d_k$ channels** of head output at position $s$. So sum over $m$:

$$d(A_i)_{b,s,c} = \sum_{m=1}^{d_k} d(\text{head}_i)_{b,s,m}\,(V_i)_{b,c,m}$$

### Step 4

$$\boxed{\;dV_i = A_i^\top\,d(\text{head}_i)\;}\qquad (B,S,S)^\top\cdot(B,S,d_k)\to(B,S,d_k)\ \checkmark$$
$$\boxed{\;dA_i = d(\text{head}_i)\,V_i^\top\;}\qquad (B,S,d_k)\cdot(B,d_k,S)\to(B,S,S)\ \checkmark$$

with $\top$ transposing only the last two axes. Compare Atom 1: $dW = X^\top dY$ and $dX = dY W^\top$. Same shapes, same structure — but here **there is no sum over the batch**, because $V_i$ is data, not a shared parameter. That distinction is the whole content of Step 3 of the Recipe.

```python
dV_h = A.transpose(0,1,3,2) @ dhead      # (B,nh,S,dk)
dA   = dhead @ V_h.transpose(0,1,3,2)    # (B,nh,S,S)
```

## 8.4 Softmax over attention weights

$$A_{b,s,c} = \frac{e^{Z_{b,s,c}}}{\sum_{k=1}^{S} e^{Z_{b,s,k}}}$$

(dropping the head index; $Z$ here means the masked, scaled scores.) This is structurally identical to Chapter 3's softmax, but with one crucial difference: **there is no cross-entropy loss sitting on top to cancel things**. The gradient $dA$ arrives from $\text{head} = AV$, so we must carry the full Jacobian through.

### Local derivatives — same two cases as §3.3

$$\frac{\partial A_{b,s,c}}{\partial Z_{b,s,c}} = A_{b,s,c}(1 - A_{b,s,c}), \qquad \frac{\partial A_{b,s,k}}{\partial Z_{b,s,c}} = -A_{b,s,k}A_{b,s,c}\ \ (k \neq c)$$

### Combining

$$dZ_{b,s,c} = \sum_{k=1}^{S} dA_{b,s,k}\cdot\frac{\partial A_{b,s,k}}{\partial Z_{b,s,c}}$$

Split off $k = c$:

$$= dA_{b,s,c}\,A_{b,s,c}(1-A_{b,s,c}) + \sum_{k\neq c} dA_{b,s,k}\left(-A_{b,s,k}A_{b,s,c}\right)$$

Factor out $A_{b,s,c}$ from both:

$$= A_{b,s,c}\left[dA_{b,s,c}(1 - A_{b,s,c}) - \sum_{k\neq c}dA_{b,s,k}A_{b,s,k}\right]$$

$$= A_{b,s,c}\left[dA_{b,s,c} - dA_{b,s,c}A_{b,s,c} - \sum_{k\neq c}dA_{b,s,k}A_{b,s,k}\right]$$

The term $dA_{b,s,c}A_{b,s,c}$ is exactly the missing $k=c$ term of the sum. Absorb it:

$$\boxed{\;dZ_{b,s,c} = A_{b,s,c}\left(dA_{b,s,c} - \sum_{k=1}^{S} dA_{b,s,k}A_{b,s,k}\right)\;}$$

### Tensor form

$$\boxed{\;dZ = A\odot\left(dA - \texttt{sum}_{\text{last dim}}(dA\odot A)\right)\;}$$

Shapes: $A, dA \in (B,n_h,S,S)$; the sum is $(B,n_h,S,1)$ and broadcasts back. ✓

```python
dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
```

**Two properties, both verified numerically and both used in Chapter 11:**

1. **Rows sum to zero.** $\sum_c dZ_{b,s,c} = \sum_c A dA - (\sum_c A)(\sum_k dA_kA_k) = \sum_c A\,dA - 1\cdot\sum_k dA_kA_k = 0$. (Measured: $\sim10^{-16}$, i.e. round-off.) Softmax outputs live on the probability simplex, so their gradients live in the simplex's tangent space.
2. **The factor of $A$ out front is a gradient throttle.** Wherever $A_{b,s,c} \approx 0$, $dZ_{b,s,c} \approx 0$. A saturated attention distribution receives almost no gradient — the single most important fact in Chapter 11.1.

## 8.5 The causal mask — and the leakage proof

### Forward

$$Z = \text{scores} + M, \qquad M_{s,c} = \begin{cases}0 & c \le s\\ -\infty & c > s\end{cases}$$

(implemented as a large negative number such as $-10^9$).

### Backward

$M$ is a **constant**. From the atom table, $Y = X + \text{const} \Rightarrow dX = dY$:

$$\boxed{\;d(\text{scores}) = dZ\;}$$

### The claim the notes ask us to prove

> **Claim.** The causal mask prevents gradient from leaking into future tokens: $d(\text{scores})_{b,s,c} = 0$ exactly, for every $c > s$.

**Proof.** For $c > s$, the forward value is $Z_{b,s,c} = \text{scores}_{b,s,c} - \infty$ (or $-10^9$). Hence

$$A_{b,s,c} = \frac{e^{-\infty}}{\sum_k e^{Z_{b,s,k}}} = \frac{0}{\text{something positive}} = 0$$

exactly. Now look at the softmax backward formula from §8.4:

$$dZ_{b,s,c} = A_{b,s,c}\left(dA_{b,s,c} - \sum_k dA_{b,s,k}A_{b,s,k}\right)$$

The factor $A_{b,s,c}$ multiplies the *entire* expression. With $A_{b,s,c} = 0$:

$$dZ_{b,s,c} = 0 \cdot (\text{anything finite}) = 0 \qquad\blacksquare$$

And since $d(\text{scores}) = dZ$, no gradient reaches the masked scores; hence none reaches $Q_{b,s}$ or $K_{b,c}$ *through that pair*. Position $s$ neither reads from nor learns from position $c > s$.

**Two things to appreciate about this proof.**

First, **the masking mechanism is self-consistent between the passes**. We did not add a second mask in the backward pass. The forward mask sets $A = 0$, and the softmax backward formula happens to have $A$ as a multiplicative prefactor, so the block propagates itself. This is elegant, and it is why you will not find a "mask the gradient" line in any transformer implementation.

Second, **it depends on $A$ being *exactly* zero.** With $-10^9$ instead of $-\infty$, $e^{-10^9 - \max}$ underflows to exactly $0.0$ in float32 and float64, so the property holds exactly in practice too. (Verified: max attention weight on any masked position, and max $|d\text{scores}|$ on masked entries, both exactly $0.0$.) But if you used a *small* mask value — say $-10$ — you would get $A \approx 7\times10^{-5}$ rather than 0 (measured in §11.4), and a tiny but nonzero gradient would flow backwards from the future. The model would leak. **The magnitude of your mask constant is a correctness property, not a numerical detail.**

## 8.6 The $\sqrt{d_k}$ scaling

$$\text{scores} = \frac{P}{\sqrt{d_k}}, \qquad P = Q_hK_h^\top$$

$\sqrt{d_k}$ is a constant, so from the atom table:

$$\boxed{\;dP = \frac{1}{\sqrt{d_k}}\,d(\text{scores})\;}$$

Trivially small in the backward pass — one scalar multiply. Chapter 11.1 shows this one line is the difference between a transformer that trains and one that does not.

## 8.7 The score matrix $P = Q_hK_h^\top$

### Step 1 — local forward

$$P_{b,s,c} = \sum_{m=1}^{d_k} (Q_h)_{b,s,m}\,(K_h^\top)_{b,m,c} = \sum_{m=1}^{d_k} (Q_h)_{b,s,m}\,(K_h)_{b,c,m}$$

(head index suppressed.) Map onto the template:

| Template | Here |
|---|---|
| $N$ (rows) | $S$ (query positions) |
| $d_{\text{in}}$ (summed) | $d_k$ — the index $m$ |
| $d_{\text{out}}$ | $S$ (key positions) |

Note this is the **mirror image** of §8.3: there $S$ was summed and $d_k$ was the output; here $d_k$ is summed and $S$ is the output. Same two dimensions, opposite roles. This is precisely the situation where shape-matching alone fails.

### Steps 2–3

Both operands are activations.

**For $Q_h$:** the element $(Q_h)_{b,s,m}$ is used against **every key position $c$**, producing $S$ scores. Fan-out → sum over $c$:

$$d(Q_h)_{b,s,m} = \sum_{c=1}^{S} dP_{b,s,c}\,(K_h)_{b,c,m}$$

**For $K_h$:** the element $(K_h)_{b,c,m}$ is used by **every query position $s$**. Fan-out → sum over $s$:

$$d(K_h)_{b,c,m} = \sum_{s=1}^{S} dP_{b,s,c}\,(Q_h)_{b,s,m}$$

### Step 4

$$\boxed{\;dQ_h = dP\cdot K_h\;} \qquad (B,n_h,S,S)\cdot(B,n_h,S,d_k)\to(B,n_h,S,d_k)\ \checkmark$$
$$\boxed{\;dK_h = dP^\top\cdot Q_h\;} \qquad (B,n_h,S,S)^\top\cdot(B,n_h,S,d_k)\to(B,n_h,S,d_k)\ \checkmark$$

**No transpose on $K_h$ in the first one** — because the forward already applied one. The notes derive this as $dK^\top = ((dP)\cdot Q^\top)$ and then transpose to get $dK = (dP)^\top Q$; same answer, and the index expressions above confirm it.

The asymmetry between the two ($dP$ vs $dP^\top$) is the mathematical statement that **attention is not symmetric**: query $s$ attending to key $c$ is a different event from query $c$ attending to key $s$, and after masking, one of them may not even be allowed.

```python
dQ_h = dP @ K_h                          # (B,nh,S,dk)
dK_h = dP.transpose(0,1,3,2) @ Q_h       # (B,nh,S,dk)
```

## 8.8 Merging heads and the Q, K, V projections

Merge the per-head gradients back:

```python
dQ = merge_heads(dQ_h)      # (B,S,d)
dK = merge_heads(dK_h)
dV = merge_heads(dV_h)
```

Then each of $Q, K, V$ came from an ordinary linear layer on the same input $\tilde{h}$:

$$Q = \tilde{h}W_q + b_q, \qquad K = \tilde{h}W_k + b_k, \qquad V = \tilde{h}W_v + b_v$$

Atom 1, three times:

$$dW_q = \tilde{h}^\top dQ, \qquad dW_k = \tilde{h}^\top dK, \qquad dW_v = \tilde{h}^\top dV$$
$$db_q = \textstyle\sum_{b,s}dQ, \qquad db_k = \textstyle\sum_{b,s}dK, \qquad db_v = \textstyle\sum_{b,s}dV$$

### And now the Fan-Out Law, one more time

$\tilde{h}$ was used **three times** — once for each projection. Therefore:

$$\boxed{\;d\tilde{h} = dQ\,W_q^\top + dK\,W_k^\top + dV\,W_v^\top\;}$$

**Three terms, added.** Not one. This is exactly the §0.4 fan-out situation, and it is the second most common place people drop a term.

```python
dxn = np.zeros_like(x_norm)
for name, dmat in (("q", dQ), ("k", dK), ("v", dV)):
    dx_, dW_, db_ = linear_backward(dmat, x_norm, params[f"W{name}"])
    grads[f"W{name}"] += dW_
    grads[f"b{name}"] += db_
    dxn += dx_                       # <-- the Fan-Out Law
```

> **Note on fused QKV.** Real GPT-2 stores $W_q, W_k, W_v$ as one $(d, 3d)$ matrix `c_attn` and splits the output. That is a pure implementation optimisation: one big matmul beats three small ones on a GPU. Mathematically identical — the three-way sum above becomes a single $d\tilde{h} = d[\,Q\|K\|V\,]\,W_{\text{attn}}^\top$, with the concatenation doing the summing for you. Which is a nice illustration that "concatenate then multiply" and "multiply then add" are the same operation seen from two angles.

## 8.9 Back through LN$_1$, and closing the block

Feed $d\tilde{h}$ into LayerNorm backward, then apply the Fan-Out Law at the attention residual junction:

$$dh^{(\ell)} = dh_{\text{from attention}} + d_{\text{res}}$$

The block is complete. This $dh^{(\ell)}$ is what gets handed to block $\ell - 1$.

---
---

# Chapter 9 — The embeddings

We have arrived back at $dh^{(0)}$, the gradient with respect to the very first hidden state. Two things produced it.

## 9.1 Positional embeddings

$$h^{(0)}_{b,s,j} = E_{\text{tok}[b,s],\,j} + (W_{\text{pos}})_{s,j}$$

$W_{\text{pos}}$ is indexed by $s$ only — it is broadcast across the batch. Broadcasting Rule (§1.6): sum over $b$.

$$\boxed{\;d(W_{\text{pos}})_{s,j} = \sum_{b=1}^{B} dh^{(0)}_{b,s,j}\;}$$

```python
grads["wpe"][:S] += dh0.sum(axis=0)      # (S,d)
```

Note the `[:S]`: only the first $S$ positional rows were used, so only they receive gradient. If your batch has sequence length 512 and $S_{\max}$ is 1024, rows 512–1023 get exactly zero gradient this step. That is correct and is why positional embeddings at the far end of the context are trained less if your data is mostly short.

## 9.2 Token embeddings: gradient by scatter-add

The lookup $E[\text{tokens}]$ is a **gather**. Its dual is a **scatter-add**.

Think about what a lookup really is. Selecting row $t$ of $E$ is the same as multiplying $E$ by a one-hot row vector $e_t$. So the "layer" is $h = e_t E$, and by Atom 1, $dE = e_t^\top\,dh$ — a matrix that is zero everywhere except row $t$, which equals $dh$. Summing over all tokens in the batch:

$$\boxed{\;dE_{v,j} = \sum_{b=1}^{B}\sum_{s=1}^{S}\Big[\,\text{tok}[b,s] = v\,\Big]\cdot dh^{(0)}_{b,s,j}\;}$$

In words: **for each token occurrence in the batch, add its hidden-state gradient to that token's embedding row.** If a token appears five times, its row receives five additions — the Fan-Out Law again.

```python
np.add.at(grads["wte"], tokens, dh0)     # scatter-add, NOT assignment
```

**Two traps here.**

- `grads["wte"][tokens] += dh0` looks equivalent and is **wrong**. NumPy's fancy-index assignment does not accumulate duplicates: if a token appears twice, only one of the two contributions survives. You need `np.add.at` (or `index_add_` / `scatter_add_` in PyTorch). Since common tokens like `the` appear many times per batch, this bug specifically corrupts the embeddings of the *most frequent* tokens.
- Only tokens present in the batch get gradient. With $V = 50257$ and a batch of 8192 tokens, at most 8192 rows are touched. This is why embedding gradients are naturally sparse — and why weight tying (§4.5) is valuable: through the unembedding path, *every* row gets gradient every step.

## 9.3 Closing the tie

Both contributions to $E$ now exist:

$$dE^{\text{total}} = \underbrace{(dZ)^\top H_{\text{out}}}_{\text{Chapter 4}} + \underbrace{\text{scatter-add}(dh^{(0)},\ \text{tokens})}_{\text{this chapter}}$$

The backward pass is complete.

---
---

# Chapter 10 — The whole backward pass, in one place

Here is everything, in execution order. This is the reference; Appendix A is the runnable version.

## 10.1 The complete algorithm

**Head.**

$$dZ = \frac{1}{B\cdot S}\left(\hat{P} - Y\right)$$
$$dH_{\text{out}} = dZ\cdot E \qquad\qquad dE \mathrel{+}= (dZ)^\top H_{\text{out}}$$
$$dh^{(L)},\; d\gamma_f,\; d\beta_f = \text{LN}^{\text{bwd}}\!\left(dH_{\text{out}}\right)$$

**For each block $\ell = L-1$ down to $0$:**

*MLP sub-block:*

$$d_{\text{res}} = dh, \qquad df_2 = dh$$
$$df_{\text{act}} = df_2 W_2^\top, \qquad dW_2 = f_{\text{act}}^\top df_2, \qquad db_2 = \textstyle\sum_{b,s} df_2$$
$$df_1 = df_{\text{act}} \odot \text{GELU}'(f_1)$$
$$d\tilde{h}_2 = df_1 W_1^\top, \qquad dW_1 = \tilde{h}_2^\top df_1, \qquad db_1 = \textstyle\sum_{b,s} df_1$$
$$dh_{\text{mid}},\, d\gamma_2,\, d\beta_2 = \text{LN}_2^{\text{bwd}}(d\tilde{h}_2); \qquad dh_{\text{mid}} \mathrel{+}= d_{\text{res}}$$

*Attention sub-block:*

$$d_{\text{res}} = dh_{\text{mid}}, \qquad d\text{MHSA}_{\text{out}} = dh_{\text{mid}}$$
$$d\text{concat} = d\text{MHSA}_{\text{out}}W_o^\top, \quad dW_o = \text{concat}^\top d\text{MHSA}_{\text{out}}, \quad db_o = \textstyle\sum_{b,s} d\text{MHSA}_{\text{out}}$$
$$d\text{head} = \text{split\_heads}(d\text{concat})$$
$$dV_h = A^\top\,d\text{head}, \qquad dA = d\text{head}\,V_h^\top$$
$$d\text{scores} = A\odot\left(dA - \texttt{sum}_{-1}(dA\odot A)\right)$$
$$dP = \frac{1}{\sqrt{d_k}}d\text{scores}$$
$$dQ_h = dP\,K_h, \qquad dK_h = dP^\top Q_h$$
$$dQ, dK, dV = \text{merge\_heads}(dQ_h, dK_h, dV_h)$$
$$dW_{q/k/v} = \tilde{h}^\top dQ/dK/dV, \qquad db_{q/k/v} = \textstyle\sum_{b,s}dQ/dK/dV$$
$$d\tilde{h} = dQ\,W_q^\top + dK\,W_k^\top + dV\,W_v^\top$$
$$dh,\, d\gamma_1,\, d\beta_1 = \text{LN}_1^{\text{bwd}}(d\tilde{h}); \qquad dh \mathrel{+}= d_{\text{res}}$$

**Embeddings.**

$$dW_{\text{pos}}[0{:}S] \mathrel{+}= \textstyle\sum_b dh, \qquad dE \mathrel{+}= \text{scatter-add}(dh,\ \text{tokens})$$

## 10.2 Every place a `+=` is mandatory

There are exactly five, and each one is the Fan-Out Law:

1. $dh_{\text{mid}} \mathrel{+}= d_{\text{res}}$ — MLP residual junction
2. $dh \mathrel{+}= d_{\text{res}}$ — attention residual junction
3. $d\tilde{h} = dQ W_q^\top + dK W_k^\top + dV W_v^\top$ — the three-way QKV fan-out
4. $dE$ receives from **both** the unembedding and the token lookup — weight tying
5. Every accumulation into a parameter's gradient buffer across the batch (implicit in the $\sum_{b,s}$)

## 10.3 Verification

Every formula above was implemented in NumPy (Appendix A) and checked against central finite differences with $\varepsilon = 10^{-4}$, across four configurations:

| configuration | loss | max error (relative to global gradient scale) |
|---|---|---|
| pre-norm, tied embeddings | 2.398382 | $5.0\times10^{-8}$ |
| pre-norm, untied | 2.395605 | $4.2\times10^{-8}$ |
| post-norm, tied | 2.388092 | $1.1\times10^{-7}$ |
| post-norm, untied | 2.389676 | $1.4\times10^{-7}$ |

Errors at the $10^{-8}$ level are the noise floor of finite differencing, not a discrepancy in the mathematics.

And an end-to-end behavioural check — gradient descent on a fixed batch, with these hand-derived gradients and nothing else:

| step | loss |
|---|---|
| 0 | 2.8239 (random guessing on $V=17$ is $\ln 17 = 2.8332$) |
| 50 | 1.1755 |
| 100 | 0.4764 |
| 200 | 0.0945 |
| 300 | 0.0410 |

The gradients are not just numerically correct; they train.

---
---

# Chapter 11 — Why GPT-2 is built this way

Now the payoff. Every architectural choice below is usually presented as a heuristic. Each one is actually a statement about the backward pass, and with Chapters 3–10 in hand we can prove them.

## 11.1 Why divide $QK^\top$ by $\sqrt{d_k}$

**The claim.** Without the $1/\sqrt{d_k}$ scaling, attention logits grow with $\sqrt{d_k}$, the softmax saturates, and — by the structure of the softmax backward pass — the gradient reaching $Q$ and $K$ collapses toward zero. The model cannot learn to attend.

### Part 1: the forward-pass fact (why the logits blow up)

Consider one attention logit before scaling:

$$P_{s,c} = \sum_{m=1}^{d_k} q_m k_m$$

Assume, as is roughly true at initialisation, that the components $q_m$ and $k_m$ are independent with mean 0 and variance 1.

$$\mathbb{E}[P_{s,c}] = \sum_m \mathbb{E}[q_m]\mathbb{E}[k_m] = 0$$

For the variance, independence lets us add term variances:

$$\text{Var}(P_{s,c}) = \sum_{m=1}^{d_k}\text{Var}(q_mk_m) = \sum_{m=1}^{d_k}\mathbb{E}[q_m^2]\mathbb{E}[k_m^2] = \sum_{m=1}^{d_k} 1\cdot 1 = d_k$$

$$\boxed{\;\text{std}(P) = \sqrt{d_k}\;}$$

For GPT-2's $d_k = 64$ that is a standard deviation of **8** — so logits spread over a range of roughly $\pm 24$. What matters for the softmax is not the range but the *gap* between the top logits, and that gap also grows with $\sigma$. Softmax exponentiates gaps, so a distribution that is merely wide becomes a distribution that is sharply peaked. The measurements below quantify exactly how peaked.

Dividing by $\sqrt{d_k}$ makes $\text{std} = 1$ regardless of head size. **That is the entire purpose: make the softmax's input scale independent of $d_k$.**

Measured, with $S = 64$ and unit-variance $q, k$:

| $d_k$ | std(logit) | top-1 minus top-2 logit | max attention prob | entropy (uniform $=\ln 64 = 4.159$) |
|---|---|---|---|---|
| 4 | 1.984 | 0.71 | 0.273 | 2.816 |
| 16 | 3.990 | 1.49 | 0.577 | 1.384 |
| 64 | 8.013 | 3.05 | 0.793 | 0.581 |
| 256 | 15.992 | 6.01 | 0.895 | 0.268 |
| 1024 | 31.968 | 12.18 | 0.948 | 0.127 |
| **any $d_k$, scaled** | **$1.00$** | **$0.38$** | **$0.107$** | **$3.69$** |

The std column tracks $\sqrt{d_k}$ to three digits ($\sqrt{4}=2$, $\sqrt{16}=4$, $\sqrt{64}=8$, $\sqrt{256}=16$, $\sqrt{1024}=32$), exactly as the variance calculation predicts.

Column three is the quantity the softmax actually responds to: the gap between the best and second-best logit in a row. It also doubles with each $4\times$ increase in $d_k$ — because it, too, scales with the standard deviation. At $d_k = 64$ that gap averages $3.05$ nats, meaning the winning key is $e^{3.05} \approx 21$ times more likely than the runner-up, and takes **79% of the attention mass** — *at random initialisation, before any learning has happened*. At $d_k = 1024$ it is 95%.

With the scaling applied, all four columns are pinned regardless of $d_k$.

### Part 2: the backward-pass fact (why saturation kills the gradient)

Now use §8.4. The softmax backward is

$$dZ_{s,c} = A_{s,c}\left(dA_{s,c} - \sum_k dA_{s,k}A_{s,k}\right)$$

**Everything is multiplied by $A_{s,c}$.** Suppose attention has saturated: one entry $A_{s,c^*} \approx 1$ and the other $S-1$ entries $\approx 0$.

- For the $S-1$ near-zero entries: $dZ_{s,c} \approx 0\cdot(\cdots) = 0$. **No gradient.**
- For the winning entry: $\sum_k dA_{s,k}A_{s,k} \approx dA_{s,c^*}\cdot 1$, so the bracket becomes $dA_{s,c^*} - dA_{s,c^*} = 0$. **Also no gradient.**

The whole row is dead. Formally, the softmax Jacobian is $\text{diag}(A) - AA^\top$; as $A \to$ one-hot, this matrix $\to \text{diag}(A) - \text{diag}(A) = 0$. **A saturated softmax has a vanishing Jacobian in every direction.**

And this is a trap the model cannot escape, because the gradient that would tell it to *un*-saturate has itself been zeroed. It is not slow learning; it is no learning.

Measured (mean $|d\text{scores}|$ for a random incoming $dA$):

| $d_k$ | unscaled | scaled by $1/\sqrt{d_k}$ |
|---|---|---|
| 4 | $1.115\times10^{-2}$ | $1.216\times10^{-2}$ |
| 16 | $8.007\times10^{-3}$ | $1.219\times10^{-2}$ |
| 64 | $4.608\times10^{-3}$ | $1.224\times10^{-2}$ |
| 256 | $2.413\times10^{-3}$ | $1.219\times10^{-2}$ |
| 1024 | $1.254\times10^{-3}$ | $1.220\times10^{-2}$ |

Unscaled, the gradient magnitude decays monotonically — by $8.9\times$ from $d_k=4$ to $d_k=1024$. Scaled, it is **flat to three significant figures across a $256\times$ range of head dimensions**. That constancy is the point: the scaling makes the trainability of an attention head independent of its width, which is what lets you scale $d_{\text{model}}$ up without re-tuning anything.

### Part 3: why $\sqrt{d_k}$ and not $d_k$

Because variance adds and standard deviation is what the softmax cares about. Dividing by $d_k$ would give $\text{std} = 1/\sqrt{d_k}$ — logits all within $\pm 0.4$ of each other, softmax nearly uniform, attention unable to discriminate. Both extremes are bad; $\sqrt{d_k}$ is the unique exponent that makes the scale invariant.

**The general principle** — worth carrying to any architecture you design: *whenever you sum $n$ independent terms and feed the result into a saturating nonlinearity, divide by $\sqrt{n}$.* This is the same principle behind Xavier and He initialisation. Attention scaling is that principle applied to a sum whose length happens to be $d_k$.

## 11.2 Why pre-norm beats post-norm

The original 2017 Transformer used **post-norm**:

$$h^{(\ell+1)} = \text{LN}\big(h^{(\ell)} + \mathcal{F}(h^{(\ell)})\big)$$

GPT-2 uses **pre-norm**:

$$h^{(\ell+1)} = h^{(\ell)} + \mathcal{F}\big(\text{LN}(h^{(\ell)})\big)$$

Everything else is identical. Here is why the second one wins, in three linked arguments.

### Argument 1: pre-norm has an exact identity path; post-norm does not

Differentiate each.

**Pre-norm:**
$$\frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}} = I + J_{\mathcal{F}}\,J_{\text{LN}}$$

**Post-norm:**
$$\frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}} = J_{\text{LN}}\big(I + J_{\mathcal{F}}\big)$$

The difference is where $J_{\text{LN}}$ sits. In pre-norm it is **inside** the branch. In post-norm it **wraps everything**.

Compose over $L$ layers:

$$\text{pre:}\quad \frac{\partial h^{(L)}}{\partial h^{(0)}} = \prod_{\ell}\left(I + J_{\mathcal{F}}^{(\ell)}J_{\text{LN}}^{(\ell)}\right) = \underbrace{I}_{\text{exact}} + \sum_{\ell}J_{\mathcal{F}}^{(\ell)}J_{\text{LN}}^{(\ell)} + \cdots$$

$$\text{post:}\quad \frac{\partial h^{(L)}}{\partial h^{(0)}} = \prod_{\ell} J_{\text{LN}}^{(\ell)}\left(I + J_{\mathcal{F}}^{(\ell)}\right) \quad\text{— no identity term anywhere}$$

Pre-norm's expansion contains a bare $I$. **The gradient has a route from the loss to layer 0 along which it is multiplied by nothing.** Post-norm's is a product of $L$ matrices; its magnitude is a product of $L$ factors, so it is exponentially sensitive to whether the typical factor is slightly above or slightly below 1.

### Argument 2: $J_{\text{LN}}$ is rank-deficient, and post-norm applies one per layer

From §5.9, we proved that LayerNorm's Jacobian is

$$J_{\text{LN}} = \frac{1}{\sigma}\left(I - \frac{\mathbf{1}\mathbf{1}^\top}{d} - \frac{\hat{x}\hat{x}^\top}{d}\right)$$

an orthogonal projection killing two directions, scaled by $1/\sigma$ — verified to $4\times10^{-10}$ in §5.9.

- **Post-norm** puts one of these in the path of *every* gradient at *every* layer. The end-to-end Jacobian is a product of $L$ rank-deficient, $1/\sigma$-scaled projections. Each layer's null space is different (it depends on that layer's $\hat{x}$), so they do not simply intersect — instead, each one drains energy from two more directions, and the damage compounds.
- **Pre-norm** puts $J_{\text{LN}}$ inside the branch, where it only affects the *correction* the branch contributes. GPT-2 applies exactly **one** un-bypassed LayerNorm in the entire model: the final $\text{LN}_f$.

**The measurement.** I built the end-to-end Jacobian $\partial h^{\text{final}}/\partial h^{(0)}$ numerically for both variants ($d = 16$, MLP branches, Xavier init, averaged over 6 seeds), giving **both** variants a single final LayerNorm so the comparison is exactly fair:

| $L$ | variant | $\sigma_1$ | $\sigma_{d-2}$ | condition number | # singular values $< 0.01\sigma_1$ |
|---|---|---|---|---|---|
| 1 | pre | 1.95 | $3.1\times10^{-1}$ | 6.3 | 2.0 |
| 1 | post | 1.85 | $3.8\times10^{-1}$ | 4.9 | 2.0 |
| 4 | pre | 2.48 | $1.2\times10^{-1}$ | 20 | 2.0 |
| 4 | post | 2.50 | $8.4\times10^{-2}$ | 30 | 2.3 |
| 16 | pre | 3.80 | $3.3\times10^{-2}$ | **115** | 2.7 |
| 16 | post | 4.41 | $5.3\times10^{-4}$ | **8 290** | 7.2 |
| 32 | pre | 6.88 | $1.2\times10^{-2}$ | **571** | 4.5 |
| 32 | post | 6.39 | $1.2\times10^{-7}$ | **$5.4\times10^{7}$** | 9.8 |
| 48 | pre | 4.89 | $7.1\times10^{-3}$ | **693** | 5.0 |
| 48 | post | 4.74 | $3.9\times10^{-10}$ | **$1.2\times10^{10}$** | 11.8 |

Read the last two columns. At $L=1$ the two are indistinguishable — as they must be, since with one layer the architectures barely differ. As depth grows they diverge violently. At 48 layers, **post-norm's Jacobian is $1.8\times10^{7}$ times worse conditioned**, and has collapsed nearly 12 of its 16 directions to numerical zero. Pre-norm at the same depth has lost 5 and has a condition number of a few hundred.

This is the entire story. Post-norm does not merely shrink gradients; it *destroys directions in gradient space*, and it destroys more of them the deeper you go. Whole components of the gradient simply cease to exist, and no learning-rate tuning recovers information that has been projected away.

### Argument 3: pre-norm is self-damping (the mechanism that makes it *stable*, not just alive)

There is a second, subtler benefit, and it explains why pre-norm is not just "residual networks again".

In pre-norm the residual stream is never renormalised, so variance **accumulates**:

$$\text{Var}\big(h^{(\ell)}\big) \approx \text{Var}\big(h^{(0)}\big) + \sum_{m<\ell}\text{Var}\big(\mathcal{F}^{(m)}\big) \quad\Longrightarrow\quad \sigma_\ell \sim \sqrt{\ell}$$

Measured (pre-norm, Xavier init, $d = 128$):

| $L$ | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| $\text{std}(h^{(L)})$ | 1.19 | 1.36 | 1.63 | 2.08 | 2.80 | 3.87 | 5.31 |
| post-norm, same setting | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Now recall that $J_{\text{LN}}$ carries a factor $1/\sigma_\ell$. So in pre-norm, **the deeper the layer, the larger $\sigma_\ell$, and hence the smaller that layer's branch Jacobian.** Deep branches automatically contribute less. The network is born with a depth-graded damping schedule that nobody had to design.

Post-norm has no such mechanism — every layer sees $\sigma = 1$ and contributes equally — which is exactly why post-norm transformers famously require **learning-rate warmup** to survive their first few thousand steps, and pre-norm ones largely do not.

### The honest counterpoint

Pre-norm is not free. Because the residual stream keeps growing while the branch outputs stay $O(1)$, later layers' contributions become proportionally smaller relative to the stream — sometimes called *representation collapse*. Very deep pre-norm models can behave a little like shallower ones. Post-norm, when you can train it (warmup, careful init, sometimes DeepNet-style scaling), sometimes reaches marginally better final quality. The modern compromise is to keep pre-norm and add a normalisation after the block too — but GPT-2's choice, in 2019, was the one that made 48-layer models trainable at all.

## 11.3 Why residual connections at all

We can now answer this in one line, and it is the same line as Argument 1.

$$\frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}} = I + J_{\mathcal{F}} \qquad\text{versus, without the residual,}\qquad \frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}} = J_{\mathcal{F}}$$

Without residuals, the end-to-end Jacobian is $\prod_\ell J_{\mathcal{F}}^{(\ell)}$. If the typical singular value of $J_{\mathcal{F}}$ is $\lambda$, the gradient at layer 0 scales as $\lambda^L$. For $L = 48$: $\lambda = 0.9$ gives $0.9^{48} \approx 0.006$ (vanishing); $\lambda = 1.1$ gives $1.1^{48} \approx 97$ (exploding). You are balancing on a knife edge, and the edge gets sharper with depth.

With residuals, the Jacobian is $I + J_{\mathcal{F}}$, whose singular values cluster around **1** rather than around $\lambda$. Multiply $L$ of those and you get a bounded, well-conditioned product. **The residual connection converts a multiplicative, exponentially-unstable gradient path into an additive, stable one.**

And it is worth noticing that this is a purely backward-pass argument. Forward, the residual just adds a number. Its value is entirely in what it does to $\partial J/\partial h$.

## 11.4 Why the causal mask leaks no gradient

Proved in §8.5. Recapping the shape of the argument because it is a good template:

1. Forward: $A_{s,c} = 0$ exactly for $c > s$, because $e^{-\infty} = 0$.
2. Backward: the softmax gradient is $dZ_{s,c} = A_{s,c}\left(\cdots\right)$, with $A_{s,c}$ as a **multiplicative prefactor**.
3. Therefore $dZ_{s,c} = 0$ for $c > s$, exactly. $\blacksquare$

The forward masking mechanism *is* the backward masking mechanism. No extra code. (Verified: attention weight and $|d\text{scores}|$ on masked positions both exactly $0.0$.)

**The practical corollary matters more than the proof.** Because the block relies on $A$ being *exactly* zero, your mask constant must be large enough to underflow. Measured, with $S=8$ and unit-variance scores:

| mask constant | max attention weight on a future position | largest gradient magnitude on masked entries |
|---|---|---|
| $-10^9$ | $0.0$ exactly | $0.0$ exactly |
| $-10$ | $6.9\times10^{-5}$ | $2.0\times10^{-4}$ |

$-10^9$ works because $e^{-10^9}$ underflows to $0.0$ in any float format. A "soft mask" of $-10$ leaves a small but genuinely nonzero attention weight, and therefore leaks real gradient backwards from future tokens. Note the leaked gradient ($2\times10^{-4}$) is only about $2400\times$ smaller than the legitimate gradient on allowed entries ($4.7\times10^{-1}$) — small, but nowhere near zero, and it accumulates over every layer, every head, and every step. The symptom is suspiciously good validation loss paired with a model that cannot actually generate, because at inference time the future it learned to peek at is not there.

## 11.5 Why LayerNorm makes training self-stabilising

Two properties, both read directly off §5.9's Jacobian.

**(a) The $1/\sigma$ factor is a free per-token adaptive learning rate.** A token whose activations have grown large gets a large $\sigma$, hence a proportionally *smaller* gradient. Tokens that are behaving normally get normal gradients. The network throttles its own updates on exactly the units that are at risk of blowing up — per token, per layer, at no computational cost, with no hyperparameter.

**(b) The two null directions remove gradients that would do nothing.** $J_{\text{LN}}$ annihilates the $\mathbf{1}$ direction and the $\hat{x}$ direction. Both annihilations are *correct*: LayerNorm's output is exactly invariant to adding a constant to all channels (removed by the mean subtraction) and to scaling all channels (removed by the variance division). Any gradient component pointing along those directions is proposing a change that would have **zero** effect on the output. LayerNorm refuses to pass it back.

That is a real service. Without it, the optimiser would spend part of every update budget wandering along directions the loss cannot see — which is a slow, invisible waste, and one of the reasons unnormalised deep networks are so sensitive to initialisation.

## 11.6 Why GELU and not ReLU

**ReLU's derivative is a step function**: 1 for $x>0$, exactly **0** for $x<0$. Any unit that lands on the negative side receives *zero* gradient. If it stays there, it is dead permanently — no gradient means no update means no escape.

**GELU's derivative is smooth and nonzero almost everywhere**:

$$\text{GELU}'(x) = 0.5(1+\tanh u) + 0.5x(1-\tanh^2 u)\sqrt{\tfrac{2}{\pi}}(1+0.134145x^2)$$

It is small but negative around $x \approx -1$, crosses zero near $x \approx -0.75$, and decays smoothly rather than switching off. **There is no dead-unit failure mode.**

Why does this matter more for a transformer than for a CNN, where ReLU works fine? Because of where the MLP sits: the residual stream is LayerNorm'd immediately before $W_1$, so $f_1$'s distribution is centred near zero. **Roughly half of all MLP hidden units sit in the negative region at any moment.** With ReLU that is half the $4d = 3072$ units contributing exactly nothing to the gradient on that token. With GELU they contribute a small, nonzero, *sign-carrying* amount. Given that GPT-2's MLPs hold two-thirds of the parameters, keeping them all differentiable is not a small matter.

The secondary benefit is smoothness. ReLU's second derivative is a delta function at the origin; GELU's is continuous. Adam's second-moment estimates are much better behaved when the loss surface has no kinks.

## 11.7 Why GPT-2 scales residual-output weights by $1/\sqrt{N}$

The GPT-2 paper specifies: initialise the weights of residual-output layers ($W_o$ and $W_2$) with an extra factor of $1/\sqrt{N}$, where $N$ is the number of residual layers.

**First, pin down what $N$ counts.** Each transformer block contains **two** residual branches — attention and MLP — so a model with $L$ blocks has $N = 2L$ residual layers, not $L$. (GPT-2 small: $L = 12$ blocks, $N = 24$.) Implementations write this as `1/math.sqrt(2 * n_layer)`. Below I use $2L$ throughout to keep the counting visible.

This falls straight out of §11.2's Argument 3. Each block adds two branches to the stream, so with $L$ blocks there are $2L$ additive contributions and

$$\text{Var}\big(h^{(L)}\big) = \text{Var}\big(h^{(0)}\big) + \sum_{i=1}^{2L}\text{Var}(\text{branch}_i) \approx 1 + 2L\cdot\text{Var}(\text{branch})$$

so $\text{std}(h^{(L)}) \sim \sqrt{2L}$ — it grows without bound as you add layers. Scaling each branch's *output* weight by $1/\sqrt{2L}$ divides each branch's variance by $2L$, giving $\text{Var}(h^{(L)}) \approx 1 + 1 = 2$: constant in depth.

Measured (pre-norm MLP stack, $d = 128$):

| $L$ | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| $\text{std}(h^{(L)})$, plain init | 1.19 | 1.36 | 1.63 | 2.08 | 2.80 | 3.87 | 5.31 |
| $\text{std}(h^{(L)})$, with $1/\sqrt{2L}$ | 1.10 | 1.10 | 1.10 | 1.10 | 1.11 | 1.11 | 1.10 |

Flat to three digits across a $64\times$ range of depth.

**Why this is a backward-pass fix, not a forward-pass one.** The forward pass would tolerate a growing stream perfectly well. The problem is that $J_{\text{LN}}$ carries $1/\sigma_\ell$, so an unbounded $\sigma_\ell$ means the gradient into deep branches is scaled by an unbounded, depth-dependent, essentially arbitrary factor. Different layers then train at wildly different effective learning rates. The $1/\sqrt{N}$ init pins $\sigma$ near 1 everywhere, so every layer's branch sees a comparable gradient scale. It is a *conditioning* fix.

## 11.8 Why weight tying, in gradient terms

Covered in §4.5, but the backward-pass reason deserves emphasis. The token embedding for a rare word receives gradient through the lookup path **only when that word appears in the batch** — perhaps once in a hundred thousand steps. Through the tied unembedding path, it receives gradient on **every step**, because every token's embedding participates in every softmax over the vocabulary.

Tying converts embedding-gradient sparsity into density. Without it, rare-token embeddings stay near their random initialisation essentially forever.

## 11.9 Why the loss divides by $B\cdot S$, and why biases sum

Two questions with the same answer: gradient scale should not depend on batch shape.

**The $\frac{1}{B\cdot S}$ in the loss.** From §3.5, $dZ = \frac{1}{BS}(\hat{P}-Y)$. Every downstream gradient inherits that factor. Meanwhile every parameter gradient *sums* over $B\cdot S$ tokens (the shared-parameter rule). The two exactly cancel, so $dW$ has the same expected magnitude whether your batch is 8 sequences or 512. **Without the division, doubling your batch size would double every gradient**, silently doubling your effective learning rate.

**Why $db = \sum_{b,s} dY$ and not a mean.** Because it is not a choice — it is what the Fan-Out Law says. The bias was *copied* to all $B\cdot S$ positions; a copy in the forward pass is a sum in the backward pass. The averaging is already handled by the $\frac{1}{BS}$ inside $dZ$. Writing `.mean()` here instead of `.sum()` divides the bias gradients by $B\cdot S$ a second time — biases then learn thousands of times more slowly than weights, and you will never see an error message.

## 11.10 Summary: the design decisions and their backward-pass justifications

| Design choice | What it fixes in the backward pass |
|---|---|
| $QK^\top/\sqrt{d_k}$ | keeps logit std at 1, so the softmax Jacobian's $A$ prefactor stays away from 0 |
| pre-norm | keeps an exact $I$ in the Jacobian; confines the rank-deficient $J_{\text{LN}}$ to the branch |
| residual connections | turns $\prod J_{\mathcal{F}}$ into $\prod(I + J_{\mathcal{F}})$ — additive, not exponential |
| LayerNorm | $1/\sigma$ throttling per token; removes gradient in the 2 directions with no effect |
| causal mask at $-\infty$ | $A = 0$ exactly, and $A$ is a multiplicative prefactor in the softmax backward |
| GELU | no dead units in the half of the MLP that sits at $x < 0$ |
| $1/\sqrt{N}$ residual init ($N=2L$) | keeps $\sigma_\ell$ depth-independent, so $1/\sigma$ scaling is uniform across layers |
| weight tying | dense gradient for rare-token embeddings |
| cross-entropy + softmax | $1/\hat{P}$ cancels $\hat{P}$; gradient is $\hat{P}-Y$, bounded and never saturating |
| $\frac{1}{BS}$ loss normalisation | cancels the batch-sum in parameter gradients; makes LR batch-size-independent |

Every row is a fact about $\partial J/\partial(\cdot)$. Not one of them is visible from the forward pass alone. That is the argument for learning to do this by hand.

---
---

# Appendix A — Reference implementation

A complete, self-contained NumPy GPT-2 with manual backward. Every formula in this book, verified.

```python
import numpy as np

SQRT_2_OVER_PI, GELU_C = np.sqrt(2.0 / np.pi), 0.044715

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(SQRT_2_OVER_PI * (x + GELU_C * x**3)))

def gelu_grad(x):
    u  = SQRT_2_OVER_PI * (x + GELU_C * x**3)
    t  = np.tanh(u)
    du = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_C * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

# ---------------- LayerNorm (Chapter 5) ----------------
def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu   = x.mean(-1, keepdims=True)
    xc   = x - mu
    var  = (xc**2).mean(-1, keepdims=True)
    std  = np.sqrt(var + eps)
    xhat = xc / std
    return gamma * xhat + beta, (xhat, std, gamma)

def layernorm_backward(dy, cache):
    xhat, std, gamma = cache
    d = xhat.shape[-1]
    dgamma = (dy * xhat).sum(axis=(0, 1))
    dbeta  = dy.sum(axis=(0, 1))
    dxhat  = dy * gamma
    s1 = dxhat.sum(-1, keepdims=True)
    s2 = (dxhat * xhat).sum(-1, keepdims=True)
    dx = (d * dxhat - s1 - xhat * s2) / (d * std)
    return dx, dgamma, dbeta

# ---------------- Linear (Chapter 1, Atom 1) ----------------
def linear_forward(x, W, b):
    return x @ W + b

def linear_backward(dy, x, W):
    lead = tuple(range(dy.ndim - 1))              # all batch-like axes
    dW = np.tensordot(x, dy, axes=(lead, lead))   # dW = X^T dY
    db = dy.sum(axis=lead)                        # db = sum dY
    dx = dy @ W.T                                 # dX = dY W^T
    return dx, dW, db

# ---------------- Head split / merge (Chapter 8.2) ----------------
def split_heads(x, nh):
    B, S, d = x.shape
    return x.reshape(B, S, nh, d // nh).transpose(0, 2, 1, 3)

def merge_heads(x):
    B, nh, S, dk = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, S, nh * dk)

# ---------------- One transformer block, forward ----------------
def block_forward(h, p, nh, mask):
    dk = h.shape[-1] // nh
    c = {"h_in": h}

    xn, c["ln1"] = layernorm_forward(h, p["ln1_g"], p["ln1_b"])
    c["xn"] = xn

    Q = linear_forward(xn, p["Wq"], p["bq"])
    K = linear_forward(xn, p["Wk"], p["bk"])
    V = linear_forward(xn, p["Wv"], p["bv"])
    Qh, Kh, Vh = split_heads(Q, nh), split_heads(K, nh), split_heads(V, nh)
    c["Qh"], c["Kh"], c["Vh"] = Qh, Kh, Vh

    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / np.sqrt(dk)
    A = softmax(np.where(mask, -1e9, scores), axis=-1)
    c["A"] = A

    concat = merge_heads(A @ Vh)
    c["concat"] = concat
    h = h + linear_forward(concat, p["Wo"], p["bo"])        # residual
    c["h_mid"] = h

    hn, c["ln2"] = layernorm_forward(h, p["ln2_g"], p["ln2_b"])
    c["hn"] = hn
    f1 = linear_forward(hn, p["W1"], p["b1"])
    fa = gelu(f1)
    c["f1"], c["fa"] = f1, fa
    h = h + linear_forward(fa, p["W2"], p["b2"])            # residual
    return h, c

# ---------------- One transformer block, backward ----------------
# NOTE on "=" vs "+=": here `g` is a FRESH per-block dict, so plain assignment
# is correct -- each block owns its own W1/W2/Wq/... . The five mandatory `+=`
# sites of section 10.2 are the ones marked FAN-OUT LAW below, plus the two
# contributions to the tied embedding. If instead you keep one global gradient
# dict across blocks (or accumulate over microbatches), every line below must
# become `+=`.
def block_backward(dh, c, p, nh, g):
    dk = dh.shape[-1] // nh

    # --- MLP sub-block (Chapter 7) ---
    d_res, df2 = dh, dh                                     # residual copies
    dfa, g["W2"], g["b2"] = linear_backward(df2, c["fa"], p["W2"])
    df1 = dfa * gelu_grad(c["f1"])
    dhn, g["W1"], g["b1"] = linear_backward(df1, c["hn"], p["W1"])
    dh_mid, g["ln2_g"], g["ln2_b"] = layernorm_backward(dhn, c["ln2"])
    dh_mid = dh_mid + d_res                                 # FAN-OUT LAW

    # --- attention sub-block (Chapter 8) ---
    d_res2 = dh_mid
    dconcat, g["Wo"], g["bo"] = linear_backward(dh_mid, c["concat"], p["Wo"])
    dhead = split_heads(dconcat, nh)

    A, Vh, Qh, Kh = c["A"], c["Vh"], c["Qh"], c["Kh"]
    dVh = A.transpose(0, 1, 3, 2) @ dhead                   # dV = A^T dhead
    dA  = dhead @ Vh.transpose(0, 1, 3, 2)                  # dA = dhead V^T

    dscores = A * (dA - (dA * A).sum(-1, keepdims=True))    # softmax bwd
    dP = dscores / np.sqrt(dk)                              # mask is additive:
                                                            # nothing to do for it
    dQh = dP @ Kh
    dKh = dP.transpose(0, 1, 3, 2) @ Qh

    dxn = np.zeros_like(c["xn"])
    for nm, dmat in (("q", merge_heads(dQh)),
                     ("k", merge_heads(dKh)),
                     ("v", merge_heads(dVh))):
        dx_, g[f"W{nm}"], g[f"b{nm}"] = linear_backward(dmat, c["xn"], p[f"W{nm}"])
        dxn += dx_                                          # FAN-OUT LAW (x3)

    dh, g["ln1_g"], g["ln1_b"] = layernorm_backward(dxn, c["ln1"])
    return dh + d_res2                                      # FAN-OUT LAW
```

**Head and embedding, forward and backward:**

```python
# forward head
H_out, lnf_cache = layernorm_forward(h, p["lnf_g"], p["lnf_b"])
logits = H_out @ E.T                                        # E is (V,d), tied
probs  = softmax(logits, axis=-1)
loss   = -np.log(np.take_along_axis(probs, targets[...,None], -1)).mean()

# backward head (Chapters 3-4)
Y = np.zeros_like(probs); np.put_along_axis(Y, targets[...,None], 1.0, -1)
dlogits = (probs - Y) / (B * S)                             # Chapter 3
dH_out  = dlogits @ E                                       # Chapter 4
g["wte"] += np.tensordot(dlogits, H_out, axes=((0,1),(0,1)))  # tied: contribution 1
dh, g["lnf_g"], g["lnf_b"] = layernorm_backward(dH_out, lnf_cache)

# ... all blocks in reverse ...

# backward embeddings (Chapter 9)
g["wpe"][:S] += dh.sum(axis=0)                              # broadcast -> sum
np.add.at(g["wte"], tokens, dh)                             # tied: contribution 2
```

**The gradient checker — write this first:**

```python
def grad_check(model, tokens, targets, eps=1e-4):
    loss, cache = model.forward(tokens, targets)
    g = model.backward(cache)
    scale = max(np.abs(v).max() for v in g.values())
    worst = 0.0
    for name, arr in model.p.items():
        flat, gflat = arr.reshape(-1), g[name].reshape(-1)
        for i in np.random.choice(flat.size, min(10, flat.size), replace=False):
            o = flat[i]
            flat[i] = o + eps; lp, _ = model.forward(tokens, targets)
            flat[i] = o - eps; lm, _ = model.forward(tokens, targets)
            flat[i] = o
            worst = max(worst, abs((lp - lm) / (2*eps) - gflat[i]) / scale)
    return worst          # want < 1e-6
```

> **Why compare errors against the *global* gradient scale rather than per-parameter.** Some gradients are genuinely tiny — at initialisation, $dW_q$ entries are around $10^{-6}$ while $dW_2$ entries are around $10^{-1}$. A per-element *relative* error on a $10^{-6}$ gradient is dominated by finite-difference noise (the noise floor at $\varepsilon=10^{-4}$ is about $10^{-12}$ absolute), and you will chase phantom bugs for hours. Scale by the largest gradient in the model and the picture is honest. This is not a loosening of the test — the absolute errors are $10^{-12}$, which is as exact as double-precision finite differencing gets.

---

# Appendix B — Every backward formula, one page

**Head**

| Forward | Backward |
|---|---|
| $J = -\frac{1}{BS}\sum Y\ln\hat{P}$, $\hat{P} = \text{softmax}(Z)$ | $dZ = \frac{1}{BS}(\hat{P}-Y)$ |
| $Z = H_{\text{out}}E^\top$ | $dH_{\text{out}} = dZ\,E$; $\;dE \mathrel{+}= dZ^\top H_{\text{out}}$ |
| $H_{\text{out}} = \text{LN}_f(h^{(L)})$ | LayerNorm backward |

**LayerNorm** ($s = \sqrt{\sigma^2+\epsilon}$)

| Forward | Backward |
|---|---|
| $\hat{x} = \frac{x-\mu}{s}$, $\;y = \gamma\hat{x}+\beta$ | $d\gamma = \sum_{b,s} dy\odot\hat{x}$; $\;d\beta = \sum_{b,s}dy$; $\;d\hat{x} = dy\odot\gamma$ |
| | $dx = \frac{1}{d\,s}\left[d\cdot d\hat{x} - \texttt{sum}_{-1}(d\hat{x}) - \hat{x}\odot\texttt{sum}_{-1}(d\hat{x}\odot\hat{x})\right]$ |

**Residual**

| Forward | Backward |
|---|---|
| $h_{\text{out}} = h_{\text{in}} + \mathcal{F}$ | $d\mathcal{F} = dh_{\text{out}}$; $\;dh_{\text{in}} = dh_{\text{out}} + d\mathcal{F}_{\to h_{\text{in}}}$ |

**MLP**

| Forward | Backward |
|---|---|
| $f_2 = f_{\text{act}}W_2 + b_2$ | $dW_2 = f_{\text{act}}^\top df_2$; $\;db_2 = \sum_{b,s}df_2$; $\;df_{\text{act}} = df_2W_2^\top$ |
| $f_{\text{act}} = \text{GELU}(f_1)$ | $df_1 = df_{\text{act}}\odot\text{GELU}'(f_1)$ |
| $f_1 = \tilde{h}_2W_1 + b_1$ | $dW_1 = \tilde{h}_2^\top df_1$; $\;db_1 = \sum_{b,s}df_1$; $\;d\tilde{h}_2 = df_1W_1^\top$ |

**Attention**

| Forward | Backward |
|---|---|
| $\text{MHSA} = \text{concat}\,W_o + b_o$ | $dW_o = \text{concat}^\top d\text{MHSA}$; $\;db_o = \sum_{b,s}d\text{MHSA}$; $\;d\text{concat} = d\text{MHSA}\,W_o^\top$ |
| $\text{concat} = \text{merge}(\text{head})$ | $d\text{head} = \text{split}(d\text{concat})$ |
| $\text{head} = A\,V_h$ | $dV_h = A^\top d\text{head}$; $\;dA = d\text{head}\;V_h^\top$ |
| $A = \text{softmax}(Z)$ | $dZ = A\odot\left(dA - \texttt{sum}_{-1}(dA\odot A)\right)$ |
| $Z = \text{scores} + M$ | $d\text{scores} = dZ$ (and $=0$ on masked entries, automatically) |
| $\text{scores} = P/\sqrt{d_k}$ | $dP = d\text{scores}/\sqrt{d_k}$ |
| $P = Q_hK_h^\top$ | $dQ_h = dP\,K_h$; $\;dK_h = dP^\top Q_h$ |
| $Q/K/V = \tilde{h}W_{q/k/v} + b_{q/k/v}$ | $dW_\bullet = \tilde{h}^\top d\bullet$; $\;db_\bullet = \sum_{b,s}d\bullet$; $\;d\tilde{h} = dQ W_q^\top + dK W_k^\top + dV W_v^\top$ |

**Embeddings**

| Forward | Backward |
|---|---|
| $h^{(0)} = E[\text{tok}] + W_{\text{pos}}[{:}S]$ | $dW_{\text{pos}}[{:}S] \mathrel{+}= \sum_b dh^{(0)}$; $\;dE \mathrel{+}= \texttt{scatter-add}(dh^{(0)}, \text{tok})$ |

---

# Appendix C — The bugs this derivation saves you from

Ranked by how long they take to find.

1. **`=` instead of `+=` at a residual junction.** Model trains, loss plateaus higher, no error. You have effectively removed the skip connection. *Symptom:* deeper models perform no better than shallow ones.

2. **Missing one of the three QKV terms in $d\tilde{h}$.** Trains fine, converges worse. *Symptom:* one of Q/K/V learns visibly slower than the others. *Test:* check $d\tilde{h}$ against `torch.autograd.grad` on a two-token input.

3. **`grads[wte][tokens] += dh` instead of `np.add.at`.** Silently drops duplicate-token contributions. *Symptom:* the *most frequent* tokens have the worst embeddings — the exact opposite of what you would expect, which is what makes it so confusing.

4. **Forgetting the second contribution to a tied $dE$.** Trains, slightly worse. *Test:* untie the weights and confirm the loss curve changes in the direction you expect.

5. **Transposing $dW$ for a square weight matrix.** No shape error for any of $W_q, W_k, W_v, W_o$ (all $(d,d)$). Gradient is the transpose of the truth. *Symptom:* loss decreases, then diverges. *Only* the gradient checker catches this.

6. **Evaluating $\phi'$ at the post-activation instead of the pre-activation.** Correct for tanh by coincidence (§0.7), wrong for GELU.

7. **`.mean()` instead of `.sum()` for bias gradients.** Biases learn $B\cdot S \approx 8000\times$ too slowly. Essentially invisible.

8. **Softmax backward without the $\odot A$ prefactor** (i.e. using $dA - \text{sum}(dA\odot A)$ alone). Rows no longer sum to zero. *Test:* assert $|\sum_c dZ_{b,s,c}| < 10^{-12}$.

9. **A causal mask constant that is too small** (e.g. $-10$). Leaks gradient from the future. *Symptom:* validation loss suspiciously good, generation incoherent. *Test:* assert `A[..., mask].max() == 0.0` exactly.

10. **Caching $\mu, \sigma^2$ but not $\hat{x}$**, then reconstructing $\hat{x}$ with a different $\epsilon$ than the forward pass used. Small, persistent gradient error. *Symptom:* gradient check fails at $10^{-4}$ but passes at $10^{-2}$, and you convince yourself it is "just numerical".

**The meta-lesson:** none of these throws an exception, and most of them still train. A hand-written backward pass is only as trustworthy as its gradient checker. Write the checker first.

---

## Closing

You now have the complete backward pass for GPT-2, derived from four rules:

- the chain rule (§0.3),
- the multivariable chain rule, which makes fan-out into summation (§0.4),
- the shared-parameter rule: weights sum over batch and sequence (§1.3),
- the shape constraint, which tells you how to write the sum as a matmul (§1.4).

Everything else — softmax, LayerNorm, attention, GELU — is those four rules applied patiently to one more operation.

And the reason to have done it by hand is Chapter 11. The $\sqrt{d_k}$, the pre-norm, the residual stream, the mask value, the $1/\sqrt{N}$ init: none of these are visible from the forward pass. They are all facts about what happens to $\partial J/\partial(\cdot)$ on the way back. An architecture is a forward pass, but a *trainable* architecture is a backward pass. That is the thing worth being able to see.
