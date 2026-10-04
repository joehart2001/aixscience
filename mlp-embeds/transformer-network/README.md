# transformer-network

Same task and same inputs as `direct-network` (predict peptide–MHC half-life
`thalf_hours` from `peptide` + HLA sequence + `allele`), but the sequences are
encoded by a **transformer encoder** instead of being flattened into an MLP.
A brute-force bet that self-attention learns a richer representation of the
peptide–groove pair than fixed per-position weights can.

Only `model.py` (the encoder) and `config.py` are unique; the data encoding,
training loop, metrics, and plots are shared from `mlp-embeds/`. The inputs are
integer amino-acid indices, exactly what the transformer consumes. See
[`../README.md`](../README.md) for the shared layout, splits, and how to run.

## What's different from direct-network

| | direct-network | transformer-network |
|---|---|---|
| Encoder | flatten per-residue embeddings → concat | [CLS] + peptide + HLA as one token stream → self-attention |
| Residue interactions | one fixed weight per (position, position) pair | content-dependent attention, per example |
| Position information | implicit in the flattened layout | learned positional + segment embeddings |
| Head | 2×256 ReLU MLP | 2×256 ReLU MLP (**unchanged**, deliberately) |
| Params | ~850K | ~242K at the default `d_model: 64` |
| `data.py` | — | identical |

The head is kept identical on purpose: any difference in results is
attributable to the encoder, not to head capacity. Note the transformer is the
*smaller* model — the flattened MLP spends most of its parameters on the first
Linear layer.

### Token stream

```
[CLS] p1 p2 ... p9   h1 h2 ... h34
  |   \_________/    \__________/
  |    segment 1       segment 2
  pooled -> concat with allele embedding -> MLP head
```

Three embeddings are summed per token: the **shared amino-acid embedding** (a
leucine is the same vector in the peptide and in the groove), a **learned
positional embedding** (attention is order-blind without it, and anchor
position is most of what determines binding), and a **segment embedding**
(which sequence the residue came from). Padded positions are masked out of
attention. The allele embedding is concatenated onto the pooled `[CLS]` vector
rather than inserted as a token, mirroring how direct-network feeds it.

Blocks are **pre-LN** (`norm_first=True`), which trains stably on a dataset
this small without the warmup schedule post-LN would need.

## Run

```bash
# all four splits -> figs/all75_split_*/   (or: make make_model transformer-network  from mlp-embeds/)
for t in templates/*.yaml; do ../../../.venv/bin/python train.py $t; done
```

Templates are `random` / `fix-peptide` / `fix-allele` / `fix-allele-group`.
See [`../README.md`](../README.md) for the figures produced and the shared layout.

## Config

Identical to direct-network's schema plus one extra block — `model:`, whose
keys are passed straight to `DirectAffinityNet`. Omit any key to use the
model's own default:

```yaml
model:
  d_model: 64            # width of the residual stream
  nhead: 4               # attention heads per layer
  num_layers: 3          # encoder blocks
  dim_feedforward: 256   # width of each block's feedforward
  allele_embed_dim: 16   # allele vector, concatenated at the head
  hidden_dim: 256        # head width (matches direct-network's MLP)
  dropout: 0.1
```

`hla_col: hla_pseudoseq` (the 34-aa contact-residue pseudosequence) is the
default in the templates — it generalizes better than the full 182-aa sequence,
and it also keeps the token stream at 44 positions instead of 192, which matters
for attention's quadratic cost.
