# direct-network

Baseline sequence model: peptide + HLA-pseudoseq + allele as integer indices →
shared amino-acid embedding → flatten → concat allele embedding → MLP head
(`model_common.mlp_head`). The simplest of the four; the others are measured
against it.

Unique files: `model.py` (`DirectAffinityNet`), `config.py` (`DEFAULTS` +
`train_from_config`), `train.py` (`train_model` → `train_common.run_training`),
`compare.py`. Everything else (data encoding, training loop, plots, metrics) is
shared from `mlp-embeds/`.

```bash
# all four splits -> figs/all75_split_*/   (or: make make_model direct-network  from mlp-embeds/)
for t in templates/*.yaml; do ../../../.venv/bin/python train.py $t; done
```

See [`../README.md`](../README.md) for the splits, figures, and shared layout.
