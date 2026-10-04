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
# train all four splits -> figs/   (or: make direct  from mlp-embeds/)
../../../.venv/bin/python compare.py templates/*.yaml
```

See [`../README.md`](../README.md) for the splits, figures, and shared layout.
