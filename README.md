# GenTUS: Generative Retrieval for Table Union Search

[![arXiv](https://img.shields.io/badge/arXiv-2607.00833-b31b1b.svg)](https://arxiv.org/abs/2607.00833)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/pdf/2607.00833)

Official implementation of **[Generative Retrieval for Table Union Search](https://arxiv.org/abs/2607.00833)**.

Shulun Zhang, Linting Wang, Yuwei Xu, Yingli Zhou, and Chenhao Ma

> GenTUS reformulates table union search as constrained generation over
> discrete semantic table identifiers, directly generating unionable tables
> instead of searching and reranking an explicit candidate pool.

## Why GenTUS?

Modern table union search systems typically rely on an
encode-search-refine pipeline, whose effectiveness depends on candidate-pool
recall and whose latency and storage costs grow with the data lake. GenTUS
replaces this pipeline with generative retrieval:

1. **Unionability-aware Semantic IDs** encode candidate tables as compact,
   discrete identifiers.
2. **Direct generation** predicts the identifiers of unionable tables from a
   query table.
3. **Constrained decoding** guarantees that every generated identifier maps
   to a valid table in the data lake.

Across seven public TUS benchmarks, GenTUS achieves the best overall retrieval
quality, with an **average rank of 1.05** versus **2.57** for the strongest
baseline, while substantially reducing online latency, retrieval-artifact
storage, and incremental-update cost.

**[Read the paper](https://arxiv.org/abs/2607.00833)** ·
**[Download the PDF](https://arxiv.org/pdf/2607.00833)**

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Put datasets under `datasets/` using the registered benchmark layout in
`src/data/loader.py`. For `TUS_small`, the expected layout is:

```text
datasets/TUS_small/tus/
  datalake/
    *.csv
  query/
    *.csv
  benchmark.pkl
```

`benchmark.pkl` maps query table names to unionable datalake table names:

```python
{
  "query.csv": ["candidate_a.csv", "candidate_b.csv"]
}
```

Set `GENTUS_DATA_ROOT=/path/to/datasets` if the datasets are stored outside the
repository.

## Run

Train:

```bash
python src/train.py \
  --dataset TUS_small \
  --model t5-base \
  --embed_model sentence-transformers/all-mpnet-base-v2 \
  --output_dir results/TUS_small/seed42 \
  --train_ratio 0.8 \
  --seed 42
```

Evaluate:

```bash
python src/evaluate.py \
  --dataset TUS_small \
  --model_dir results/TUS_small/seed42 \
  --sid_path results/TUS_small/seed42/semantic_ids.json \
  --split_path results/TUS_small/seed42/split.json \
  --k 60 \
  --output results/TUS_small/seed42/eval_summary.json
```

Or run both steps:

```bash
DATASET=TUS_small K=60 ./scripts/run_gentus.sh
```

## Tests

```bash
pytest -q
```

## Citation

If you find GenTUS useful in your research, please cite our paper:

```bibtex
@article{zhang2026gentus,
  title         = {Generative Retrieval for Table Union Search},
  author        = {Zhang, Shulun and Wang, Linting and Xu, Yuwei and Zhou, Yingli and Ma, Chenhao},
  year          = {2026},
  eprint        = {2607.00833},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB}
}
```

## Structure

```text
src/train.py          # train GenTUS and build Semantic IDs
src/evaluate.py       # evaluate on the held-out query split
src/protocol.py       # query split and ground-truth preprocessing
src/semantic_id/      # RQ-VAE and Semantic-ID management
src/model/            # seq2seq model and constrained decoder
src/data/             # dataset loading and table serialization
```
