# GenTUS: Table Union Search via Constrained Generation of Semantic Table Identifiers

[![arXiv](https://img.shields.io/badge/arXiv-2607.00833-b31b1b.svg)](https://arxiv.org/abs/2607.00833)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/pdf/2607.00833)

Implementation of **GenTUS: Table Union Search via Constrained Generation of Semantic Table Identifiers**.

Shulun Zhang, Linting Wang, Yuwei Xu, Yingli Zhou, Jieming Shi, and Chenhao Ma

The linked arXiv paper is the earlier preprint, *Generative Retrieval for Table Union Search*; its title, author list, and results differ from the current submission.

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

Experiments on seven public TUS benchmarks evaluate retrieval quality, online
latency, retrieval-artifact construction time, and storage.

**[Read the earlier preprint](https://arxiv.org/abs/2607.00833)** ·
**[Download its PDF](https://arxiv.org/pdf/2607.00833)**

## Installation

Use Python 3.11 or later.

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

The table encoder uses BERT column representations, learned attention pooling,
and a projection to normalized 256-dimensional vectors. Place the trained
dataset-specific encoder checkpoint at the path in `configs/<dataset>.json`,
or supply `--encoder-checkpoint`. Checkpoints contain `model` (the full state
dictionary) and `config`; the loader requires all encoder weights. Checkpoint
files are not bundled in this repository.

The seven configurations below use retained experiment parameters and fixed
query lists in `configs/splits/`. Each configuration records the ground-truth
and encoder SHA256, seed, and source records. The runner verifies the input
hashes before executing.

| Dataset | k | Quantizers × codebook size | Generator epochs | Seed |
|---|---:|---:|---:|---:|
| TUS_small | 60 | 4 × 256 | 30 | 42 |
| TUS_large | 60 | 6 × 512 | 30 | 42 |
| SANTOS_small | 10 | 4 × 256 | 80 | 42 |
| SANTOS_large | 20 | 8 × 1024 | 80 | 42 |
| wiki_union | 40 | 6 × 512 | 30 | 42 |
| LakeBench_opendata_1k | 60 | 6 × 512 | 30 | 42 |
| LakeBench_opendata_30k | 60 | 6 × 512 | 30 | 42 |

Source metrics in these files identify retained individual runs, not a new
reproduction of the paper's aggregated results. SANTOS_small training parameters
come from the retained experiment launcher; its individual training summary
was not retained. SANTOS_large uses the retained seven-query evaluation split.

Train and evaluate:

```bash
DATASET=TUS_small ./scripts/run_gentus.sh
```

Run one stage, or inspect the commands without starting training:

```bash
python scripts/run_gentus.py --dataset SANTOS_large --stage train
python scripts/run_gentus.py --dataset SANTOS_large --stage evaluate
python scripts/run_gentus.py --dataset SANTOS_large --dry-run
```

For local model caches:

```bash
python scripts/run_gentus.py --dataset TUS_small \
  --encoder-checkpoint /path/to/tus_small_model.pt \
  --model /path/to/t5-base --embed-model /path/to/bert-base-uncased \
  --local-files-only
```

Evaluation saves aggregate metrics, per-query metrics, and rankings under
`results/<dataset>/seed42/`. AP@k is normalized by `min(k, |G(q)|)`; precision
uses k and recall uses the ground-truth size. Self matches and repeated hits
are excluded. The retained LakeBench configurations restrict ground truth to
the indexed candidate pool.

## Tests

```bash
pytest -q
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
