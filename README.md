# GenTUS

GenTUS is a generative table-union search method based on Semantic IDs and
constrained decoding.

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

## Structure

```text
src/train.py          # train GenTUS and build Semantic IDs
src/evaluate.py       # evaluate on the held-out query split
src/protocol.py       # query split and ground-truth preprocessing
src/semantic_id/      # RQ-VAE and Semantic-ID management
src/model/            # seq2seq model and constrained decoder
src/data/             # dataset loading and table serialization
```
