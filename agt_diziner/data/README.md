# Data

`raw/*.conll` → preprocessing → `<preset>/train.json` and `<preset>/test.json`.

Preprocessing preserves tokens and BIO labels, removes duplicate training examples and train/test overlaps, filters language, and samples the configured subset. `dataset_audit.json` records raw file hashes, cleaning counts, and export sizes. Dev remains unused.

The pilot exports up to 200 training-pool examples and 40 test examples. Training needs at least 40 eligible examples. Use `--config` and `--output` for another preset; see the [main README](../README.md).

## JSON format

Each split is an array of records:

```json
[
  {
    "id": "train:0",
    "tokens": ["John", "works", "."],
    "labels": ["B-PER", "O", "O"],
    "lang": "en",
    "source": "example"
  }
]
```

IDs retain raw example positions before cleaning. Labels align with tokens and use [schema.json](../schema.json). Training excludes gold labels from model input; testing uses them for scoring.

[Raw CoNLL format](raw/README.md) · [Train and test commands](../README.md#run)
