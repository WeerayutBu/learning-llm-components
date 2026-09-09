# Data

`raw/*.conll` → preprocessing → `<preset>/train.json` and `<preset>/test.json`.

Preprocessing preserves tokens and BIO labels, removes duplicate training examples and train/test overlaps, filters language, and samples the configured subset. `metadata.json` records raw file hashes, cleaning counts, and export sizes. Dev remains unused.

The pilot exports up to 200 training-pool examples and 40 test examples. Training needs at least 40 eligible examples. Use `--config` and `--output` for another preset; see the [main README](../README.md).

Each dataset owns its schema and provenance:

```text
raw/       train.conll  test.conll  dev.conll  schema.json  metadata.json
pilot/     train.json   test.json             schema.json  metadata.json
pilot-en/  train.json   test.json             schema.json  metadata.json
pilot-th/  train.json   test.json             schema.json  metadata.json
full/      train.json   test.json             schema.json  metadata.json
```

Preprocessing reads `raw/schema.json` and copies it into the output dataset. Training and testing load `schema.json` beside the supplied split; paired splits must have matching schemas. `metadata.json` keeps source provenance and preparation settings within the dataset. Saved prompt configurations retain their schema for prediction.

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

IDs retain raw example positions before cleaning. Labels align with tokens and use the dataset’s own `schema.json`. Training excludes gold labels from model input; testing uses them for scoring.

[Raw CoNLL format](raw/README.md) · [Train and test commands](../README.md#run)

## Git storage

Raw CoNLL files, processed split JSON files, and generated runs stay local and are ignored by Git. Commit schemas, metadata, documentation, and the small `tokens.json` example. A fresh checkout needs the raw data from the dataset and revision recorded in `raw/metadata.json` before preprocessing.
