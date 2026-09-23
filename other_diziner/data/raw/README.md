# Raw CoNLL data

Each dataset has its own directory containing `train.conll`, `test.conll`, `dev.conll`, `schema.json`, and `metadata.json`.

```text
raw/
├── finer/       # Thai/English financial NER
├── conll2003/   # English general NER
└── demo/        # Small CoNLL-2003 pilot example (train/test only)
```

## Financial NER (`finer`)

Original splits from `weerayut/thai-english-financial-ner`, stored in `finer/`. This folder name refers to the existing bilingual dataset, not the separate FiNER-139 corpus:

| File | Examples | Use |
|---|---:|---|
| `train.conll` | 45,488 | Prompt refinement |
| `test.conll` | 12,985 | Evaluation |
| `dev.conll` | 6,529 | Original validation split; currently unused |

[finer/metadata.json](finer/metadata.json) records the source revision and file hashes.

## Included demo

[demo/](demo/metadata.json) contains 200 training-pool and 40 test sentences sampled from CoNLL-2003 with seed 42 (test seed 43), after removing duplicate training texts and train/test overlaps, with its own schema and provenance. These small CoNLL files are included in Git; the full raw splits remain ignored.

In the main workflow, preprocess with `--data data/raw/demo --output data/demo`, then use `data/demo/train.json`, `data/demo/test.json`, and a fresh `--output runs/demo-conll2003`, using `--config config.pilot.en.json` for preprocessing and training/testing. Training selects 40 sentences across five iterations from the 200-sentence pool. Compare baseline and updated prompts on the same held-out test sentences; this small sample is exploratory and does not guarantee an improvement or establish a benchmark trend. Preprocessing assigns IDs by demo row position.

## CoNLL-2003

[conll2003/](conll2003/metadata.json) contains the English splits from the [Hugging Face mirror](https://huggingface.co/datasets/tomaarsen/conll2003/tree/3a0c0677d3177b6122fceed3239dda7a5de6a514): 14,041 train, 3,250 dev, and 3,453 test sentences. Its schema uses `PER`, `ORG`, `LOC`, and `MISC`. Tokens and IOB2 NER labels are preserved; POS/chunk columns are omitted. Dev is retained but unused by the pipeline. Full CoNLL files remain local and ignored by Git.

From `other_diziner/`, prepare the English pilot subset:

```bash
uv run python preprocess.py --config config.pilot.en.json --data data/raw/conll2003 --output data/conll2003
```

This samples up to 200 training and 40 test sentences after cleaning. Follow the [main run commands](../../README.md#run) with `--config config.pilot.en.json`, `data/conll2003/train.json`, `data/conll2003/test.json`, and a fresh `--output runs/conll2003`. The dataset schema is loaded automatically.

## Format

One token and BIO label per row, separated by a tab. Blank lines separate examples. Metadata supplies language and source:

```text
# {"lang": "en", "source": "example"}
John	B-PER
works	O
at	O
Google	B-ORG
.	O

```

Space-separated columns also work. For multi-column files, the first column is the token and the last is the label. `-DOCSTART-` marks a boundary. Tokens and labels are preserved literally; use labels from the selected dataset’s `schema.json`.

The supplied files already include language metadata. For custom files without it, pass `--lang en` or `--lang th` to preprocessing. Mixed-language files need metadata before each example; `# lang=en` and `# lang=th` also work.

[Preprocess, train, and test](../../README.md#run).
