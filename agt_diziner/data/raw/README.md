# Raw CoNLL data

Original splits from `weerayut/thai-english-financial-ner`:

| File | Examples | Use |
|---|---:|---|
| `train.conll` | 45,488 | Prompt refinement |
| `test.conll` | 12,985 | Evaluation |
| `dev.conll` | 6,529 | Original validation split; currently unused |

[metadata.json](metadata.json) records the source revision and file hashes.

## Included demo

[demo/](demo/metadata.json) contains 40 training and 40 test examples from the bilingual pilot, with its own schema and provenance. These small CoNLL files are included in Git; the full raw splits remain ignored.

In the main workflow, preprocess with `--data data/raw/demo --output data/demo`, then use `data/demo/train.json`, `data/demo/test.json`, and `--output runs/demo`. The default pilot config fits this demo. Preprocessing assigns IDs by demo row position, so use a separate run from the full pilot.

## Format

One token and BIO label per row, separated by a tab. Blank lines separate examples. Metadata supplies language and source:

```text
# {"lang": "en", "source": "example"}
John	B-PER
works	O
at	O
Google	B-ORG_COM
.	O

```

Space-separated columns also work. For multi-column files, the first column is the token and the last is the label. `-DOCSTART-` marks a boundary. Tokens and labels are preserved literally; use labels from [schema.json](schema.json).

The supplied files already include language metadata. For custom files without it, pass `--lang en` or `--lang th` to preprocessing. Mixed-language files need metadata before each example; `# lang=en` and `# lang=th` also work.

[Preprocess, train, and test](../../README.md#run).
