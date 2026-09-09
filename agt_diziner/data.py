"""Preserve the dataset's token coordinates; keep gold out of pilot documents."""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import random


@dataclass(frozen=True)
class Document:
    id: str
    tokens: tuple[str, ...]
    lang: str
    source: str

    def public(self):
        return asdict(self)  # There is deliberately no labels field.


def fingerprint(tokens):
    return hashlib.sha256(json.dumps(list(tokens), ensure_ascii=False).encode()).hexdigest()


def bio_spans(tags):
    """Decode chunk-level BIO, treating an orphan I-X as B-X (IOB2 repair).

    The dataset is chunked at 80 tokens; entities can start before a chunk.
    Coordinates are [start, end), and both boundaries AND type must match.
    """
    spans, start, kind = set(), None, None
    for i, tag in enumerate([*tags, "O"]):
        if tag != "O" and (not isinstance(tag, str) or len(tag) < 3
                            or tag[:2] not in ("B-", "I-")):
            raise ValueError(f"Invalid BIO tag: {tag!r}")
        prefix, new_kind = ("O", None) if tag == "O" else (tag[0], tag[2:])
        if kind is not None and (prefix != "I" or new_kind != kind):
            spans.add((start, i, kind))
            start, kind = None, None
        if prefix != "O" and kind is None:
            start, kind = i, new_kind
    return spans


def spans_bio(spans, n_tokens):
    tags = ["O"] * n_tokens
    for start, end, label in sorted(spans):
        if not 0 <= start < end <= n_tokens or any(t != "O" for t in tags[start:end]):
            raise ValueError("Invalid or overlapping spans")
        tags[start:end] = [f"B-{label}"] + [f"I-{label}"] * (end - start - 1)
    return tags


def prepare(ds, schema):
    """Load gold only for test scoring; use train labels solely to validate schema names.

    Drop exact train/test text overlap and duplicate training texts before sampling.
    Validation data is unused. Never select documents using their entity labels.
    """
    held_out = {fingerprint(r["tokens"]) for r in ds["test"]}
    train, test, gold, seen = [], [], {}, set()
    audit = {"split_sizes": {k: len(v) for k, v in ds.items()},
             "train_duplicates_removed": 0, "train_test_overlap_removed": 0,
             "test_orphan_I_tokens": 0}
    types = set()
    for split in ("train", "test"):
        for i, row in enumerate(ds[split]):
            tokens, labels = row["tokens"], row["labels"]
            if not tokens or len(tokens) != len(labels) or not all(isinstance(t, str) for t in tokens):
                raise ValueError(f"Invalid tokens/labels at {split}:{i}")
            spans = bio_spans(labels)
            types.update(s[2] for s in spans)
            doc = Document(f"{split}:{i}", tuple(tokens), row["lang"], row["source"])
            if split == "test":
                prev = "O"
                for tag in labels:
                    if tag.startswith("I-") and prev not in ("B-" + tag[2:], tag):
                        audit["test_orphan_I_tokens"] += 1
                    prev = tag
                test.append(doc)
                gold[doc.id] = spans
                continue
            key = fingerprint(tokens)
            if key in held_out:
                audit["train_test_overlap_removed"] += 1
            elif key in seen:
                audit["train_duplicates_removed"] += 1
            else:
                train.append(doc)
                seen.add(key)
    if types - set(schema):
        raise ValueError(f"Schema is missing entity types: {sorted(types - set(schema))}")
    audit.update(entity_types=sorted(types), pilot_pool=len(train),
                 train_languages=dict(Counter(d.lang for d in train)),
                 test_languages=dict(Counter(d.lang for d in test)))
    return train, test, gold, audit



def filter_language(train, test, gold, language=None):
    """Filter prepared documents without renumbering IDs or exposing training gold.

    Global de-duplication and held-out overlap removal must happen first.
    None preserves the original bilingual experiment.
    """
    if language not in (None, 'en', 'th'):
        raise ValueError("language must be null, 'en', or 'th'")
    if language is None:
        return train, test, gold
    train = [d for d in train if d.lang == language]
    test = [d for d in test if d.lang == language]
    if not train or not test:
        raise ValueError(f"Language {language!r} requires nonempty training and test pools")
    return train, test, {d.id: gold[d.id] for d in test}

def balanced_sample(docs, count, seed):
    """Round-robin language sampling for a pilot/evaluation subset."""
    if not 0 < count <= len(docs):
        raise ValueError(f"Requested {count} documents from {len(docs)}")
    rng = random.Random(seed)
    groups = {lang: [d for d in docs if d.lang == lang] for lang in sorted({d.lang for d in docs})}
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while len(result) < count:
        for group in groups.values():
            if group and len(result) < count:
                result.append(group.pop())
    return result


def diverse_groups(docs, group_size, iterations, seed, method="tfidf", pool_size=None,
                   embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """K-means representatives without replacement, following upstream grouping.

    Character TF-IDF is the lightweight adaptation. Optional multilingual sentence
    embeddings preserve the upstream encoder+K-means structure for Thai/English.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import pairwise_distances
    pool = balanced_sample(docs, min(pool_size or len(docs), len(docs)), seed)
    if len(pool) < group_size * iterations:
        raise ValueError("Pilot pool is too small for disjoint iteration groups")
    texts = [" ".join(d.tokens) for d in pool]
    if method == "tfidf":
        vectors = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=12000).fit_transform(texts)
    elif method == "sentence-transformer":
        from sentence_transformers import SentenceTransformer
        vectors = SentenceTransformer(embedding_model).encode(texts, normalize_embeddings=True,
                                                             show_progress_bar=True)
    else:
        raise ValueError(f"Unknown grouping method: {method}")
    remaining, groups = list(range(len(pool))), []
    for iteration in range(iterations):
        matrix = vectors[remaining]
        km = KMeans(n_clusters=group_size, random_state=seed + iteration, n_init=10).fit(matrix)
        selected = []
        for cluster in range(group_size):
            members = np.flatnonzero(km.labels_ == cluster)
            if len(members):
                distances = pairwise_distances(matrix[members], km.cluster_centers_[cluster:cluster + 1])[:, 0]
                selected.append(remaining[int(members[int(distances.argmin())])])
        # Identical feature vectors can leave empty clusters; fill deterministically.
        selected += [i for i in remaining if i not in selected][:group_size - len(selected)]
        groups.append([pool[i] for i in selected])
        remaining = [i for i in remaining if i not in selected]
    return groups


def load_local_split(path, schema, *, with_gold=False):
    """Read an exported JSON array without changing token coordinates or IDs."""
    from pathlib import Path
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path}: expected a nonempty JSON array of documents')
    docs, gold, seen = [], {}, set()
    for row in rows:
        if not isinstance(row, dict) or not {'id', 'tokens', 'labels', 'lang', 'source'} <= row.keys():
            raise ValueError(f'{path}: each document needs id, tokens, labels, lang, source')
        identifier, tokens, labels = row['id'], row['tokens'], row['labels']
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f'{path}: duplicate or invalid document ID {identifier!r}')
        if (not isinstance(tokens, list) or not tokens or
                not all(isinstance(t, str) and t for t in tokens) or
                not isinstance(labels, list) or len(tokens) != len(labels) or
                not all(isinstance(t, str) for t in labels)):
            raise ValueError(f'{path}: invalid tokens/labels for {identifier}')
        if row['lang'] not in ('en', 'th') or not isinstance(row['source'], str):
            raise ValueError(f'{path}: invalid language/source for {identifier}')
        spans = bio_spans(labels)
        if {s[2] for s in spans} - set(schema):
            raise ValueError(f'{path}: unknown entity label for {identifier}')
        seen.add(identifier)
        docs.append(Document(identifier, tuple(tokens), row['lang'], row['source']))
        if with_gold:
            gold[identifier] = spans
    return docs, gold


def load_schema(directory):
    """Load the label definitions owned by a dataset directory."""
    from pathlib import Path
    path = Path(directory) / 'schema.json'
    try:
        schema = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read dataset schema at {path}: {exc}') from exc
    if not isinstance(schema, dict) or not schema or not all(
            isinstance(k, str) and k and (
                (isinstance(v, str) and v) or
                (isinstance(v, dict) and isinstance(v.get('definition'), str) and v['definition'])
            ) for k, v in schema.items()):
        raise ValueError(f'{path}: expected a nonempty mapping of labels to descriptions')
    return schema
