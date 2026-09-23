"""Read local CoNLL splits without changing tokens or BIO labels."""
import hashlib
import json
from pathlib import Path

from data import bio_spans


def read_conll(path, language=None):
    path = Path(path)
    rows, tokens, labels, metadata = [], [], [], {}

    def flush():
        nonlocal tokens, labels, metadata
        if not tokens:
            return
        lang = metadata.get('lang', language)
        if lang not in ('en', 'th'):
            raise ValueError(f'{path}: document {len(rows)} needs language metadata or --lang en/th')
        rows.append({'tokens': tokens, 'labels': labels, 'lang': lang,
                     'source': metadata.get('source', str(path.resolve()))})
        tokens, labels, metadata = [], [], {}

    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            flush()
            continue
        if line.startswith('# {') or line.startswith('# lang='):
            flush()
            try:
                value = json.loads(line[2:]) if line.startswith('# {') else {'lang': line[len('# lang='):].strip()}
                if not isinstance(value, dict):
                    raise ValueError('metadata must be an object')
                metadata.update(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{number}: invalid document metadata') from exc
            continue
        if line.startswith('# ') and '\t' not in line:
            continue
        columns = line.split('\t') if '\t' in line else line.split()
        if columns and columns[0] == '-DOCSTART-':
            flush()
            continue
        if len(columns) < 2 or not columns[0]:
            raise ValueError(f'{path}:{number}: expected token and BIO label columns')
        token, label = columns[0], columns[-1].strip()
        try:
            bio_spans([label])
        except ValueError as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
        tokens.append(token)
        labels.append(label)
    flush()
    if not rows:
        raise ValueError(f'{path}: no labeled documents found')
    return rows


def load_raw_data(directory, language=None):
    directory = Path(directory)
    paths = {split: directory / f'{split}.conll' for split in ('train', 'test')}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f'Missing raw data: {path}. Add train.conll and test.conll to {directory}.')
    ds = {split: read_conll(path, language) for split, path in paths.items()}
    provenance = {'source_format': 'conll', 'raw_files': {
        split: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for split, path in paths.items()}}
    metadata_path = directory / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        provenance['source_metadata'] = {'path': str(metadata_path.resolve()),
                                         'sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
        provenance.update({key: metadata[key] for key in ('dataset', 'revision') if key in metadata})
    return ds, provenance
