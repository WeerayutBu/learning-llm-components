import json
from pathlib import Path

import pytest

from data import load_local_split
from preprocess import export_dataset

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / 'data/raw/schema.json').read_text())


def fixture_dataset():
    def row(tokens, labels, lang='en'):
        return dict(tokens=tokens, labels=labels, lang=lang, source='fixture')
    return {
        'train': [row(['shared'], ['O']), row(['บริษัท', '_', 'A\\B', '\t'], ['B-ORG_COM', 'I-ORG_COM', 'O', 'O'], 'th'),
                  row(['บริษัท', '_', 'A\\B', '\t'], ['B-ORG_COM', 'I-ORG_COM', 'O', 'O'], 'th')],
        'test': [row(['shared'], ['I-PER'])], 'validation': []}


def test_export_preserves_tokens_bio_ids_and_removes_leakage(tmp_path):
    config = dict(language=None, pool_size=None, test_size=None, seed=42)
    audit = export_dataset(fixture_dataset(), config, tmp_path, schema=SCHEMA)
    rows = json.loads((tmp_path / 'train.json').read_text())
    assert len(rows) == 1 and rows[0]['id'] == 'train:1'
    assert rows[0]['tokens'] == ['บริษัท', '_', 'A\\B', '\t']
    assert rows[0]['labels'] == ['B-ORG_COM', 'I-ORG_COM', 'O', 'O']
    docs, gold = load_local_split(tmp_path / 'test.json', SCHEMA, with_gold=True)
    assert docs[0].id == 'test:0' and gold == {'test:0': {(0, 1, 'PER')}}
    assert json.loads((tmp_path / 'test.json').read_text())[0]['labels'] == ['I-PER']
    assert 'labels' not in docs[0].public()
    assert audit['train_duplicates_removed'] == audit['train_test_overlap_removed'] == 1
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    export_dataset(fixture_dataset(), config, tmp_path, schema=SCHEMA)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('change', ['duplicate_id', 'length', 'label', 'language', 'empty'])
def test_invalid_local_files_are_rejected(tmp_path, change):
    row = dict(id='test:0', tokens=['John'], labels=['B-PER'], lang='en', source='fixture')
    rows = [row]
    if change == 'duplicate_id':
        rows.append(row.copy())
    elif change == 'length':
        row['labels'] = []
    elif change == 'label':
        row['labels'] = ['B-UNKNOWN']
    elif change == 'language':
        row['lang'] = 'invalid'
    else:
        rows = []
    path = tmp_path / 'test.json'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        load_local_split(path, SCHEMA, with_gold=True)


def test_pipeline_requires_explicit_data(tmp_path):
    from pipeline import run
    with pytest.raises(ValueError, match='Provide --data'):
        run(json.loads((ROOT / 'config.pilot.json').read_text()), tmp_path)


def test_schema_is_owned_by_dataset_and_exported(tmp_path):
    from data import load_schema
    custom = {'PER': {'definition': 'A person'}, 'ORG_COM': {'definition': 'A company'}}
    export_dataset(fixture_dataset(), dict(language=None, pool_size=None, test_size=None, seed=42),
                   tmp_path, schema=custom)
    assert load_schema(tmp_path) == custom
    assert json.loads((tmp_path / 'metadata.json').read_text())['schema_file'] == 'schema.json'
    with pytest.raises(ValueError, match='Cannot read dataset schema'):
        load_schema(tmp_path / 'missing')


def test_paired_datasets_must_have_matching_schemas(tmp_path):
    from pipeline import run
    train, test = tmp_path / 'train', tmp_path / 'test'
    train.mkdir()
    test.mkdir()
    (train / 'train.json').write_text('[]')
    (test / 'test.json').write_text('[]')
    (train / 'schema.json').write_text(json.dumps({'PER': 'Person'}))
    (test / 'schema.json').write_text(json.dumps({'ORG_COM': 'Company'}))
    config = json.loads((ROOT / 'config.pilot.json').read_text())
    with pytest.raises(ValueError, match='same schema'):
        run(config, tmp_path / 'run', stage='train', data_path=train / 'train.json',
            test_data_path=test / 'test.json')
