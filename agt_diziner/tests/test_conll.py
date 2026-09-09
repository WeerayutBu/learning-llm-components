import json
from pathlib import Path

import pytest

from conll import load_raw_data, read_conll
from preprocess import export_dataset


def test_columns_boundaries_unicode_and_final_sentence(tmp_path):
    path = tmp_path / 'train.conll'
    path.write_text('\ufeff-DOCSTART- -X- O O\n\nJohn NNP B-NP B-PER\nworks VB O O\n\n'
                    '# {"lang": "th", "source": "fixture"}\nบริษัท\tB-ORG_COM\n_\tI-ORG_COM', encoding='utf-8')
    rows = read_conll(path, 'en')
    assert len(rows) == 2
    assert rows[0]['tokens'] == ['John', 'works']
    assert rows[0]['labels'] == ['B-PER', 'O']
    assert rows[1] == dict(tokens=['บริษัท', '_'], labels=['B-ORG_COM', 'I-ORG_COM'], lang='th', source='fixture')


@pytest.mark.parametrize('text, error', [('John\tB-PER', 'language'), ('John', 'expected token'),
                                        ('John\tPERSON', 'Invalid BIO'), ('', 'no labeled')])
def test_actionable_invalid_raw_errors(tmp_path, text, error):
    path = tmp_path / 'test.conll'
    path.write_text(text)
    with pytest.raises(ValueError, match=error):
        read_conll(path)


def test_missing_split_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match='Add train.conll and test.conll'):
        load_raw_data(tmp_path, 'en')


def test_raw_to_json_cleaning_and_provenance(tmp_path):
    (tmp_path / 'train.conll').write_text('shared\tO\n\nJohn\tB-PER\n\nJohn\tB-PER\n')
    (tmp_path / 'test.conll').write_text('shared\tO\n')
    ds, provenance = load_raw_data(tmp_path, 'en')
    config = dict(language='en', pool_size=200, test_size=1, seed=42)
    output = tmp_path / 'processed'
    audit = export_dataset(ds, config, output, provenance, schema=json.loads((Path(__file__).resolve().parents[1] / 'data/raw/schema.json').read_text()))
    assert audit['train_duplicates_removed'] == audit['train_test_overlap_removed'] == 1
    assert audit['source_format'] == 'conll' and 'dataset' not in audit and 'revision' not in audit
    assert len(audit['raw_files']['train']['sha256']) == 64
    rows = json.loads((output / 'train.json').read_text())
    assert rows[0]['id'] == 'train:1' and rows[0]['tokens'] == ['John']
    assert {p.name for p in output.iterdir()} == {'train.json', 'test.json', 'schema.json', 'metadata.json'}
