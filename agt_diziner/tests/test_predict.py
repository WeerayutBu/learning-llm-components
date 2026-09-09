import json
import sys

import pytest

from conll import read_conll
import predict


@pytest.mark.parametrize("batch", [False, True, "records"])
def test_predict_saves_conll_with_bio_labels(tmp_path, monkeypatch, batch):
    tokens = tmp_path / 'tokens.json'
    first = ['John', 'works', 'at', 'New', 'York', '.']
    sentences = [first, ['Facebook', '.']] if batch else [first]
    value = [{"tokens": sentence} for sentence in sentences] if batch == "records" else (sentences if batch else first)
    tokens.write_text(json.dumps(value))
    monkeypatch.setattr(sys, 'argv', ['predict.py', '--prompt', 'unused.json', '--tokens', str(tokens),
                                    '--lang', 'en', '--output', str(tmp_path / 'output')])
    monkeypatch.setattr(predict, 'load_prompt', lambda _: {'model': 'fixture', 'prompt_id': 'fixture'})
    class FakeRouter:
        def __init__(self, *args, **kwargs): pass
        def preflight(self, models): pass
        def close(self): pass
    monkeypatch.setattr(predict, 'Router', FakeRouter)
    def annotate(router, prompt, docs, purpose):
        assert [list(doc.tokens) for doc in docs] == sentences
        assert [doc.id for doc in docs] == [f'inference:{i}' for i in range(len(sentences))]
        return {'inference:0': {(0, 1, 'PER'), (3, 5, 'LOC_PLACE')}, 'inference:1': {(0, 1, 'ORG_COM')}}
    monkeypatch.setattr(predict, 'annotate_prompt', annotate)
    predict.main()
    rows = read_conll(tmp_path / 'output/predictions.conll')
    assert [row['tokens'] for row in rows] == sentences
    if batch:
        assert rows[1]['labels'] == ['B-ORG_COM', 'O']
    assert rows[0]['labels'] == ['B-PER', 'O', 'O', 'B-LOC_PLACE', 'I-LOC_PLACE', 'O']
    assert not (tmp_path / 'output/predictions.json').exists()


@pytest.mark.parametrize('value', [[], [[]], ['John', ['Google']], [['John'], [123]], [['']], [{}], [{'tokens': []}], [{'tokens': ['John'], 'labels': ['B-PER']}], [{'tokens': [1]}]])
def test_invalid_batches_fail_before_api(tmp_path, monkeypatch, value):
    tokens = tmp_path / 'tokens.json'
    tokens.write_text(json.dumps(value))
    monkeypatch.setattr(sys, 'argv', ['predict.py', '--prompt', 'unused.json', '--tokens', str(tokens), '--lang', 'en'])
    monkeypatch.setattr(predict, 'load_prompt', lambda _: {})
    monkeypatch.setattr(predict, 'Router', lambda *a, **k: pytest.fail('Invalid input must not call the API'))
    with pytest.raises(SystemExit) as exc:
        predict.main()
    assert exc.value.code == 2
