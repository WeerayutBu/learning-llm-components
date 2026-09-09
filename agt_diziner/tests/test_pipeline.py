from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from data import Document, balanced_sample, bio_spans, fingerprint, prepare, spans_bio
from metrics import analyze, disagreement, evaluate, strict_score
from pipeline import (initial_guidelines, pilot, refine, validate_entities, validate_guidelines)
from router import Router, parse_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / 'data/raw/schema.json').read_text())
DOC = Document('train:0', ('บริษัท', '_', 'ตัวอย่าง', 'หุ้น'), 'th', 'synthetic')


def test_bio_roundtrip_and_chunk_initial_inside():
    spans = {(0, 3, 'ORG_COM'), (3, 4, 'TICKER')}
    assert bio_spans(spans_bio(spans, 4)) == spans
    assert bio_spans(['I-PER', 'I-PER', 'O', 'I-ORG_COM']) == {(0, 2, 'PER'), (3, 4, 'ORG_COM')}
    assert bio_spans(['B-PER', 'B-PER']) == {(0, 1, 'PER'), (1, 2, 'PER')}


@pytest.mark.parametrize('entities', [
    [{'start': -1, 'end': 2, 'label': 'PER'}],
    [{'start': 0, 'end': 5, 'label': 'PER'}],
    [{'start': True, 'end': 2, 'label': 'PER'}],
    [{'start': 1, 'end': 1, 'label': 'PER'}],
    [{'start': 0, 'end': 1, 'label': 'UNKNOWN'}],
    [{'start': 0, 'end': 2, 'label': 'PER'}, {'start': 1, 'end': 3, 'label': 'ORG_COM'}],
])
def test_invalid_spans_are_rejected(entities):
    with pytest.raises(ValueError):
        validate_entities({'entities': entities}, DOC, SCHEMA)


def test_thai_indices_preserve_separators():
    response = {'entities': [{'start': 0, 'end': 3, 'label': 'ORG_COM'}]}
    validate_entities(response, DOC, SCHEMA)
    assert DOC.tokens[0:3] == ('บริษัท', '_', 'ตัวอย่าง')
    assert set(DOC.public()) == {'id', 'tokens', 'lang', 'source'}


def test_strict_metric_penalizes_boundary_type_and_document_errors():
    gold = {('a', 0, 2, 'PER'), ('b', 0, 1, 'ORG_COM')}
    assert strict_score({('a', 0, 1, 'PER'), ('b', 0, 1, 'PER')}, gold)['f1'] == 0
    score = strict_score({('a', 0, 2, 'PER')}, gold)
    assert score == {'precision': 1, 'recall': 0.5, 'f1': 2/3, 'tp': 1, 'fp': 0, 'fn': 1}
    assert strict_score(set(), set())['f1'] == 0  # No entity evidence ≠ perfect agreement.


def test_paper_disagreement_equations():
    d = disagreement({'B-PER': 0.5, 'I-PER': 0.5})
    assert d == {'label_conflict': 0.5, 'type_confusion': 0, 'boundary_uncertainty': 1, 'score': 1}
    d = disagreement({'B-PER': 0.5, 'B-ORG_COM': 0.5})
    assert d['label_conflict'] == d['type_confusion'] == 0.5
    assert d['boundary_uncertainty'] == 0
    assert disagreement({'O': 1})['score'] == 0


def test_weights_elite_and_global_hotspots():
    preds = {'a': {DOC.id: {(0, 1, 'PER'), (2, 3, 'ORG_COM')}},
             'b': {DOC.id: {(0, 1, 'PER')}}, 'c': {DOC.id: {(2, 3, 'ORG_COM')}}}
    report = analyze([DOC], preds, 0.5)
    assert report['weights'] == {'a': 0.5, 'b': 0.25, 'c': 0.25}
    assert report['elite'] == ['a']
    assert report['mean_agreement']['a'] == pytest.approx(2/3)
    assert [(h['start'], h['end']) for h in report['hotspots']] == [(0, 1), (2, 3)]
    assert report['hotspot_token_count'] == 2
    assert report['errors_vs_consensus']['b'] == {'O→Ent': 1}


def test_all_outside_not_a_perfect_candidate_and_no_fake_hotspots():
    report = analyze([DOC], {m: {DOC.id: set()} for m in ['a', 'b', 'c']})
    assert report['uniform_weight_fallback']
    assert not report['hotspots']
    assert all(s == 0 for s in report['mean_agreement'].values())


def test_missing_annotations_are_not_silently_dropped():
    with pytest.raises(ValueError, match='same documents'):
        analyze([DOC], {'a': {DOC.id: set()}, 'b': {}})
    with pytest.raises(ValueError, match='every selected'):
        evaluate({}, {DOC.id: set()}, [DOC])


def test_data_split_isolation_and_duplicate_removal():
    def row(tokens, labels, lang='th'):
        return {'tokens': tokens, 'labels': labels, 'lang': lang, 'source': 'fixture'}
    ds = {'train': [row(['shared'], ['O']), row(['train'], ['B-PER']), row(['train'], ['B-PER'])],
          'test': [row(['shared'], ['I-PER'])], 'validation': []}
    train, test, gold, audit = prepare(ds, SCHEMA)
    assert len(train) == len(test) == 1
    assert fingerprint(train[0].tokens) != fingerprint(test[0].tokens)
    assert audit['train_test_overlap_removed'] == audit['train_duplicates_removed'] == 1
    assert audit['test_orphan_I_tokens'] == 1
    assert gold == {'test:0': {(0, 1, 'PER')}}
    assert 'labels' not in train[0].public()


def test_evaluation_reports_languages_without_merging_document_spans():
    en = Document('test:1', ('Example',), 'en', 'fixture')
    pred = {DOC.id: {(0, 1, 'PER')}, en.id: set()}
    gold = {DOC.id: {(0, 1, 'PER')}, en.id: {(0, 1, 'PER')}}
    result = evaluate(pred, gold, [DOC, en])
    assert result['micro']['f1'] == 2/3
    assert result['by_language']['th']['f1'] == 1
    assert result['by_language']['en']['f1'] == 0


class FakeRouter:
    def __init__(self):
        self.calls = []

    def complete(self, model, prompt, validate, purpose):
        self.calls.append((model, prompt, purpose))
        if purpose.startswith('supervisor-phase1'):
            result = {'patterns': ['Missing names'], 'principles': ['Recognize named companies']}
        elif purpose.startswith('supervisor-phase2'):
            result = {'needs': ['Avoid missing explicit names']}
        elif purpose.startswith(('supervisor-phase3', 'supervisor-phase4')):
            result = initial_guidelines(['a', 'b', 'c'])
            result['common'].append('4. Identify explicitly named companies.')
        else:
            spans = {'a': [(0, 1, 'PER'), (2, 3, 'ORG_COM')],
                     'b': [(0, 1, 'PER')], 'c': [(2, 3, 'ORG_COM')]}[model]
            result = {'entities': [{'start': s, 'end': e, 'label': t} for s, e, t in spans]}
        validate(result)
        return result


def test_full_pilot_executes_all_four_phases_and_selects_without_gold(tmp_path):
    config = json.loads((ROOT / 'config.pilot.json').read_text())
    config.update(annotators=['a', 'b', 'c'], supervisor='supervisor', hotspot_fraction=0.5)
    router = FakeRouter()
    other = Document('train:1', ('Jane', 'met', 'Example', 'today'), 'en', 'fixture')
    selected = pilot(router, [[DOC], [other]], SCHEMA, config, tmp_path)
    assert selected[0]['model'] == 'a'
    assert len(selected) == 3
    purposes = [p for _, _, p in router.calls]
    assert 'supervisor-phase1' in purposes
    assert 'supervisor-phase2:b' in purposes and 'supervisor-phase2:c' in purposes
    assert 'supervisor-phase2:a' not in purposes  # Elite excluded from Phase 2.
    assert 'supervisor-phase3' in purposes and 'supervisor-phase4' in purposes
    assert (tmp_path / 'iteration_01' / 'guidelines.json').exists()
    assert not any('test:' in prompt for _, prompt, _ in router.calls)
    snapshot = json.loads((tmp_path / 'selected_model.json').read_text())
    assert snapshot['schema'] == SCHEMA


def test_guideline_schema_and_model_ids_cannot_drift():
    config = json.loads((ROOT / 'config.pilot.json').read_text())
    old = initial_guidelines(['a', 'b', 'c'])
    changed = deepcopy(old)
    changed['schema'] = {'NEW_TYPE': 'not allowed'}
    with pytest.raises(ValueError):
        validate_guidelines(changed, ['a', 'b', 'c'], old, config)
    changed = deepcopy(old)
    del changed['model_specific']['b']
    with pytest.raises(ValueError):
        validate_guidelines(changed, ['a', 'b', 'c'], old, config)


def test_balanced_sampling_is_seeded_and_does_not_inspect_gold():
    docs = [Document(str(i), (str(i),), 'th' if i % 2 else 'en', 'fixture') for i in range(20)]
    assert balanced_sample(docs, 4, 42) == balanced_sample(docs, 4, 42)
    assert {d.lang for d in balanced_sample(docs, 2, 42)} == {'th', 'en'}


def test_json_parser_does_not_salvage_truncated_or_prose_results():
    assert parse_json('```json\n{"entities": []}\n```') == {'entities': []}
    for raw in ['{"entities": [', 'explanation {"entities": []}', '[]']:
        with pytest.raises(ValueError):
            parse_json(raw)


def make_router(tmp_path, monkeypatch, handler, **kwargs):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic-test-key')
    router = Router(tmp_path, **kwargs)
    router.http.close()
    router.http = httpx.Client(base_url='https://openrouter.ai/api/v1/', transport=httpx.MockTransport(handler))
    router.catalog = {'a': {'pricing': {'prompt': '0.000001', 'completion': '0.000001'},
                            'supported_parameters': ['temperature']}}
    return router


def test_client_retries_invalid_output_and_resumes_cache_without_spending(tmp_path, monkeypatch):
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json={'choices': [{'finish_reason': 'length' if count == 1 else 'stop',
                                                      'message': {'content': '{"entities": []}'}}],
                                       'usage': {'cost': 0.001}})
    router = make_router(tmp_path, monkeypatch, handler)
    validate = lambda r: validate_entities(r, DOC, SCHEMA)
    assert router.complete('a', 'prompt', validate, 'annotation') == {'entities': []}
    assert count == 2
    assert router.complete('a', 'prompt', validate, 'annotation') == {'entities': []}
    assert count == 2 and router.cache_hits == 1
    assert router.usage()['reported_cost_usd'] == 0.002
    assert 'synthetic-test-key' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json*'))
    router.close()


def test_budget_blocks_request_before_transport(tmp_path, monkeypatch):
    def handler(request):
        raise AssertionError('Must not send an over-budget request')
    router = make_router(tmp_path, monkeypatch, handler, max_cost_usd=0.000001)
    with pytest.raises(RuntimeError, match='Cost budget'):
        router.complete('a', 'prompt', lambda x: None, 'test')
    router.close()


def test_train_then_test_reuses_frozen_prompts_and_split(tmp_path, monkeypatch):
    import pipeline
    from prompts import load_prompt, render_prompt

    config = json.loads((ROOT / 'config.pilot.json').read_text())
    config.update(annotators=['a', 'b', 'c'], supervisor='supervisor', test_size=1, hotspot_fraction=0.5)
    test_doc = Document('test:0', ('Jane', 'met', 'Example', 'today'), 'en', 'fixture')
    gold = {test_doc.id: {(0, 1, 'PER'), (2, 3, 'ORG_COM')}}
    write_local_fixture(tmp_path, [DOC], [test_doc], gold)
    monkeypatch.setattr(pipeline, 'diverse_groups', lambda *a: [[DOC], [DOC]])
    routers = []

    class StageRouter(FakeRouter):
        def __init__(self, *args):
            super().__init__()
            routers.append(self)

        def preflight(self, models):
            self.models = models

        def usage(self):
            return {'requests': len(self.calls)}

        def close(self):
            pass

    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(
        load_dataset=lambda *a, **k: pytest.fail('Pipeline must use local data')))
    monkeypatch.setattr(pipeline, 'Router', StageRouter)
    selected = pipeline.run(config, tmp_path / 'run', stage='train', data_path=tmp_path / 'data/train.json')
    assert not any(purpose.startswith('test') for _, _, purpose in routers[0].calls)
    assert not (tmp_path / 'run' / 'summary.json').exists()
    baseline = load_prompt(tmp_path / 'run' / 'baseline_model.json')
    improved = load_prompt(tmp_path / 'run' / 'improved_model.json')
    assert baseline['model'] == improved['model'] == 'a'
    assert baseline['iteration'] == 0 and improved['iteration'] == 1
    assert baseline['prompt_id'] != improved['prompt_id']
    assert selected[0]['iteration'] == 0  # Paper selection includes the baseline.
    frozen_manifest = (tmp_path / 'run' / 'manifest.json').read_bytes()
    frozen_split = (tmp_path / 'run' / 'split_manifest.json').read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError('Test-only mode must reuse training artifacts')

    (tmp_path / 'data/train.json').unlink()  # Test-only mode needs only its explicit test file.
    monkeypatch.setattr(pipeline, 'pilot', forbidden)
    monkeypatch.setattr(pipeline, 'initial_guidelines', forbidden)
    monkeypatch.setattr(pipeline, 'diverse_groups', forbidden)
    summary = pipeline.run(config, tmp_path / 'run', stage='test', data_path=tmp_path / 'data/test.json')
    calls = routers[-1].calls
    assert all(purpose.startswith('test') for _, _, purpose in calls)
    assert ('a', render_prompt(baseline, test_doc), 'test-baseline:test:0') in calls
    assert ('a', render_prompt(improved, test_doc), 'test-selected:test:0') in calls
    assert summary['prompt_comparison']['prompt_id'] == improved['prompt_id']
    assert summary['prompt_comparison']['baseline_prompt_id'] == baseline['prompt_id']
    assert summary['prompt_comparison']['metrics']['micro']['f1'] == 1
    assert (tmp_path / 'run' / 'verification.json').exists()
    assert (tmp_path / 'run' / 'manifest.json').read_bytes() == frozen_manifest
    assert (tmp_path / 'run' / 'split_manifest.json').read_bytes() == frozen_split
    saved_test = (tmp_path / 'data/test.json').read_bytes()
    (tmp_path / 'data/test.json').write_bytes(saved_test + b' ')
    with pytest.raises(ValueError, match='data'):
        pipeline.run(config, tmp_path / 'run', stage='test', data_path=tmp_path / 'data/test.json')
    (tmp_path / 'data/test.json').write_bytes(saved_test)
    evaluation = json.loads(next((tmp_path / 'run/evaluation').glob('*.json')).read_text())
    assert evaluation['documents'][0]['tokens'] == list(test_doc.tokens)
    # Frozen aliases cannot be edited and silently tested.
    (tmp_path / 'run' / 'improved_model.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        pipeline.run(config, tmp_path / 'run', stage='test', data_path=tmp_path / 'data/test.json')


def test_test_stage_requires_training_artifacts(tmp_path):
    from pipeline import run
    config = json.loads((ROOT / 'config.pilot.json').read_text())
    with pytest.raises(ValueError, match='run --stage train'):
        run(config, tmp_path, stage='test', data_path=tmp_path / 'data/test.json')
    assert not (tmp_path / 'manifest.json').exists()


def test_repeated_invalid_response_gets_three_real_attempts_then_recovers(tmp_path, monkeypatch):
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json={'choices': [{'finish_reason': 'length' if count <= 3 else 'stop',
                                                      'message': {'content': '{}'}}], 'usage': {'cost': 0.001}})
    router = make_router(tmp_path, monkeypatch, handler, attempts=3)
    with pytest.raises(ValueError, match='after 3 attempts'):
        router.complete('a', 'prompt', lambda x: None, 'test')
    assert count == 3 and not list((tmp_path / 'cache').glob('*.json'))
    router.close()
    resumed = make_router(tmp_path, monkeypatch, handler, attempts=3)
    assert resumed.complete('a', 'prompt', lambda x: None, 'test') == {}
    assert count == 4
    assert resumed.complete('a', 'prompt', lambda x: None, 'test') == {}
    assert count == 4 and resumed.usage()['requests'] == 4
    resumed.close()


def test_malformed_api_envelope_is_accounted_and_retried(tmp_path, monkeypatch):
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        if count <= 2:
            return httpx.Response(200, text='broken JSON')
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})
    router = make_router(tmp_path, monkeypatch, handler)
    assert router.complete('a', 'prompt', lambda x: None, 'test') == {}
    assert router.usage()['requests'] == 3
    assert router.usage()['budget_accounted_usd'] > 0
    assert len(list((tmp_path / 'responses').glob('*.json'))) == 3
    router.close()


def test_interrupted_request_stays_reserved_and_blocks_overspend(tmp_path, monkeypatch):
    def interrupted(request):
        # Check disk BEFORE transport returns, simulating a kill during an in-flight request.
        records = [json.loads(p.read_text()) for p in (tmp_path / 'requests').glob('*.json')]
        assert len(records) == 1 and records[0]['status'] == 'pending'
        raise KeyboardInterrupt
    router = make_router(tmp_path, monkeypatch, interrupted, max_requests=1)
    with pytest.raises(KeyboardInterrupt):
        router.complete('a', 'prompt', lambda x: None, 'test')
    router.close()
    def forbidden(request):
        raise AssertionError('Pending request must count against the budget')
    resumed = make_router(tmp_path, monkeypatch, forbidden, max_requests=1)
    assert resumed.usage()['pending_requests'] == 1
    with pytest.raises(RuntimeError, match='Request budget'):
        resumed.complete('a', 'prompt', lambda x: None, 'test')
    resumed.close()


def test_changed_code_is_rejected_without_overwriting_manifest(tmp_path, monkeypatch):
    import pipeline
    config = json.loads((ROOT / 'config.pilot.json').read_text())
    config.update(test_size=1)
    write_local_fixture(tmp_path, [DOC], [Document('test:0', ('other',), 'th', 'fixture')], {'test:0': set()})
    monkeypatch.setattr(pipeline, 'diverse_groups', lambda *a: [[DOC]])
    pipeline.run(config, tmp_path / 'run', prepare_only=True, data_path=tmp_path / 'data/train.json')
    original = (tmp_path / 'run' / 'manifest.json').read_bytes()
    monkeypatch.setattr(pipeline, 'source_hashes', lambda: {'pipeline.py': 'changed'})
    with pytest.raises(ValueError, match='source'):
        pipeline.run(config, tmp_path / 'run', prepare_only=True, data_path=tmp_path / 'data/train.json')
    assert (tmp_path / 'run' / 'manifest.json').read_bytes() == original


def test_prompt_rendering_preserves_dollars_braces_thai_and_detects_edits():
    from prompts import make_prompt, render_prompt, validate_prompt
    guidelines = initial_guidelines(['a'])
    guidelines['common'].append('Keep literal $language and {braces}.')
    artifact = make_prompt('a', SCHEMA, guidelines, 0)
    doc = Document('new', ('บริษัท', '$indexed_tokens', '{x}', '_'), 'th', 'fixture')
    rendered = render_prompt(artifact, doc)
    assert 'Keep literal $language and {braces}.' in rendered
    assert '$indexed_tokens' in rendered and 'บริษัท' in rendered
    artifact['template'] += 'modified'
    with pytest.raises(ValueError, match='modified'):
        validate_prompt(artifact)


def test_no_refinement_does_not_export_a_fake_improved_prompt(tmp_path):
    config = json.loads((ROOT / 'config.pilot.json').read_text())
    config.update(annotators=['a', 'b', 'c'])
    pilot(FakeRouter(), [[DOC]], SCHEMA, config, tmp_path)
    assert json.loads((tmp_path / 'prompt_comparison.json').read_text())['improved'] is None
    assert (tmp_path / 'baseline_model.json').exists()
    assert not (tmp_path / 'improved_model.json').exists()


def test_overlap_retry_supplies_conflicting_spans_and_previous_answer(tmp_path, monkeypatch):
    doc = Document('train:11082', ('What', 'are', 'your', 'thoughts', 'and', 'opinions', 'on',
                   'the', 'Metaverse', 'FB', ',', 'AAPL', 'and', 'some', 'other', 'companies'),
                   'en', 'fixture')
    rejected = {'entities': [{'start': 8, 'end': 16, 'label': 'ORG_COM'},
                             {'start': 9, 'end': 11, 'label': 'TICKER'},
                             {'start': 11, 'end': 13, 'label': 'TICKER'}]}
    repaired = {'entities': [{'start': 9, 'end': 10, 'label': 'TICKER'},
                             {'start': 11, 'end': 12, 'label': 'TICKER'}]}
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 2:
            assert [m['role'] for m in body['messages']] == ['user', 'assistant', 'user']
            assert json.loads(body['messages'][1]['content']) == rejected
            correction = body['messages'][2]['content']
            assert 'ORG_COM [8,16)' in correction
            assert 'TICKER [9,11)' in correction
            assert 'FB' in correction and 'complete corrected entities list' in correction
        result = rejected if len(bodies) == 1 else repaired
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                     'message': {'content': json.dumps(result)}}]})
    router = make_router(tmp_path, monkeypatch, handler)
    validate = lambda result: validate_entities(result, doc, SCHEMA)
    assert router.complete('a', 'original task', validate, 'annotation') == repaired
    assert len(bodies) == 2
    assert router.complete('a', 'original task', validate, 'annotation') == repaired
    assert len(bodies) == 2
    assert bodies[1]['messages'][0]['content'] == 'original task'
    router.close()


def test_overlap_validation_still_rejects_nested_and_duplicate_spans():
    nested = {'entities': [{'start': 0, 'end': 3, 'label': 'ORG_COM'},
                           {'start': 1, 'end': 2, 'label': 'TICKER'}]}
    with pytest.raises(ValueError, match=r'entities\[0\].*conflicts with entities\[1\]'):
        validate_entities(nested, DOC, SCHEMA)
    duplicate = {'entities': [{'start': 0, 'end': 1, 'label': 'PER'}] * 2}
    with pytest.raises(ValueError, match='Overlapping or duplicate'):
        validate_entities(duplicate, DOC, SCHEMA)


def test_default_retry_budget_can_repair_after_three_rejections(tmp_path, monkeypatch):
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        content = '{"entities": [{"start": 0, "end": 1, "label": "PER"}, {"start": 0, "end": 1, "label": "PER"}]}' if count <= 3 else '{"entities": []}'
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': content}}], 'usage': {'cost': 0.001}})
    router = make_router(tmp_path, monkeypatch, handler)
    assert router.complete('a', 'prompt', lambda r: validate_entities(r, DOC, SCHEMA), 'annotation') == {'entities': []}
    assert count == 4
    assert router.usage()['requests'] == 4
    router.close()


def test_unknown_label_feedback_names_rejected_and_allowed_labels():
    with pytest.raises(ValueError, match="Unknown entity label") as error:
        validate_entities({'entities': [{'start': 0, 'end': 1, 'label': 'PERSON'}]}, DOC, SCHEMA)
    assert "'PERSON'" in str(error.value)
    assert 'entities[0]' in str(error.value)
    assert all(label in str(error.value) for label in SCHEMA)


def test_invalid_index_feedback_identifies_span_and_token_convention():
    with pytest.raises(ValueError, match="Indices must satisfy") as error:
        validate_entities({'entities': [{'start': 99, 'end': 120, 'label': 'PER'}]}, DOC, SCHEMA)
    text = str(error.value)
    assert 'entities[0]' in text and 'start=99' in text and 'end=120' in text
    assert 'TOKEN indices' in text and 'not character offsets' in text
    assert 'do not merely clamp' in text


@pytest.mark.parametrize('language', ['en', 'th'])
def test_language_pilot_filters_before_grouping_and_preserves_ids(tmp_path, monkeypatch, language):
    import pipeline
    config = json.loads((ROOT / f'config.pilot.{language}.json').read_text())
    config.update(iterations=1, group_size=1, test_size=1, pool_size=2, supervisor='supervisor')
    def row(text, lang):
        return {'tokens': [text], 'labels': ['O'], 'lang': lang, 'source': 'fixture'}
    ds = {'train': [row('English train', 'en'), row('Thai train', 'th')],
          'test': [row('English test', 'en'), row('Thai test', 'th')], 'validation': []}
    from preprocess import export_dataset
    export_dataset(ds, config, tmp_path / 'data', schema=SCHEMA)
    grouped = []
    def grouping(docs, *args):
        assert {d.lang for d in docs} == {language}
        grouped.append([d.id for d in docs])
        return [docs]
    monkeypatch.setattr(pipeline, 'diverse_groups', grouping)
    monkeypatch.setattr(pipeline, 'Router', lambda *a, **k: pytest.fail('Preparation must not call the API'))
    audit = pipeline.run(config, tmp_path / 'run', prepare_only=True, data_path=tmp_path / 'data/train.json')
    index = 0 if language == 'en' else 1
    split = json.loads((tmp_path / 'run' / 'split_manifest.json').read_text())
    assert split == {'pilot_groups': [[f'train:{index}']], 'test_ids': [f'test:{index}']}
    assert audit['experiment_scope'] == {'language': language, 'train_pool': 1, 'test_pool': 1}
    pipeline.run(config, tmp_path / 'run', prepare_only=True, data_path=tmp_path / 'data/train.json')
    assert len(grouped) == 1  # Resume reuses the frozen language-specific IDs.
    config['language'] = 'th' if language == 'en' else 'en'
    with pytest.raises(ValueError, match='changed'):
        pipeline.run(config, tmp_path / 'run', prepare_only=True, data_path=tmp_path / 'data/train.json')


def test_language_filter_limits_gold_and_keeps_global_overlap_protection():
    from data import filter_language
    def row(text, lang):
        return {'tokens': [text], 'labels': ['B-PER'], 'lang': lang, 'source': 'fixture'}
    ds = {'train': [row('shared', 'en'), row('unique en', 'en'), row('unique th', 'th')],
          'test': [row('shared', 'th'), row('heldout en', 'en')], 'validation': []}
    train, test, gold, audit = prepare(ds, SCHEMA)
    en_train, en_test, en_gold = filter_language(train, test, gold, 'en')
    assert audit['train_test_overlap_removed'] == 1
    assert [d.id for d in en_train] == ['train:1']
    assert [d.id for d in en_test] == ['test:1']
    assert set(en_gold) == {'test:1'}
    assert filter_language(train, test, gold) == (train, test, gold)
    with pytest.raises(ValueError, match='language must'):
        filter_language(train, test, gold, 'fr')
    with pytest.raises(ValueError, match='nonempty'):
        filter_language(en_train, en_test, en_gold, 'th')


@pytest.mark.parametrize('entities, expected', [
    ([{'start': 0, 'end': 3, 'label': 'ORG_COM'}, {'start': 1, 'end': 2, 'label': 'TICKER'}],
     {(0, 1, 'ORG_COM'), (1, 2, 'TICKER'), (2, 3, 'ORG_COM')}),
    ([{'start': 1, 'end': 2, 'label': 'TICKER'}, {'start': 0, 'end': 3, 'label': 'ORG_COM'}],
     {(0, 3, 'ORG_COM')}),
    ([{'start': 0, 'end': 1, 'label': 'PER'}] * 2, {(0, 1, 'PER')}),
    ([{'start': 0, 'end': 1, 'label': 'PER'}, {'start': 2, 'end': 3, 'label': 'ORG_COM'}],
     {(0, 1, 'PER'), (2, 3, 'ORG_COM')}),
])
def test_upstream_overlap_normalization_preserves_input_order(entities, expected):
    from pipeline import normalize_entities
    value = {'entities': entities}
    original = deepcopy(value)
    spans, report = normalize_entities(value, DOC, SCHEMA)
    assert spans == expected
    assert value == original  # The auditable raw response is never mutated.
    spans_bio(spans, len(DOC.tokens))  # Consensus accepts the flattened result.


def test_reported_nvidia_overlap_continues_in_one_call_and_records_raw_output(tmp_path, monkeypatch):
    from pipeline import annotate_prompt
    from prompts import make_prompt
    doc = Document('train:17955', tuple(['prefix'] * 8 + ['NVIDIA', ',', 'Square', 'and', 'Roku',
                                                        'dominating', 'their', 'markets']), 'en', 'fixture')
    raw = {'entities': [{'start': 8, 'end': 13, 'label': 'ORG_COM'},
                        {'start': 10, 'end': 16, 'label': 'ORG_COM'}]}
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                       'message': {'content': json.dumps(raw)}}], 'usage': {'cost': 0.001}})
    router = make_router(tmp_path, monkeypatch, handler)
    artifact = make_prompt('a', SCHEMA, initial_guidelines(['a']), 0)
    expected = {doc.id: {(8, 10, 'ORG_COM'), (10, 16, 'ORG_COM')}}
    assert annotate_prompt(router, artifact, [doc], 'pilot-1') == expected
    assert annotate_prompt(router, artifact, [doc], 'pilot-1') == expected
    assert len(calls) == 1 and router.cache_hits == 1
    record = json.loads(next((tmp_path / 'normalization').rglob('*.json')).read_text())
    assert record['raw_entities'] == raw['entities']
    assert record['overwritten_token_indices'] == [10, 11, 12]
    assert record['normalized_entities'] == [[8, 10, 'ORG_COM'], [10, 16, 'ORG_COM']]
    # Flattening does not manufacture the semantically correct three company mentions.
    assert expected[doc.id] != {(8, 9, 'ORG_COM'), (10, 11, 'ORG_COM'), (12, 13, 'ORG_COM')}
    cached = json.loads(next((tmp_path / 'cache').glob('*.json')).read_text())
    assert json.loads(cached['response']['choices'][0]['message']['content']) == raw
    router.close()


def test_normalization_still_rejects_unknown_labels_and_out_of_range_spans():
    from pipeline import normalize_entities
    for entity in [{'start': 0, 'end': 1, 'label': 'UNKNOWN'},
                   {'start': 0, 'end': 99, 'label': 'PER'}]:
        with pytest.raises(ValueError):
            normalize_entities({'entities': [entity]}, DOC, SCHEMA)


def write_local_fixture(folder, train, test, gold):
    from router import write_json
    write_json(folder / 'data/schema.json', SCHEMA)
    for split, docs in [('train', train), ('test', test)]:
        write_json(folder / 'data' / f'{split}.json', [
            {**d.public(), 'labels': spans_bio(gold.get(d.id, set()), len(d.tokens))}
            for d in docs])


def test_repeated_empty_span_uses_fresh_repair_and_never_accepts_invalid_cache(tmp_path, monkeypatch):
    doc = Document('train:0', ('IVL', 'rose'), 'en', 'fixture')
    bad = {'entities': [{'start': 0, 'end': 0, 'label': 'TICKER'}]}
    good = {'entities': [{'start': 0, 'end': 1, 'label': 'TICKER'}]}
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 2:
            assert [m['role'] for m in body['messages']] == ['user', 'assistant', 'user']
            assert "Token 0 is 'IVL'" in body['messages'][-1]['content']
            assert 'end=1' in body['messages'][-1]['content']
        if len(bodies) == 3:
            assert len(body['messages']) == 1
            assert 'Annotate the input afresh' in body['messages'][0]['content']
            assert 'ORIGINAL TASK' in body['messages'][0]['content']
            assert not list((tmp_path / 'cache').glob('*.json'))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(good if len(bodies) == 3 else bad)}}], 'usage': {'cost': 0.001}})
    router = make_router(tmp_path, monkeypatch, handler)
    validate = lambda value: validate_entities(value, doc, SCHEMA)
    assert router.complete('a', 'ORIGINAL TASK', validate, 'annotation') == good
    assert router.complete('a', 'ORIGINAL TASK', validate, 'annotation') == good
    assert len(bodies) == 3 and router.usage()['requests'] == 3
    router.close()
