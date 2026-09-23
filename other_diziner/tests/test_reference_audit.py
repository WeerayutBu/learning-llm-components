"""Optional offline differential audit: set DIZINER_REFERENCE to the pinned checkout."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from data import Document, bio_spans
from metrics import analyze, disagreement, evaluate, strict_score
from pipeline import normalize_entities

PIN = '2577f8ce7f06f0554e88b1931f9e0c9626c24f17'
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def upstream():
    location = os.environ.get('DIZINER_REFERENCE')
    if not location:
        pytest.skip('Set DIZINER_REFERENCE to run the upstream differential audit')
    root = Path(location)
    assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    scope = {'np': np, 'cohen_kappa_score': cohen_kappa_score, 'Counter': Counter,
             'vprint': lambda *a, **k: None, 'DEFAULT_USE_BOUNDARY_VARIANT': True,
             'DEFAULT_COALITION_CUTOFF': 0.5, 'VERBOSE': 0}
    wanted = {
        'utils_disagreement.py': ['extract_spans_from_bio', 'calculate_strict_span_f1', 'is_B', 'is_I', 'start_prob', 'inside_prob',
                                 'calculate_cohen_kappa_bio', 'calculate_hybrid_weights', 'calculate_auto_weights'],
        'disagreement_analysis.py': ['D_bio', 'D_type', 'U_boundary', 'coalition_indices'],
        'utils_annotator.py': ['convert_entities_to_bio', 'convert_bio_to_entities'],
        'error_analysis.py': ['compute_majority_voting'],
    }
    # Execute only these pure functions, never the upstream module's API/config imports.
    for name, functions in wanted.items():
        source = (root / name).read_bytes()
        assert source == subprocess.check_output(['git', '-C', str(root), 'show', f'{PIN}:{name}'])
        tree = ast.parse(source)
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in functions]
        assert {n.name for n in nodes} == set(functions)
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(module, name, 'exec'), scope)
    return scope


def test_disagreement_equations_match_upstream_for_2000_distributions(upstream):
    rng = random.Random(42)
    tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
    cases = [{tag: 1.0} for tag in tags]
    for _ in range(2000):
        mass = [rng.random() for _ in tags]
        cases.append(dict(zip(tags, [v / sum(mass) for v in mass])))
    for votes in cases:
        types = Counter()
        for tag, probability in votes.items():
            types['O' if tag == 'O' else tag[2:]] += probability
        result = disagreement(votes)
        assert result['label_conflict'] == pytest.approx(upstream['D_bio'](votes))
        assert result['type_confusion'] == pytest.approx(upstream['D_type'](types))
        assert result['boundary_uncertainty'] == pytest.approx(upstream['U_boundary'](votes))


def test_overlap_assignment_matches_for_1000_random_cases(upstream):
    rng = random.Random(51)
    tokens = ('บริษัท', '_', 'IVL', 'rose', '.')
    doc = Document('test', tokens, 'th', 'fixture')
    starts, offset = [], 0
    for token in tokens:
        starts.append(offset)
        offset += len(token) + 1
    for _ in range(1000):
        entities = []
        for _ in range(rng.randrange(10)):
            start = rng.randrange(len(tokens))
            entities.append({'start': start, 'end': rng.randrange(start + 1, len(tokens) + 1),
                             'label': rng.choice(['PER', 'ORG'])})
        spans, normalized = normalize_entities({'entities': entities}, doc, {'PER': 'Person', 'ORG': 'Organization'})
        chars = [{'start_pos': starts[e['start']], 'end_pos': starts[e['end'] - 1] + len(tokens[e['end'] - 1]),
                  'type': e['label']} for e in entities]
        tags = upstream['convert_entities_to_bio'](list(tokens), chars, ' '.join(tokens), {'PER', 'ORG'})
        assert normalized['bio_labels'] == tags
        assert spans == upstream['extract_spans_from_bio'](tags)


def test_empty_entity_agreement_is_an_intentional_difference(upstream):
    assert upstream['calculate_strict_span_f1'](['O', 'O'], ['O', 'O']) == 1
    assert strict_score(set(), set())['f1'] == 0


def test_upstream_annotation_decoder_differs_from_local_decoder(upstream):
    assert upstream['convert_bio_to_entities'](['John'], ['I-PER']) == []
    assert bio_spans(['I-PER']) == {(0, 1, 'PER')}
    upstream_entities = upstream['convert_bio_to_entities'](['John', 'Google'], ['B-PER', 'I-ORG'])
    assert len(upstream_entities) == 1 and upstream_entities[0]['type'] == 'PER'
    assert bio_spans(['B-PER', 'I-ORG']) == {(0, 1, 'PER'), (1, 2, 'ORG')}


def test_hybrid_weights_differ_from_paper_span_only_weights(upstream):
    labels = {'a': ['O', 'O', 'O', 'B-ORG', 'B-PER', 'O'],
              'b': ['B-ORG', 'O', 'B-PER', 'O', 'B-PER', 'B-ORG'],
              'c': ['B-ORG', 'B-PER', 'B-ORG', 'B-PER', 'B-PER', 'B-ORG']}
    doc = Document('d', tuple('abcdef'), 'en', 'fixture')
    local = analyze([doc], {m: {'d': bio_spans(tags)} for m, tags in labels.items()})['weights']
    other = upstream['calculate_auto_weights'](labels)
    assert local == pytest.approx({'a': 0.2464788732394366, 'b': 0.39436619718309857, 'c': 0.35915492957746475})
    assert any(abs(local[m] - other[m]) > 1e-5 for m in local)


def test_percentile_ties_select_more_than_local_quota(upstream):
    doc = Document('d', ('a', 'b', 'c', 'd', 'e'), 'en', 'fixture')
    predictions = {'a': {'d': {(i, i + 1, 'PER') for i in range(5)}}, 'b': {'d': set()}}
    report = analyze([doc], predictions)
    scores = [token['score'] for token in report['tokens']]
    # Pinned extract_hotspot_blocks uses U_star >= global percentile threshold.
    assert sum(score >= np.percentile(scores, 80) for score in scores) == 5
    assert report['hotspot_token_count'] == 1


def test_saved_summary_recomputes_from_predictions():
    run = ROOT / 'runs/pilot-recovered'
    if not (run / 'summary.json').exists():
        pytest.skip('Recovered run is not available')
    summary = json.loads((run / 'summary.json').read_text())
    data = json.loads((ROOT / 'data/pilot/test.json').read_text())
    by_id = {row['id']: row for row in data}
    docs = [Document(i, tuple(by_id[i]['tokens']), by_id[i]['lang'], by_id[i]['source']) for i in summary['test_ids']]
    gold = {doc.id: bio_spans(by_id[doc.id]['labels']) for doc in docs}
    for choice in summary['results']:
        for id_key, metrics_key in [('prompt_id', 'metrics'), ('baseline_prompt_id', 'baseline')]:
            saved = json.loads((run / 'evaluation' / f"{choice[id_key]}.json").read_text())
            predictions = {i: {tuple(span) for span in spans} for i, spans in saved['predictions'][choice['model']].items()}
            assert evaluate(predictions, gold, docs) == choice[metrics_key]
    assert summary['mean_top_three_f1'] == pytest.approx(sum(r['metrics']['micro']['f1'] for r in summary['results']) / 3)
    report = json.loads((ROOT / 'docs/pilot-results.json').read_text())
    assert report['source_summary_sha256'] == hashlib.sha256((run / 'summary.json').read_bytes()).hexdigest()


def test_supervisor_consensus_differs_from_upstream_error_analysis(upstream):
    from types import SimpleNamespace
    labels = {'a': ['O', 'O', 'O', 'B-ORG', 'B-PER', 'O'],
              'b': ['B-ORG', 'O', 'B-PER', 'O', 'B-PER', 'B-ORG'],
              'c': ['B-ORG', 'B-PER', 'B-ORG', 'B-PER', 'B-PER', 'B-ORG']}
    doc = Document('d', tuple('abcdef'), 'en', 'fixture')
    local = analyze([doc], {m: {'d': bio_spans(tags)} for m, tags in labels.items()})
    analyzer = SimpleNamespace(results_by_model={m: {'detailed_results': [{'predicted_labels': tags}]}
                                                 for m, tags in labels.items()},
                               test_samples=[{'tokens': list(doc.tokens)}])
    upstream_votes = upstream['compute_majority_voting'](analyzer)[0]
    assert upstream_votes[3] == 'B-ORG'  # Unweighted three-way tie: first model wins.
    assert local['tokens'][3]['consensus'] == 'O'  # Model b has the highest weight.
