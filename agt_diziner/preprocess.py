"""Export cleaned, reproducible JSON splits before training."""
import argparse
import json
from pathlib import Path

from data import balanced_sample, filter_language, prepare, load_schema
from router import write_json
from conll import load_raw_data

HERE = Path(__file__).resolve().parent


def export_dataset(ds, config, output, provenance=None, *, schema):
    train, test, gold, audit = prepare(ds, schema)
    audit.update(provenance or {'source_format': 'in-memory'})
    train, test, gold = filter_language(train, test, gold, config.get('language'))
    if config.get('pool_size'):
        train = balanced_sample(train, min(config['pool_size'], len(train)), config['seed'])
    if config['test_size'] is not None:
        test = balanced_sample(test, config['test_size'], config['seed'] + 1)
    required_train = config.get('iterations', 0) * config.get('group_size', 0)
    if len(train) < required_train:
        raise ValueError(f'Pilot requires {required_train} training examples after cleaning; found {len(train)}. '
                         'Add raw data or reduce iterations/group_size in the config.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split, docs in [('train', train), ('test', test)]:
        rows = []
        for doc in docs:
            # IDs retain the original dataset row, even after filtering/sampling.
            original = ds[split][int(doc.id.split(':')[1])]
            labels = list(original['labels'])
            rows.append({**doc.public(), 'labels': labels})
        write_json(output / f'{split}.json', rows)
    audit.update(exported_documents={'train': len(train), 'test': len(test)},
                 preprocessing_config={k: config.get(k) for k in ('language', 'pool_size', 'test_size', 'seed')})
    audit.update(schema_file='schema.json', format_version=1)
    write_json(output / 'schema.json', schema)
    write_json(output / 'metadata.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.pilot.json')
    parser.add_argument('--output', type=Path, default=HERE / 'data' / 'pilot')
    parser.add_argument('--data', type=Path, default=HERE / 'data' / 'raw',
                        help='Directory containing train.conll and test.conll')
    parser.add_argument('--lang', choices=['en', 'th'],
                        help='Language for documents without metadata; defaults to config language')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        schema = load_schema(args.data)
        ds, provenance = load_raw_data(args.data, args.lang or config.get('language'))
        audit = export_dataset(ds, config, args.output, provenance, schema=schema)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Preprocessing complete: {audit['exported_documents']['train']} training documents, "
          f"{audit['exported_documents']['test']} test documents.")
    print(f"Training data saved to: {(args.output / 'train.json').resolve()}")
    print(f"Test data saved to: {(args.output / 'test.json').resolve()}")
    print(f"Data cleaning report: {(args.output / 'metadata.json').resolve()}")


if __name__ == '__main__':
    main()
