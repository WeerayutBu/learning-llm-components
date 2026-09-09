"""Use a frozen selected configuration on JSON records containing token arrays."""
import argparse
import json
import sys
from pathlib import Path

from data import Document, spans_bio
from pipeline import annotate_prompt
from prompts import load_prompt
from router import Router, write_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", "--model", dest="model", type=Path, required=True,
                        help="Frozen baseline_model.json, improved_model.json, or selected_model.json")
    parser.add_argument("--tokens", type=Path, required=True, help='JSON array of records, each containing "tokens"')
    parser.add_argument("--lang", choices=["th", "en"], required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    args = parser.parse_args()
    if not args.tokens.is_file():
        parser.error(f"Token file not found: {args.tokens.resolve()}. "
                     'Create a JSON array such as [{"tokens": ["John", "works", "."]}] '
                     'or pass --tokens with an existing file path.')
    try:
        tokens = json.loads(args.tokens.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read token JSON at {args.tokens.resolve()}: {exc}")
    try:
        selected = load_prompt(args.model)
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot load prompt at {args.model.resolve()}: {exc}")
    if not isinstance(tokens, list) or not tokens:
        parser.error('--tokens must contain a nonempty array of records with a tokens field')
    if all(isinstance(row, dict) for row in tokens):
        if any(set(row) != {'tokens'} for row in tokens):
            parser.error('Each input record must contain only the tokens field')
        sentences = [row['tokens'] for row in tokens]
    else:
        sentences = [tokens] if all(isinstance(t, str) for t in tokens) else tokens
    docs = []
    for index, sentence in enumerate(sentences):
        if not isinstance(sentence, list) or not sentence or not all(isinstance(t, str) for t in sentence):
            parser.error(f'Sentence {index + 1} must be a nonempty array of token strings')
        if any(not token or any(c in token for c in '\t\r\n') for token in sentence):
            parser.error(f'Sentence {index + 1}: CoNLL tokens must be nonempty and contain no tabs or line breaks')
        docs.append(Document(f'inference:{index}', tuple(sentence), args.lang, 'user'))
    router = Router(args.output, max_requests=10, max_cost_usd=1.0)
    try:
        router.preflight([selected["model"]])
        predictions = annotate_prompt(router, selected, docs, "predict")
        blocks = []
        for doc in docs:
            labels = spans_bio(predictions[doc.id], len(doc.tokens))
            metadata = {'id': doc.id, 'lang': args.lang, 'model': selected['model'], 'prompt_id': selected['prompt_id']}
            block = '# ' + json.dumps(metadata, ensure_ascii=False) + '\n'
            block += ''.join(f'{token}\t{label}\n' for token, label in zip(doc.tokens, labels)) + '\n'
            blocks.append(block)
        result = ''.join(blocks)
        prediction_path = args.output / 'predictions.conll'
        write_text(prediction_path, result)
        print(result, end='')
        print(f"Saved predictions to: {prediction_path.resolve()}\n"
              "Open this file to see token/BIO-label rows, with a blank line between sentences.", file=sys.stderr, flush=True)
    finally:
        router.close()


if __name__ == "__main__":
    main()
