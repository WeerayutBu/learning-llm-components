"""DiZiNER §3.3 equations and strict entity micro-F1; no gold in analyze()."""
from collections import Counter
from itertools import combinations
import math

from data import spans_bio


def strict_score(predicted, gold):
    tp = len(predicted & gold)
    p, g = len(predicted), len(gold)
    return {"precision": tp / p if p else 0.0, "recall": tp / g if g else 0.0,
            "f1": 2 * tp / (p + g) if p + g else 0.0,
            "tp": tp, "fp": p - tp, "fn": g - tp}


def pooled(predictions):
    return {(doc_id, *span) for doc_id, spans in predictions.items() for span in spans}


def disagreement(probabilities):
    conflict = 1 - sum(p * p for p in probabilities.values())
    entity_mass = 1 - probabilities.get("O", 0.0)
    types = Counter()
    for tag, prob in probabilities.items():
        if tag != "O":
            types[tag[2:]] += prob
    confusion = 1 - sum((p / entity_mass) ** 2 for p in types.values()) if entity_mass > 1e-12 else 0.0
    starts = sum(p for t, p in probabilities.items() if t.startswith("B-"))
    inside = sum(p for t, p in probabilities.items() if t.startswith("I-"))
    boundary = max(4 * starts * (1 - starts), 4 * inside * (1 - inside))
    return {"label_conflict": max(0.0, conflict), "type_confusion": max(0.0, confusion),
            "boundary_uncertainty": max(0.0, boundary), "score": max(0.0, conflict, confusion, boundary)}


def error_kind(pred, consensus):
    if pred == consensus:
        return None
    if pred == "O":
        return "O→Ent"
    if consensus == "O":
        return "Ent→O"
    return "Ent→Ent" if pred[2:] != consensus[2:] else "Span Error"


def analyze(docs, predictions, hotspot_fraction=0.2):
    models = list(predictions)
    if len(models) < 2:
        raise ValueError("Agreement requires at least two independent models")
    ids = {d.id for d in docs}
    if any(set(predictions[m]) != ids for m in models):
        raise ValueError("Every model must annotate exactly the same documents")
    pairs, means = {}, {m: 0.0 for m in models}
    for a, b in combinations(models, 2):
        f1 = strict_score(pooled(predictions[a]), pooled(predictions[b]))["f1"]
        pairs[f"{a} | {b}"] = f1
        means[a] += f1 / (len(models) - 1)
        means[b] += f1 / (len(models) - 1)
    total = sum(means.values())
    weights = {m: means[m] / total if total else 1 / len(models) for m in models}
    elite, mass = [], 0.0
    for m in sorted(models, key=lambda m: (-weights[m], m)):
        elite.append(m)
        mass += weights[m]
        if mass >= 0.5 - 1e-12:
            break
    tokens, by_doc, errors = [], {}, {m: Counter() for m in models}
    for doc in docs:
        tags = {m: spans_bio(predictions[m][doc.id], len(doc.tokens)) for m in models}
        entries = []
        for i, token in enumerate(doc.tokens):
            votes = Counter()
            for m in models:
                votes[tags[m][i]] += weights[m]
            # Deterministic ties; prefer O, then lexical BIO tag order.
            consensus = min(votes, key=lambda tag: (-votes[tag], tag != "O", tag))
            entry = {"doc_id": doc.id, "index": i, "token": token,
                     "votes": dict(votes), "consensus": consensus,
                     "annotations": {m: tags[m][i] for m in models}, **disagreement(votes)}
            entries.append(entry)
            for m in models:
                kind = error_kind(tags[m][i], consensus)
                if kind:
                    errors[m][kind] += 1
        tokens.extend(entries)
        by_doc[doc.id] = entries
    # Rank all tokens as in §3.3, but don't invent hotspots from zero disagreement.
    quota = math.ceil(len(tokens) * hotspot_fraction)
    ranked = sorted(tokens, key=lambda t: (-t["score"], t["doc_id"], t["index"]))[:quota]
    chosen = {(t["doc_id"], t["index"]) for t in ranked if t["score"] > 1e-12}
    hotspots = []
    for doc in docs:
        entries, i = by_doc[doc.id], 0
        while i < len(entries):
            if (doc.id, i) not in chosen:
                i += 1
                continue
            start = i
            while i < len(entries) and (doc.id, i) in chosen:
                i += 1
            hotspots.append({"doc_id": doc.id, "lang": doc.lang, "start": start, "end": i,
                             "context_start": max(0, start - 5),
                             "context": doc.tokens[max(0, start - 5):min(len(entries), i + 5)],
                             "tokens": entries[start:i], "max_score": max(t["score"] for t in entries[start:i])})
    return {"pairwise_f1": pairs, "mean_agreement": means, "weights": weights, "elite": elite,
            "uniform_weight_fallback": total == 0, "tokens": tokens,
            "hotspots": hotspots, "errors_vs_consensus": {m: dict(v) for m, v in errors.items()},
            "token_count": len(tokens), "hotspot_token_count": len(chosen)}


def evaluate(predictions, gold, docs):
    ids = {d.id for d in docs}
    if set(predictions) != ids or not ids <= set(gold):
        raise ValueError("Evaluation requires one prediction and gold entry for every selected document")
    def score_subset(selected):
        return strict_score(pooled({i: predictions[i] for i in selected}), pooled({i: gold[i] for i in selected}))
    p, g = pooled(predictions), pooled({i: gold[i] for i in ids})
    return {"micro": score_subset(ids), "documents": len(ids),
            "by_language": {lang: score_subset({d.id for d in docs if d.lang == lang})
                            for lang in sorted({d.lang for d in docs})},
            "by_type": {label: strict_score({s for s in p if s[-1] == label}, {s for s in g if s[-1] == label})
                        for label in sorted({s[-1] for s in p | g})}}
