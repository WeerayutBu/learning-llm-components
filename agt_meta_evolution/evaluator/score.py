"""PROTECTED — never enters a prompt. SkyDiscover's `evaluator/`.

A searcher that can read the evaluator fits it; one that can edit it deletes the
case it fails. Neither loop ever sees this file.
"""

import json
from pathlib import Path

_CASES = json.loads((Path(__file__).parent / "cases.json").read_text())
PAIRS = [(c["a"], c["b"], c["human"]) for c in _CASES["pairs"]]
HUMAN = [h for _, _, h in PAIRS]


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def score(src: str) -> tuple[float | None, str]:
    """(score, note). None = rejected. Correlation with human judgement, -1..1.

    Deterministic: the same program always scores the same. Nothing here is timed.
    """
    ns: dict = {}
    try:
        exec(src, ns)
    except Exception as e:
        return None, f"exec error: {e}"

    f = ns.get("similarity")
    if not callable(f):
        return None, "no similarity()"

    got = []
    for a, b, _ in PAIRS:
        try:
            v = f(a, b)
        except Exception as e:
            return None, f"raised on {a[:24]!r}: {e}"
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 <= v <= 1.0:
            return None, f"not a similarity in [0,1]: {v!r}"
        got.append(float(v))

    if len(set(got)) == 1:                       # a constant correlates with nothing
        return None, f"constant {got[0]} for every pair"
    return _pearson(got, HUMAN), "ok"
