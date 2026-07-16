"""PROTECTED — never enters a prompt. Karpathy's `prepare.py`.

An agent that can edit the verifier optimises the verifier: it deletes the failing
case instead of passing it.
"""

import json
import time
from pathlib import Path

_CASES = json.loads((Path(__file__).parent / "cases.json").read_text())
TESTS = [tuple(t) for t in _CASES["tests"]]   # (n, expected)
WORKLOAD = _CASES["workload"]


def measure(src: str) -> tuple[float | None, str]:
    """(seconds, note). None = rejected. Correct first, then fast — never the reverse."""
    ns: dict = {}
    try:
        exec(src, ns)
    except Exception as e:
        return None, f"exec error: {e}"

    f = ns.get("solve")
    if not callable(f):
        return None, "no solve()"

    for n, want in TESTS:
        try:
            got = f(n)
        except Exception as e:
            return None, f"raised on n={n}: {e}"
        if got != want:
            return None, f"WRONG on n={n}: got {got}, want {want}"

    t0 = time.perf_counter()
    f(WORKLOAD)
    return time.perf_counter() - t0, "ok"
