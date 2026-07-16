# EDITABLE — the inner loop rewrites this. SkyDiscover's initial_program.py.


def similarity(a: str, b: str) -> float:
    """How alike are two texts? 0.0 = unrelated, 1.0 = identical.

    It should rate paraphrases high, shrug off typos, and rate a negation LOW —
    "the door is open" and "the door is not open" are not the same claim.
    """
    if a == b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b), 1)
